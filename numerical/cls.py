import argparse, json, os, random
from typing import List
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

from synthetic_data import SynthConfig, generate_synthetic, FeatureDataset
from ot_policy import apply_ot_policy_per_class
from metrics import topk_accuracy, worstq_accuracy
import sdro_core_min as core
from util import AdapterEncoder

def build_dataloaders(Xtr, ytr, Xte, yte, batch_size=256, seed=0):
    train_ds = FeatureDataset(Xtr, ytr)
    test_ds  = FeatureDataset(Xte, yte)
    g = torch.Generator(); g.manual_seed(seed)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False))

def compute_support_tensors(Xtest, ytest, support_idx_by_class, device):
    idx_all = []; support_class_indices = []
    for idxs in support_idx_by_class: idx_all.extend(idxs)
    Xsup_all = Xtest[idx_all].to(device) if len(idx_all)>0 else torch.empty(0, Xtest.size(1), device=device)
    ptr=0
    for idxs in support_idx_by_class:
        support_class_indices.append(list(range(ptr, ptr+len(idxs)))); ptr+=len(idxs)
    assert ptr == Xsup_all.size(0)
    return Xsup_all, support_class_indices

def prepare_priors(enc, head, Xtrain, ytrain, Xsup_all, support_class_indices, K, eps_sample, eps_class, metric_diag, device, cov_inflation=3.0):
    with torch.no_grad():
        mtr = enc(Xtrain.to(device))
    base_by_class = [(mtr[(ytrain==c).nonzero(as_tuple=False).view(-1)]).detach() for c in range(K)]
    base_stats = core.compute_base_stats(base_by_class, ridge=1e-4)
    with torch.no_grad():
        zsup = enc(Xsup_all) if Xsup_all.numel()>0 else torch.empty(0, base_by_class[0].size(1), device=device)
    C = core.low_level_softmin_cost(zsup, base_by_class, eps_sample, metric_diag)
    a = torch.full((len(base_by_class),), 1.0 / len(base_by_class), device=device)
    b = torch.full((Xsup_all.size(0),),   1.0 / max(1,Xsup_all.size(0)), device=device) if Xsup_all.numel()>0 else torch.tensor([], device=device)
    T = core.sinkhorn_log(C, eps_class=eps_class, a=a, b=b, iters=200) if Xsup_all.numel()>0 else torch.zeros(C.size(0), 0, device=device)
    w_bc = core.class_weights_from_transport(T, support_class_indices) if Xsup_all.numel()>0 else torch.full((len(base_by_class), K), 1.0/len(base_by_class), device=device)
    priors = core.build_priors_from_HOT(base_stats, w_bc, cov_inflation=cov_inflation)
    return priors

def train_epoch_erm(enc, head, loader, opt, device):
    enc.train(); head.train(); loss_sum=0.0; n=0
    for xb,yb in loader:
        xb,yb=xb.to(device), yb.to(device)
        logits=head(enc(xb))
        loss=F.cross_entropy(logits,yb)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item()*xb.size(0); n+=xb.size(0)
    return loss_sum/max(1,n)

def train_epoch_dro(model, loader, priors, opt, newton_steps, device):
    model.train(); loss_sum=0.0; n=0
    for xb,yb in loader:
        xb,yb=xb.to(device), yb.to(device)
        m=model.encoder(xb); ell=core.robust_logits_from_priors(model,m,priors,newton_steps=newton_steps)
        loss=F.cross_entropy(ell,yb)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters()),1.0); opt.step()
        loss_sum += loss.item()*xb.size(0); n+=xb.size(0)
    return loss_sum/max(1,n)

@torch.no_grad()
def evaluate(model_or_head, enc, loader, priors, use_dro, newton_steps, device, worst_q=0.10):
    enc.eval()
    if isinstance(model_or_head, nn.Module) and hasattr(model_or_head, "encoder"):
        model=model_or_head
    else:
        class _W(nn.Module):
            def __init__(self, enc, head):
                super().__init__(); self.encoder=enc; self.head=head
                self.eps=0.5; self.rho=1.0; self.metric_raw=nn.Parameter(torch.zeros(enc.d if hasattr(enc,'d') else loader.dataset.X.size(1)), requires_grad=False)
            def metric_diag(self): return torch.ones_like(self.metric_raw)+1e-6
        model=_W(enc, model_or_head)
    all_logits, all_y = [], []
    for xb,yb in loader:
        xb,yb=xb.to(device), yb.to(device)
        m=model.encoder(xb)
        logits = core.robust_logits_from_priors(model,m,priors,newton_steps=newton_steps) if use_dro else model_or_head(m)
        all_logits.append(logits.cpu()); all_y.append(yb.cpu())
    logits=torch.cat(all_logits,0); y=torch.cat(all_y,0)
    return topk_accuracy(logits,y,1), worstq_accuracy(logits,y,q=worst_q)

def ema_blend_priors(old, new, tau: float = 0.3):
    if old is None or tau<=0.0: return new
    out=new
    for k in dir(new):
        if k.startswith("_"): continue
        v_new=getattr(new,k,None); v_old=getattr(old,k,None)
        if isinstance(v_new, torch.Tensor) and isinstance(v_old, torch.Tensor) and v_new.shape==v_old.shape:
            setattr(out,k,(1.0-tau)*v_old+tau*v_new)
    return out

STAGE1_EPOCHS=5; FREEZE_PRIORS=5; REFRESH_INTERVAL=3; PRIOR_EMA_TAU=0.3

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--K",type=int,default=8)
    p.add_argument("--d_a",type=int,default=6); p.add_argument("--d_b",type=int,default=4)
    p.add_argument("--n_train",type=int,default=6000); p.add_argument("--n_test",type=int,default=3000)
    p.add_argument("--alpha_test",type=float,default=0.2)
    p.add_argument("--support_min",type=int,default=1); p.add_argument("--support_max",type=int,default=5)
    p.add_argument("--ot_reg",type=float,default=0.2)
    p.add_argument("--eps_sample",type=float,default=1.0); p.add_argument("--eps_class",type=float,default=0.8); p.add_argument("--rho",type=float,default=0.6)
    p.add_argument("--newton_steps",type=int,default=8)
    p.add_argument("--shift_a_mean",type=float,default=0.6); p.add_argument("--shift_a_cov_scale",type=float,default=1.15); p.add_argument("--rotate_a_deg",type=float,default=15.0)
    p.add_argument("--worstq",type=float,default=0.10)
    p.add_argument("--hidden",type=int,default=256); p.add_argument("--epochs",type=int,default=30); p.add_argument("--batch_size",type=int,default=256); p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--device",type=str,default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--prior_temp", type=float, default=0.1)
    p.add_argument("--prec_max", type=float, default=10.0)
    args=p.parse_args()

    cfg=SynthConfig(seed=args.seed,K=args.K,d_a=args.d_a,d_b=args.d_b,n_train=args.n_train,n_test=args.n_test,
                    alpha_test=args.alpha_test,support_min=args.support_min,support_max=args.support_max,
                    shift_a_mean=args.shift_a_mean,shift_a_cov_scale=args.shift_a_cov_scale,rotate_a_deg=args.rotate_a_deg)
    data=generate_synthetic(cfg); d=data.X_train.size(1); K=args.K; device=torch.device(args.device)

    tr_loader, te_loader = build_dataloaders(data.X_train,data.y_train,data.X_test,data.y_test,batch_size=args.batch_size,seed=args.seed)

    enc_erm=AdapterEncoder(d=d,hidden=args.hidden,ln=True).to(device); head_erm=nn.Linear(d,K).to(device)
    enc_erm_ot=AdapterEncoder(d=d,hidden=args.hidden,ln=True).to(device); head_erm_ot=nn.Linear(d,K).to(device)
    enc_dro=AdapterEncoder(d=d,hidden=args.hidden,ln=True).to(device); head_dro=nn.Linear(d,K).to(device)

    Xsup_all, support_class_indices = compute_support_tensors(data.X_test,data.y_test,data.support_idx_by_class,device)
    Ahat_tr = apply_ot_policy_per_class(data.a_train.to(device), data.y_train.to(device),
                                        Xsup_all[:, :args.d_a] if Xsup_all.numel()>0 else torch.empty(0,args.d_a,device=device),
                                        support_class_indices, reg=args.ot_reg, metric_diag=torch.ones(args.d_a,device=device))
    Xtr_ot = torch.cat([Ahat_tr, data.b_train.to(device)], dim=1)

    opt_erm=torch.optim.Adam(list(enc_erm.parameters())+list(head_erm.parameters()), lr=args.lr)
    opt_erm_ot=torch.optim.Adam(list(enc_erm_ot.parameters())+list(head_erm_ot.parameters()), lr=args.lr)
    opt_dro=torch.optim.Adam(list(enc_dro.parameters())+list(head_dro.parameters()), lr=args.lr, weight_decay=1e-4)

    dro_model=core.PGSDRO(enc_dro, head_dro, d=d, eps=args.eps_sample, rho=args.rho).to(device)
    dro_model.prior_temp = args.prior_temp
    dro_model.prec_max   = args.prec_max

    metric_diag=torch.ones(d,device=device); metric_diag[:args.d_a]=4.0

    for epoch in range(1, STAGE1_EPOCHS+1):
        loss_erm=train_epoch_erm(enc_erm,head_erm,tr_loader,opt_erm,device)
        erm2_loader,_=build_dataloaders(Xtr_ot.cpu(),data.y_train,data.X_test,data.y_test,batch_size=args.batch_size,seed=args.seed)
        loss_erm_ot=train_epoch_erm(enc_erm_ot,head_erm_ot,erm2_loader,opt_erm_ot,device)
        with torch.no_grad():
            enc_erm.eval(); enc_erm_ot.eval()
            acc_c1,wacc_c1=evaluate(head_erm,enc_erm,te_loader,None,False,args.newton_steps,device,worst_q=args.worstq)
            acc_c2,wacc_c2=evaluate(head_erm_ot,enc_erm_ot,te_loader,None,False,args.newton_steps,device,worst_q=args.worstq)
        print(f"[Warm {epoch:02d}] ERM loss={loss_erm:.3f} || Acc C1={acc_c1:.3f} C2={acc_c2:.3f} | Worst{int(args.worstq*100)}% C1={wacc_c1:.3f} C2={wacc_c2:.3f}")

    dro_model.encoder.load_state_dict(enc_erm_ot.state_dict()); dro_model.head.load_state_dict(head_erm_ot.state_dict())

    print("Preparing initial priors for R2...")
    priors = prepare_priors(dro_model.encoder.eval(), dro_model.head, Xtr_ot, data.y_train.to(device),
                            Xsup_all, support_class_indices, K, args.eps_sample, args.eps_class, metric_diag, device, cov_inflation=3.0)
    acc_hist_c1, acc_hist_c2, acc_hist_r2 = [], [], []
    wacc_hist_c1, wacc_hist_c2, wacc_hist_r2 = [], [], []

    for epoch in range(STAGE1_EPOCHS+1, args.epochs+1):
        loss_erm=train_epoch_erm(enc_erm,head_erm,tr_loader,opt_erm,device)
        erm2_loader,_=build_dataloaders(Xtr_ot.cpu(),data.y_train,data.X_test,data.y_test,batch_size=args.batch_size,seed=args.seed)
        loss_erm_ot=train_epoch_erm(enc_erm_ot,head_erm_ot,erm2_loader,opt_erm_ot,device)

        dro_model.eps=args.eps_sample
        rho0,rho1=0.2,args.rho
        t=(epoch-(STAGE1_EPOCHS+1))/max(1,(args.epochs-(STAGE1_EPOCHS+1)))
        dro_model.rho=rho0 + t*(rho1-rho0)

        loss_dro=train_epoch_dro(dro_model,erm2_loader,priors,opt_dro,args.newton_steps,device)

        do_refresh = (epoch >= STAGE1_EPOCHS + FREEZE_PRIORS) and (((epoch - (STAGE1_EPOCHS + FREEZE_PRIORS)) % REFRESH_INTERVAL) == 0)
        if do_refresh:
            new_priors = prepare_priors(dro_model.encoder.eval(), dro_model.head, Xtr_ot, data.y_train.to(device),
                                        Xsup_all, support_class_indices, K, args.eps_sample, args.eps_class, metric_diag, device, cov_inflation=3.0)
            priors = ema_blend_priors(priors, new_priors, tau=0.3)

        with torch.no_grad():
            enc_erm.eval(); head_erm.eval(); enc_erm_ot.eval(); head_erm_ot.eval()
            acc_c1,wacc_c1=evaluate(head_erm,enc_erm,te_loader,None,False,args.newton_steps,device,worst_q=args.worstq)
            acc_c2,wacc_c2=evaluate(head_erm_ot,enc_erm_ot,te_loader,None,False,args.newton_steps,device,worst_q=args.worstq)
            acc_r2,wacc_r2=evaluate(dro_model,dro_model.encoder,te_loader,priors,True,args.newton_steps,device,worst_q=args.worstq)
        acc_hist_c1.append(float(acc_c1))
        acc_hist_c2.append(float(acc_c2))
        acc_hist_r2.append(float(acc_r2))
        wacc_hist_c1.append(float(wacc_c1))
        wacc_hist_c2.append(float(wacc_c2))
        wacc_hist_r2.append(float(wacc_r2))
        print(f"[Epoch {epoch:02d}] ERM loss={loss_erm:.3f} | DRO loss={loss_dro:.3f} || Acc C1={acc_c1:.3f}  C2={acc_c2:.3f}  R2={acc_r2:.3f} | Worst{int(args.worstq*100)}% C1={wacc_c1:.3f}  C2={wacc_c2:.3f}  R2={wacc_r2:.3f}")

    import numpy as np

    def last_k_stats(hist, k=5):
        n = min(k, len(hist))
        arr = np.array(hist[-n:], dtype=float)
        mean = float(arr.mean())
        std  = float(arr.std(ddof=0))
        return n, mean, std

    n1, c1_mean, c1_std = last_k_stats(acc_hist_c1, k=5)
    n2, c2_mean, c2_std = last_k_stats(acc_hist_c2, k=5)
    n3, r2_mean, r2_std = last_k_stats(acc_hist_r2, k=5)

    nw1, wc1_mean, wc1_std = last_k_stats(wacc_hist_c1, k=5)
    nw2, wc2_mean, wc2_std = last_k_stats(wacc_hist_c2, k=5)
    nw3, wr2_mean, wr2_std = last_k_stats(wacc_hist_r2, k=5)

    print(f"[Acc last {min(n1,n2,n3)} epochs] "
        f"C1 mean={c1_mean:.3f}±{c1_std:.3f} | "
        f"C2 mean={c2_mean:.3f}±{c2_std:.3f} | "
        f"R2 mean={r2_mean:.3f}±{r2_std:.3f}")
    print(f"[Worst{int(args.worstq*100)}% last {min(nw1,nw2,nw3)} epochs] "
        f"C1 mean={wc1_mean:.3f}±{wc1_std:.3f} | "
        f"C2 mean={wc2_mean:.3f}±{wc2_std:.3f} | "
        f"R2 mean={wr2_mean:.3f}±{wr2_std:.3f}")

    out = dict(
        acc_c1=float(acc_c1), acc_c2=float(acc_c2), acc_r2=float(acc_r2),
        wacc_c1=float(wacc_c1), wacc_c2=float(wacc_c2), wacc_r2=float(wacc_r2),
        worstq_acc_c1=float(wacc_c1), worstq_acc_c2=float(wacc_c2), worstq_acc_r2=float(wacc_r2),
        last5_acc_summary=dict(
            n=min(n1,n2,n3),
            C1=dict(mean=c1_mean, std=c1_std),
            C2=dict(mean=c2_mean, std=c2_std),
            R2=dict(mean=r2_mean, std=r2_std),
        ),
        last5_worstq_summary=dict(
            n=min(nw1,nw2,nw3),
            C1=dict(mean=wc1_mean, std=wc1_std),
            C2=dict(mean=wc2_mean, std=wc2_std),
            R2=dict(mean=wr2_mean, std=wr2_std),
        ),
        cfg=vars(cfg), args=vars(args)
    )

    with open("synth_summary.json","w") as f: import json; json.dump(out,f,indent=2); print("Saved synth_summary.json")

if __name__=="__main__":
    main()
