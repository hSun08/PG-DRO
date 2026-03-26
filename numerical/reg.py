import argparse, json
from typing import List, Tuple

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from synthetic_data import SynthConfig, generate_synthetic
from ot_policy import apply_ot_policy_per_class
import sdro_core_min as core
from util import AdapterEncoder


class FeatureDatasetFloat(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X.float()
        self.y = y.float()
    def __len__(self): return self.X.size(0)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def build_dataloaders(Xtr, ytr, Xte, yte, batch_size=256, seed=0):
    train_ds = FeatureDatasetFloat(Xtr, ytr)
    test_ds  = FeatureDatasetFloat(Xte, yte)
    g = torch.Generator(); g.manual_seed(seed)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  generator=g, num_workers=0),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False, generator=g, num_workers=0))


def compute_support_tensors(Xtest, ytest, support_idx_by_class, device):
    idx_all = []; support_class_indices = []
    for idxs in support_idx_by_class: idx_all.extend(idxs)
    Xsup_all = Xtest[idx_all].to(device) if len(idx_all)>0 else torch.empty(0, Xtest.size(1), device=device)
    ptr=0
    for idxs in support_idx_by_class:
        support_class_indices.append(list(range(ptr, ptr+len(idxs)))); ptr+=len(idxs)
    assert ptr == Xsup_all.size(0)
    return Xsup_all, support_class_indices


@torch.no_grad()
def prepare_priors_a(Xa_train: torch.Tensor, y_train: torch.Tensor,
                     Xa_sup: torch.Tensor, support_class_indices, K: int,
                     eps_sample: float, eps_class: float, device: torch.device,
                     cov_inflation: float = 3.0):
    base_by_class = [(Xa_train[(y_train==c).nonzero(as_tuple=False).view(-1)]).detach()
                     for c in range(K)]
    base_stats = core.compute_base_stats(base_by_class, ridge=1e-4)
    d_a = Xa_train.size(1)
    metric_diag = torch.ones(d_a, device=device)
    C = core.low_level_softmin_cost(Xa_sup, base_by_class, eps_sample, metric_diag)
    a = torch.full((K,), 1.0/K, device=device)
    b = torch.full((Xa_sup.size(0),), 1.0/max(1,Xa_sup.size(0)), device=device) if Xa_sup.numel()>0 else torch.tensor([], device=device)
    T = core.sinkhorn_log(C, eps_class=eps_class, a=a, b=b, iters=200) if Xa_sup.numel()>0 else torch.zeros(C.size(0), 0, device=device)
    w_bc = core.class_weights_from_transport(T, support_class_indices) if Xa_sup.numel()>0 else torch.full((K, K), 1.0/K, device=device)
    priors_a = core.build_priors_from_HOT(base_stats, w_bc, cov_inflation=cov_inflation)
    return priors_a


def prior_penalty_on_a(xb: torch.Tensor, d_a: int, priors_a,
                       temp: float = 0.1, reduction: str = 'mean', margin: float = -1.0):
    a = xb[:, :d_a]
    mu  = priors_a.mu_prior
    var = priors_a.var_prior.clamp_min(1e-6)
    pr  = (1.0/var).clamp(max=10.0)
    x  = a.unsqueeze(1)
    mu = mu.unsqueeze(0)
    pr = pr.unsqueeze(0)
    quad = -0.5 * ((x-mu)*(x-mu)*pr).sum(dim=2)
    s = quad.max(dim=1).values
    pen = temp * torch.relu(margin - s)
    return (pen.mean() if reduction=='mean' else pen), s


huber = nn.SmoothL1Loss(beta=1.0)

def train_epoch_erm_reg(enc, head, loader, opt, device):
    enc.train(); head.train(); loss_sum=0.0; n=0
    for xb,yb in loader:
        xb,yb=xb.to(device), yb.to(device).float()
        pred=head(enc(xb)).squeeze(-1)
        loss=huber(pred,yb)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item()*xb.size(0); n+=xb.size(0)
    return loss_sum/max(1,n)


def train_epoch_dro_reg(model, loader, priors_a, opt, device,
                        prior_temp=0.1, lmbda=1.0, d_a: int = 0):
    model.train(); loss_sum=0.0; n=0
    for xb,yb in loader:
        xb,yb=xb.to(device), yb.to(device).float()
        m=model.encoder(xb)
        pred=model.head(m).squeeze(-1)
        loss_pred=huber(pred,yb)
        pen,_=prior_penalty_on_a(xb, d_a=d_a, priors_a=priors_a, temp=prior_temp, reduction='mean')
        loss=loss_pred + lmbda*pen
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()),1.0)
        opt.step()
        loss_sum += loss.item()*xb.size(0); n+=xb.size(0)
    return loss_sum/max(1,n)


@torch.no_grad()
def evaluate_reg(model_or_head, enc, loader, priors_a, device, worst_q=0.10, d_a: int = 0):
    enc.eval()
    if isinstance(model_or_head, nn.Module) and hasattr(model_or_head, "encoder"):
        model=model_or_head
    else:
        class _W(nn.Module):
            def __init__(self, enc, head):
                super().__init__(); self.encoder=enc; self.head=head
        model=_W(enc, model_or_head)
    preds, ys, diffs = [], [], []
    for xb,yb in loader:
        xb=xb.to(device); yb=yb.to(device).float()
        m=model.encoder(xb)
        pred=model.head(m).squeeze(-1)
        _, s = prior_penalty_on_a(xb, d_a=d_a, priors_a=priors_a, temp=0.0, reduction='none', margin=-2.0)
        diff = (-s)
        preds.append(pred.cpu()); ys.append(yb.cpu()); diffs.append(diff.cpu())
    pred = torch.cat(preds); y = torch.cat(ys); diff = torch.cat(diffs)
    rmse = torch.mean((pred - y)**2).sqrt().item()
    mae  = torch.mean(torch.abs(pred - y)).item()
    k = max(1, int(len(y)*worst_q))
    worst_idx = torch.topk(diff, k=k).indices
    wrmse = torch.mean((pred[worst_idx] - y[worst_idx])**2).sqrt().item()
    return rmse, mae, wrmse


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

class DROModel(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head    = head

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--K",type=int,default=8)
    p.add_argument("--d_a",type=int,default=6); p.add_argument("--d_b",type=int,default=4)
    p.add_argument("--n_train",type=int,default=6000); p.add_argument("--n_test",type=int,default=3000)
    p.add_argument("--alpha_test",type=float,default=0.2)
    p.add_argument("--support_min",type=int,default=1); p.add_argument("--support_max",type=int,default=5)
    p.add_argument("--ot_reg",type=float,default=0.2)
    p.add_argument("--eps_sample",type=float,default=1.0); p.add_argument("--eps_class",type=float,default=0.8)
    p.add_argument("--prior_temp", type=float, default=0.1)
    p.add_argument("--lmbda_penalty", type=float, default=1.0)
    p.add_argument("--shift_a_mean",type=float,default=0.6)
    p.add_argument("--shift_a_cov_scale",type=float,default=1.15)
    p.add_argument("--rotate_a_deg",type=float,default=15.0)
    p.add_argument("--reg_noise", type=float, default=0.5)
    p.add_argument("--worstq",type=float,default=0.10)
    p.add_argument("--hidden",type=int,default=256); p.add_argument("--epochs",type=int,default=30); p.add_argument("--batch_size",type=int,default=256); p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--device",type=str,default="cuda" if torch.cuda.is_available() else "cpu")
    args=p.parse_args()
    device=torch.device(args.device)
    cfg=SynthConfig(
        seed=args.seed, K=args.K, d_a=args.d_a, d_b=args.d_b,
        n_train=args.n_train, n_test=args.n_test,
        alpha_test=args.alpha_test, support_min=args.support_min, support_max=args.support_max,
        shift_a_mean=args.shift_a_mean, shift_a_cov_scale=args.shift_a_cov_scale, rotate_a_deg=args.rotate_a_deg
    )
    data=generate_synthetic(cfg)
    d=data.X_train.size(1); K=args.K
    Xa_tr = data.a_train.to(device); Xb_tr = data.b_train.to(device)
    Xa_te = data.a_test.to(device);  Xb_te = data.b_test.to(device)
    Xtr   = data.X_train.to(device); Xte   = data.X_test.to(device)
    g1, g2 = np.random.RandomState(args.seed), np.random.RandomState(args.seed+1)
    w_a_t = torch.from_numpy(g1.normal(size=args.d_a).astype(np.float32)).to(device)
    w_b_t = torch.from_numpy(g2.normal(size=args.d_b).astype(np.float32)).to(device)
    noise_tr = torch.randn(Xa_tr.size(0), device=device) * args.reg_noise
    noise_te = torch.randn(Xa_te.size(0), device=device) * args.reg_noise
    y_train_reg = (Xa_tr @ w_a_t + Xb_tr @ w_b_t + noise_tr).float()
    y_test_reg  = (Xa_te @ w_a_t + Xb_te @ w_b_t + noise_te).float()
    tr_loader, te_loader = build_dataloaders(Xtr, y_train_reg, Xte, y_test_reg,
                                             batch_size=args.batch_size, seed=args.seed)
    enc_erm=AdapterEncoder(d=d,hidden=args.hidden,ln=True).to(device); head_erm=nn.Linear(d,1).to(device)
    enc_erm_ot=AdapterEncoder(d=d,hidden=args.hidden,ln=True).to(device); head_erm_ot=nn.Linear(d,1).to(device)
    enc_dro=AdapterEncoder(d=d,hidden=args.hidden,ln=True).to(device); head_dro=nn.Linear(d,1).to(device)
    Xsup_all, support_class_indices = compute_support_tensors(Xte, data.y_test, data.support_idx_by_class, device)
    Xa_sup = Xsup_all[:, :args.d_a] if Xsup_all.numel()>0 else torch.empty(0,args.d_a,device=device)
    Ahat_tr = apply_ot_policy_per_class(
        Xa_tr, data.y_train.to(device),
        Xa_sup,
        support_class_indices, reg=args.ot_reg, metric_diag=torch.ones(args.d_a,device=device)
    )
    Xtr_ot = torch.cat([Ahat_tr, Xb_tr], dim=1)
    g_noise = torch.Generator(device=device).manual_seed(args.seed + 123)
    noise_tr_ot = torch.randn(Xa_tr.size(0), generator=g_noise, device=device) * args.reg_noise
    y_train_reg_ot = (Ahat_tr @ w_a_t + Xb_tr @ w_b_t + noise_tr_ot).float()
    opt_erm=torch.optim.Adam(list(enc_erm.parameters())+list(head_erm.parameters()), lr=args.lr)
    opt_erm_ot=torch.optim.Adam(list(enc_erm_ot.parameters())+list(head_erm_ot.parameters()), lr=args.lr)
    opt_dro=torch.optim.Adam(list(enc_dro.parameters())+list(head_dro.parameters()), lr=args.lr, weight_decay=1e-4)
    dro_model=DROModel(enc_dro, head_dro).to(device)
    priors_a = prepare_priors_a(
        Xa_train=Xa_tr, y_train=data.y_train.to(device),
        Xa_sup=Xa_sup, support_class_indices=support_class_indices, K=K,
        eps_sample=args.eps_sample, eps_class=args.eps_class,
        device=device, cov_inflation=3.0
    )
    for epoch in range(1, STAGE1_EPOCHS+1):
        loss_erm = train_epoch_erm_reg(enc_erm, head_erm, tr_loader, opt_erm, device)
        erm2_loader,_ = build_dataloaders(Xtr_ot, y_train_reg_ot, Xte, y_test_reg,
                                          batch_size=args.batch_size, seed=args.seed)
        loss_erm_ot = train_epoch_erm_reg(enc_erm_ot, head_erm_ot, erm2_loader, opt_erm_ot, device)
        with torch.no_grad():
            enc_erm.eval(); enc_erm_ot.eval()
            rmse_c1, mae_c1, wrmse_c1 = evaluate_reg(head_erm,   enc_erm,   te_loader, priors_a, device, worst_q=args.worstq, d_a=args.d_a)
            rmse_c2, mae_c2, wrmse_c2 = evaluate_reg(head_erm_ot, enc_erm_ot, te_loader, priors_a, device, worst_q=args.worstq, d_a=args.d_a)
        print(f"[Warm {epoch:02d}] ERM loss={loss_erm:.3f} || "
              f"C1 RMSE={rmse_c1:.3f} MAE={mae_c1:.3f} w{int(args.worstq*100)}%RMSE={wrmse_c1:.3f} | "
              f"C2 RMSE={rmse_c2:.3f} MAE={mae_c2:.3f} w{int(args.worstq*100)}%RMSE={wrmse_c2:.3f}")
    dro_model.encoder.load_state_dict(enc_erm_ot.state_dict())
    dro_model.head.load_state_dict(head_erm_ot.state_dict())
    rmse_hist_c1, rmse_hist_c2, rmse_hist_r2 = [], [], []
    mae_hist_c1,  mae_hist_c2,  mae_hist_r2  = [], [], []
    wrmse_hist_c1, wrmse_hist_c2, wrmse_hist_r2 = [], [], []
    for epoch in range(STAGE1_EPOCHS+1, args.epochs+1):
        loss_erm = train_epoch_erm_reg(enc_erm, head_erm, tr_loader, opt_erm, device)
        erm2_loader,_ = build_dataloaders(Xtr_ot, y_train_reg_ot, Xte, y_test_reg,
                                          batch_size=args.batch_size, seed=args.seed)
        loss_erm_ot = train_epoch_erm_reg(enc_erm_ot, head_erm_ot, erm2_loader, opt_erm_ot, device)
        loss_dro = train_epoch_dro_reg(dro_model, erm2_loader, priors_a, opt_dro, device,
                                       prior_temp=args.prior_temp, lmbda=args.lmbda_penalty, d_a=args.d_a)
        do_refresh = (epoch >= STAGE1_EPOCHS + FREEZE_PRIORS) and (((epoch - (STAGE1_EPOCHS + FREEZE_PRIORS)) % REFRESH_INTERVAL) == 0)
        if do_refresh:
            priors_new = prepare_priors_a(
                Xa_train=Ahat_tr, y_train=data.y_train.to(device),
                Xa_sup=Xa_sup, support_class_indices=support_class_indices, K=K,
                eps_sample=args.eps_sample, eps_class=args.eps_class,
                device=device, cov_inflation=3.0
            )
            priors_a = ema_blend_priors(priors_a, priors_new, tau=PRIOR_EMA_TAU)
        with torch.no_grad():
            enc_erm.eval(); head_erm.eval(); enc_erm_ot.eval(); head_erm_ot.eval()
            rmse_c1, mae_c1, wrmse_c1 = evaluate_reg(head_erm,   enc_erm,   te_loader, priors_a, device, worst_q=args.worstq, d_a=args.d_a)
            rmse_c2, mae_c2, wrmse_c2 = evaluate_reg(head_erm_ot, enc_erm_ot, te_loader, priors_a, device, worst_q=args.worstq, d_a=args.d_a)
            rmse_r2, mae_r2, wrmse_r2 = evaluate_reg(dro_model,  dro_model.encoder, te_loader, priors_a, device, worst_q=args.worstq, d_a=args.d_a)
        rmse_hist_c1.append(float(rmse_c1))
        rmse_hist_c2.append(float(rmse_c2))
        rmse_hist_r2.append(float(rmse_r2))
        mae_hist_c1.append(float(mae_c1))
        mae_hist_c2.append(float(mae_c2))
        mae_hist_r2.append(float(mae_r2))
        wrmse_hist_c1.append(float(wrmse_c1))
        wrmse_hist_c2.append(float(wrmse_c2))
        wrmse_hist_r2.append(float(wrmse_r2))
        print(f"[Epoch {epoch:02d}] ERM loss={loss_erm:.3f} | DRO loss={loss_dro:.3f} || "
              f"C1 RMSE={rmse_c1:.3f} MAE={mae_c1:.3f} w{int(args.worstq*100)}%RMSE={wrmse_c1:.3f} | "
              f"C2 RMSE={rmse_c2:.3f} MAE={mae_c2:.3f} w{int(args.worstq*100)}%RMSE={wrmse_c2:.3f} | "
              f"R2 RMSE={rmse_r2:.3f} MAE={mae_r2:.3f} w{int(args.worstq*100)}%RMSE={wrmse_r2:.3f}")
    def last_k_stats(hist, k=5):
        n = min(k, len(hist)); arr = np.array(hist[-n:], dtype=float)
        return n, float(arr.mean()), float(arr.std(ddof=0))
    n1, c1_mean, c1_std = last_k_stats(rmse_hist_c1, k=5)
    n2, c2_mean, c2_std = last_k_stats(rmse_hist_c2, k=5)
    n3, r2_mean, r2_std = last_k_stats(rmse_hist_r2, k=5)
    nmae1, mae_c1_mean, mae_c1_std = last_k_stats(mae_hist_c1, k=5)
    nmae2, mae_c2_mean, mae_c2_std = last_k_stats(mae_hist_c2, k=5)
    nmae3, mae_r2_mean, mae_r2_std = last_k_stats(mae_hist_r2, k=5)
    nwr1, wrmse_c1_mean, wrmse_c1_std = last_k_stats(wrmse_hist_c1, k=5)
    nwr2, wrmse_c2_mean, wrmse_c2_std = last_k_stats(wrmse_hist_c2, k=5)
    nwr3, wrmse_r2_mean, wrmse_r2_std = last_k_stats(wrmse_hist_r2, k=5)
    print(f"[RMSE last {min(n1,n2,n3)} epochs] "
          f"C1 mean={c1_mean:.3f}±{c1_std:.3f} | "
          f"C2 mean={c2_mean:.3f}±{c2_std:.3f} | "
          f"R2 mean={r2_mean:.3f}±{r2_std:.3f}")
    print(f"[MAE last {min(nmae1,nmae2,nmae3)} epochs] "
          f"C1 mean={mae_c1_mean:.3f}±{mae_c1_std:.3f} | "
          f"C2 mean={mae_c2_mean:.3f}±{mae_c2_std:.3f} | "
          f"R2 mean={mae_r2_mean:.3f}±{mae_r2_std:.3f}")
    print(f"[Worst{int(args.worstq*100)}% RMSE last {min(nwr1,nwr2,nwr3)} epochs] "
          f"C1 mean={wrmse_c1_mean:.3f}±{wrmse_c1_std:.3f} | "
          f"C2 mean={wrmse_c2_mean:.3f}±{wrmse_c2_std:.3f} | "
          f"R2 mean={wrmse_r2_mean:.3f}±{wrmse_r2_std:.3f}")
    out = dict(
        rmse_c1=float(rmse_hist_c1[-1]) if rmse_hist_c1 else None,
        rmse_c2=float(rmse_hist_c2[-1]) if rmse_hist_c2 else None,
        rmse_r2=float(rmse_hist_r2[-1]) if rmse_hist_r2 else None,
        mae_c1=float(mae_hist_c1[-1]) if mae_hist_c1 else None,
        mae_c2=float(mae_hist_c2[-1]) if mae_hist_c2 else None,
        mae_r2=float(mae_hist_r2[-1]) if mae_hist_r2 else None,
        wrmse_c1=float(wrmse_hist_c1[-1]) if wrmse_hist_c1 else None,
        wrmse_c2=float(wrmse_hist_c2[-1]) if wrmse_hist_c2 else None,
        wrmse_r2=float(wrmse_hist_r2[-1]) if wrmse_hist_r2 else None,
        worstq_rmse_c1=float(wrmse_hist_c1[-1]) if wrmse_hist_c1 else None,
        worstq_rmse_c2=float(wrmse_hist_c2[-1]) if wrmse_hist_c2 else None,
        worstq_rmse_r2=float(wrmse_hist_r2[-1]) if wrmse_hist_r2 else None,
        last5_rmse_summary=dict(
            n=min(n1,n2,n3),
            C1=dict(mean=c1_mean, std=c1_std),
            C2=dict(mean=c2_mean, std=c2_std),
            R2=dict(mean=r2_mean, std=r2_std),
        ),
        last5_mae_summary=dict(
            n=min(nmae1,nmae2,nmae3),
            C1=dict(mean=mae_c1_mean, std=mae_c1_std),
            C2=dict(mean=mae_c2_mean, std=mae_c2_std),
            R2=dict(mean=mae_r2_mean, std=mae_r2_std),
        ),
        last5_worstq_rmse_summary=dict(
            n=min(nwr1,nwr2,nwr3),
            C1=dict(mean=wrmse_c1_mean, std=wrmse_c1_std),
            C2=dict(mean=wrmse_c2_mean, std=wrmse_c2_std),
            R2=dict(mean=wrmse_r2_mean, std=wrmse_r2_std),
        ),
        reg_weights=dict(
            w_a=w_a_t.detach().cpu().tolist(),
            w_b=w_b_t.detach().cpu().tolist()
        ),
        args=vars(args), cfg=vars(cfg)
    )
    with open("synth_reg_summary.json","w") as f:
        json.dump(out,f,indent=2)
        print("Saved synth_reg_summary.json")


if __name__=="__main__":
    torch.set_float32_matmul_precision('high')
    main()
