from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class ResNetEncoder(nn.Module):
    def __init__(self, feature_dim: int, arch: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if arch == "resnet18":
            backbone = tvm.resnet18(
                weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None
            )
        elif arch == "resnet50":
            backbone = tvm.resnet50(
                weights=tvm.ResNet50_Weights.DEFAULT if pretrained else None
            )
        else:
            raise ValueError(f"Unsupported arch: {arch}")

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Linear(in_features, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


@dataclass
class BaseStats:
    mu_b: torch.Tensor
    var_b: torch.Tensor
    count_b: torch.Tensor


@dataclass
class PriorParams:
    mu_b: torch.Tensor
    var_b: torch.Tensor
    w_bc: torch.Tensor


@dataclass
class PosteriorParams:
    mu_bx: torch.Tensor
    var_bx: torch.Tensor
    log_alpha: torch.Tensor


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def compute_base_stats(features_by_class: List[torch.Tensor], ridge: float = 1e-4) -> BaseStats:
    if not features_by_class:
        raise ValueError("features_by_class cannot be empty")

    mus, variances, counts = [], [], []
    for class_index, features in enumerate(features_by_class):
        if features.numel() == 0:
            raise ValueError(f"base class {class_index} has no encoded features")
        mu = features.mean(dim=0)
        centered = features - mu
        variances.append(centered.pow(2).mean(dim=0) + ridge)
        mus.append(mu)
        counts.append(torch.tensor([features.shape[0]], device=features.device))

    return BaseStats(
        mu_b=torch.stack(mus, dim=0),
        var_b=torch.stack(variances, dim=0),
        count_b=torch.cat(counts, dim=0),
    )


def low_level_softmin_cost(
    z_query: torch.Tensor,
    base_samples_by_class: List[torch.Tensor],
    eps_sample: float,
) -> torch.Tensor:
    costs = []
    for features in base_samples_by_class:
        z2 = z_query.pow(2).sum(dim=1, keepdim=True)
        b2 = features.pow(2).sum(dim=1, keepdim=True).transpose(0, 1)
        d2 = z2 + b2 - 2.0 * (z_query @ features.t())
        softmin = -eps_sample * torch.logsumexp(-0.5 * d2 / eps_sample, dim=1)
        costs.append(softmin)
    return torch.stack(costs, dim=0)


def sinkhorn_log(
    cost: torch.Tensor,
    eps_class: float,
    a: torch.Tensor,
    b: torch.Tensor,
    iters: int = 100,
) -> torch.Tensor:
    kernel_log = -cost / eps_class
    log_a = a.log().unsqueeze(1)
    log_b = b.log().unsqueeze(0)
    u_log = torch.zeros_like(log_a)
    v_log = torch.zeros_like(log_b)

    for _ in range(iters):
        u_log = log_a - torch.logsumexp(kernel_log + v_log, dim=1, keepdim=True)
        v_log = log_b - torch.logsumexp(
            (kernel_log + u_log).transpose(0, 1), dim=1, keepdim=True
        ).transpose(0, 1)

    return torch.exp(u_log + kernel_log + v_log)


def class_weights_from_transport(
    transport: torch.Tensor,
    support_class_indices: List[List[int]],
) -> torch.Tensor:
    weights = []
    for indices in support_class_indices:
        if not indices:
            raise ValueError("support_class_indices contains an empty class")
        weights.append(transport[:, indices].sum(dim=1))

    w_bc = torch.stack(weights, dim=1)
    return w_bc / w_bc.sum(dim=0, keepdim=True).clamp_min(1e-12)


def build_priors_from_hot(base_stats: BaseStats, w_bc: torch.Tensor, cov_inflation: float) -> PriorParams:
    return PriorParams(
        mu_b=base_stats.mu_b,
        var_b=base_stats.var_b * cov_inflation,
        w_bc=w_bc,
    )


def kernelize_to_posterior_mixture(
    encoded_inputs: torch.Tensor,
    priors: PriorParams,
    eps: float,
) -> PosteriorParams:
    num_base, feature_dim = priors.mu_b.shape
    num_target = priors.w_bc.shape[1]
    batch_size = encoded_inputs.shape[0]

    inv_post = 1.0 / priors.var_b + (1.0 / eps)
    var_bx = 1.0 / inv_post
    mu_bx = var_bx.unsqueeze(0) * (
        (priors.mu_b / priors.var_b).unsqueeze(0) + encoded_inputs.unsqueeze(1) / eps
    )

    mu_bx = mu_bx.transpose(0, 1).unsqueeze(1).expand(num_base, num_target, batch_size, feature_dim)
    var_bx = var_bx.unsqueeze(1).unsqueeze(2).expand(num_base, num_target, batch_size, feature_dim)

    log_norm = -0.5 * torch.log1p(priors.var_b / eps).sum(dim=1)
    coef = 1.0 / (eps + priors.var_b)
    delta = encoded_inputs.unsqueeze(1) - priors.mu_b.unsqueeze(0)
    quad = -0.5 * (coef.unsqueeze(0) * delta.pow(2)).sum(dim=2)
    log_kernel = log_norm.unsqueeze(0) + quad

    log_w = priors.w_bc.clamp_min(1e-12).log().unsqueeze(2).expand(num_base, num_target, batch_size)
    log_k = log_kernel.transpose(0, 1).unsqueeze(1).expand(num_base, num_target, batch_size)
    log_alpha_unnorm = log_w + log_k
    log_alpha = log_alpha_unnorm - torch.logsumexp(log_alpha_unnorm, dim=0, keepdim=False)

    return PosteriorParams(mu_bx=mu_bx.contiguous(), var_bx=var_bx.contiguous(), log_alpha=log_alpha)


def robust_logit_newton(
    w_c: torch.Tensor,
    b_c: torch.Tensor,
    mu_bx_c: torch.Tensor,
    var_bx_c: torch.Tensor,
    log_alpha_c: torch.Tensor,
    rho: float,
    eps: float,
    newton_steps: int = 3,
    lam_floor: float = 1e-3,
    lam_cap: float = 100.0,
    step_clip: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    beta_b = mu_bx_c @ w_c
    q_b = var_bx_c @ (w_c ** 2)

    alpha = log_alpha_c.exp()
    beta_bar = (alpha * beta_b).sum(dim=0)
    q_bar = (alpha * (q_b + (beta_b - beta_bar.unsqueeze(0)) ** 2)).sum(dim=0)
    lam = torch.sqrt(q_bar.clamp_min(1e-12) / (2.0 * eps * rho)).clamp(lam_floor, lam_cap)

    for _ in range(newton_steps):
        inv_lam = 1.0 / lam
        inv_le = inv_lam / eps
        inv_le2 = inv_le ** 2
        u = (b_c + beta_b) * inv_le + 0.5 * q_b * inv_le2

        shift = u.max(dim=0, keepdim=True).values
        log_s = torch.logsumexp(log_alpha_c + (u - shift), dim=0) + shift.squeeze(0)
        alpha_tilt = torch.exp(log_alpha_c + u - log_s.unsqueeze(0))

        u_prime = -(b_c + beta_b) / (lam ** 2 * eps) - q_b / (lam ** 3 * eps ** 2)
        f_prime = rho + eps * log_s + lam * eps * (alpha_tilt * u_prime).sum(dim=0)

        u_second = 2.0 * (b_c + beta_b) / (lam ** 3 * eps) + 3.0 * q_b / (lam ** 4 * eps ** 2)
        mean_u_prime = (alpha_tilt * u_prime).sum(dim=0)
        var_u_prime = (alpha_tilt * (u_prime - mean_u_prime.unsqueeze(0)) ** 2).sum(dim=0)
        f_second = 2.0 * eps * mean_u_prime + lam * eps * ((alpha_tilt * u_second).sum(dim=0) + var_u_prime)

        step = (f_prime / (f_second + 1e-12)).clamp(min=-step_clip * lam, max=step_clip * lam)
        lam = (lam - step).clamp(lam_floor, lam_cap)

    inv_lam = 1.0 / lam
    inv_le = inv_lam / eps
    inv_le2 = inv_le ** 2
    u = (b_c + beta_b) * inv_le + 0.5 * q_b * inv_le2
    log_s = torch.logsumexp(log_alpha_c + u, dim=0)
    ell = lam * rho + lam * eps * log_s
    return ell, lam.detach()


class PGDROModel(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module, feature_dim: int, eps: float, rho: float):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.feature_dim = feature_dim
        self.eps = eps
        self.rho = rho
        self.base_stats: Optional[BaseStats] = None

    @torch.no_grad()
    def update_base_stats(self, base_features_by_class: List[torch.Tensor]) -> BaseStats:
        self.base_stats = compute_base_stats(base_features_by_class)
        return self.base_stats

    def forward_batch_loss(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        priors: PriorParams,
        newton_steps: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        encoded = self.encoder(inputs)
        posterior = kernelize_to_posterior_mixture(encoded, priors, eps=self.eps)

        logits = []
        lambdas = []
        for class_index in range(priors.w_bc.shape[1]):
            weight = self.head.weight[class_index]
            bias = self.head.bias[class_index]
            ell_c, lam_c = robust_logit_newton(
                weight,
                bias,
                posterior.mu_bx[:, class_index],
                posterior.var_bx[:, class_index],
                posterior.log_alpha[:, class_index],
                rho=self.rho,
                eps=self.eps,
                newton_steps=newton_steps,
            )
            logits.append(ell_c)
            lambdas.append(lam_c)

        robust_logits = torch.stack(logits, dim=1)
        loss = F.cross_entropy(robust_logits, labels)
        stats = {
            "loss_ce": loss.detach(),
            "ell_mean": robust_logits.detach().mean(),
            "lambda_mean": torch.stack(lambdas, dim=1).mean(),
        }
        return loss, stats


@torch.no_grad()
def robust_logits_from_priors(
    model: PGDROModel,
    inputs_or_features: torch.Tensor,
    priors: PriorParams,
    newton_steps: int,
) -> torch.Tensor:
    if inputs_or_features.dim() >= 4:
        encoded = model.encoder(inputs_or_features)
    else:
        encoded = inputs_or_features

    posterior = kernelize_to_posterior_mixture(encoded, priors, eps=model.eps)
    logits = []
    for class_index in range(priors.w_bc.shape[1]):
        weight = model.head.weight[class_index]
        bias = model.head.bias[class_index]
        ell_c, _ = robust_logit_newton(
            weight,
            bias,
            posterior.mu_bx[:, class_index],
            posterior.var_bx[:, class_index],
            posterior.log_alpha[:, class_index],
            rho=model.rho,
            eps=model.eps,
            newton_steps=newton_steps,
        )
        logits.append(ell_c)
    return torch.stack(logits, dim=1)
