import torch


@torch.no_grad()
def feature_norm_stats(encoder, loader, device: str) -> float:
    total_norm = 0.0
    total_count = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        features = encoder(images)
        total_norm += features.norm(dim=1).sum().item()
        total_count += features.size(0)
    return total_norm / max(1, total_count)


@torch.no_grad()
def add_gaussian_noise(features: torch.Tensor, eps_rel: float, mean_l2: float) -> torch.Tensor:
    noise = torch.randn_like(features)
    noise = noise / noise.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return features + eps_rel * mean_l2 * noise


@torch.no_grad()
def add_laplace_noise(features: torch.Tensor, eps_rel: float, mean_l2: float) -> torch.Tensor:
    uniform = torch.rand_like(features) - 0.5
    noise = -torch.sign(uniform) * torch.log1p(-2.0 * torch.abs(uniform) + 1e-12)
    noise = noise / noise.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return features + eps_rel * mean_l2 * noise

