import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from datasets import (
    DATASET_CHOICES,
    build_transforms,
    dataset_num_classes,
    load_dataset,
    target_test_split,
)
from noise import add_gaussian_noise, add_laplace_noise, feature_norm_stats
from pg_dro import (
    PGDROModel,
    ResNetEncoder,
    build_priors_from_hot,
    class_weights_from_transport,
    l2_normalize,
    low_level_softmin_cost,
    robust_logits_from_priors,
    sinkhorn_log,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_support_indices(dataset, shots_per_class: int, num_classes: int, seed: int) -> Tuple[List[int], List[List[int]]]:
    rng = random.Random(seed)
    targets = dataset.targets if hasattr(dataset, "targets") else [label for _, label in dataset]

    class_to_indices = {class_index: [] for class_index in range(num_classes)}
    for index, label in enumerate(targets):
        label = int(label)
        if 0 <= label < num_classes:
            class_to_indices[label].append(index)

    support_indices = []
    support_class_indices = []
    cursor = 0
    for class_index in range(num_classes):
        candidates = class_to_indices[class_index]
        chosen = rng.sample(candidates, min(shots_per_class, len(candidates)))
        support_indices.extend(chosen)
        support_class_indices.append(list(range(cursor, cursor + len(chosen))))
        cursor += len(chosen)

    return support_indices, support_class_indices


@torch.no_grad()
def encode_subset(
    encoder: nn.Module,
    dataset,
    indices: List[int],
    device: str,
    batch_size: int,
    num_workers: int,
) -> torch.Tensor:
    if not indices:
        raise ValueError("support set is empty")

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    encoded_batches = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        encoded_batches.append(encoder(images).detach())
    return torch.cat(encoded_batches, dim=0)


@torch.no_grad()
def encode_base_bank(
    encoder: nn.Module,
    loader: DataLoader,
    num_base_classes: int,
    device: str,
    max_per_class: int,
) -> List[torch.Tensor]:
    buckets: List[List[torch.Tensor]] = [[] for _ in range(num_base_classes)]
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        features = encoder(images)
        for row in range(features.size(0)):
            class_index = int(labels[row])
            if len(buckets[class_index]) < max_per_class:
                buckets[class_index].append(features[row].detach().cpu())

    encoded_by_class = []
    for class_index, bucket in enumerate(buckets):
        if not bucket:
            raise ValueError(f"base class {class_index} has no encoded samples")
        encoded_by_class.append(torch.stack(bucket, dim=0).to(device))
    return encoded_by_class


def set_encoder_requires_grad(encoder: nn.Module, requires_grad: bool) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad = requires_grad


def set_encoder_bn_mode(encoder: nn.Module, training: bool, freeze_affine: bool) -> None:
    for module in encoder.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train(training)
            if module.weight is not None:
                module.weight.requires_grad_(not freeze_affine)
            if module.bias is not None:
                module.bias.requires_grad_(not freeze_affine)


def summarize_wbc(w_bc: torch.Tensor) -> Tuple[float, float, float]:
    weights = w_bc.clamp_min(1e-12)
    entropy = (-(weights * weights.log()).sum(dim=0)).mean().item()
    top1 = weights.max(dim=0).values.mean().item()
    perplexity = float(torch.exp(torch.tensor(entropy)))
    return entropy, top1, perplexity


def evaluate(
    model: PGDROModel,
    loader: DataLoader,
    priors,
    newton_steps: int,
    eval_noise: str,
    noise_eps: float,
    device: str,
) -> float:
    model.eval()
    mean_feat_l2 = feature_norm_stats(model.encoder, loader, device) if eval_noise != "clean" else 1.0

    total = 0
    correct = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            features = model.encoder(images)
            if eval_noise == "gauss":
                features = add_gaussian_noise(features, noise_eps, mean_feat_l2)
            elif eval_noise == "laplace":
                features = add_laplace_noise(features, noise_eps, mean_feat_l2)
            elif eval_noise != "clean":
                raise ValueError(f"Unsupported eval_noise: {eval_noise}")

            logits = robust_logits_from_priors(model, features, priors, newton_steps)
            predictions = logits.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    return 100.0 * correct / max(1, total)


def train_pg_dro(args) -> float:
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)

    train_transform, eval_transform = build_transforms(image_size=args.image_size)

    base_bank_dataset = load_dataset(
        args.base_dataset,
        split="train",
        transform=eval_transform,
        cifar_root=args.cifar_root,
        tiny_root=args.tiny_root,
        mini_root=args.mini_root,
        tiered_root=args.tiered_root,
    )
    target_train_dataset = load_dataset(
        args.target_dataset,
        split="train",
        transform=train_transform,
        cifar_root=args.cifar_root,
        tiny_root=args.tiny_root,
        mini_root=args.mini_root,
        tiered_root=args.tiered_root,
    )
    target_support_dataset = load_dataset(
        args.target_dataset,
        split="train",
        transform=eval_transform,
        cifar_root=args.cifar_root,
        tiny_root=args.tiny_root,
        mini_root=args.mini_root,
        tiered_root=args.tiered_root,
    )
    target_test_dataset = load_dataset(
        args.target_dataset,
        split=target_test_split(args.target_dataset),
        transform=eval_transform,
        cifar_root=args.cifar_root,
        tiny_root=args.tiny_root,
        mini_root=args.mini_root,
        tiered_root=args.tiered_root,
    )

    num_base_classes = dataset_num_classes(args.base_dataset, base_bank_dataset)
    num_target_classes = dataset_num_classes(args.target_dataset, target_support_dataset)

    base_loader = DataLoader(
        base_bank_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        target_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    support_indices, support_class_indices = sample_support_indices(
        target_support_dataset,
        shots_per_class=args.shots_per_class,
        num_classes=num_target_classes,
        seed=args.seed,
    )
    support_batch_size = max(1, min(args.support_batch_size, len(support_indices)))
    support_train_loader = DataLoader(
        Subset(target_train_dataset, support_indices),
        batch_size=support_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    encoder = ResNetEncoder(
        feature_dim=args.feature_dim,
        arch=args.arch,
        pretrained=args.pretrained,
    )
    head = nn.Linear(args.feature_dim, num_target_classes)
    model = PGDROModel(
        encoder=encoder,
        head=head,
        feature_dim=args.feature_dim,
        eps=args.eps_sample,
        rho=args.rho,
    ).to(args.device)

    optimizer = optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": args.lr * 0.5},
            {"params": model.head.parameters(), "lr": args.lr},
        ],
        weight_decay=1e-4,
    )

    save_dir = args.save_dir or os.path.join(
        ".",
        "checkpoints",
        f"{args.base_dataset}_to_{args.target_dataset}_{args.eval_noise}",
    )
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"PG-DRO image training ({args.base_dataset} -> {args.target_dataset})")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Backbone: {args.arch}, pretrained={args.pretrained}, feature_dim={args.feature_dim}")
    print(f"Target classes: {num_target_classes} | Shots per class: {args.shots_per_class}")
    print(f"Batch size (test): {args.batch_size} | Support batch size: {support_batch_size}")
    print(f"LR: {args.lr} | Epochs: {args.epochs} | Eval noise: {args.eval_noise} (eps={args.noise_eps})")
    print(f"Support shots per class: {[len(indices) for indices in support_class_indices]}")

    best_acc = 0.0
    history = []

    for epoch in range(args.epochs):
        frozen = epoch < args.freeze_epochs
        set_encoder_requires_grad(model.encoder, not frozen)

        print(f"\n[Epoch {epoch + 1}/{args.epochs}] recomputing priors ...")
        model.eval()
        with torch.no_grad():
            base_features_by_class = encode_base_bank(
                model.encoder,
                base_loader,
                num_base_classes=num_base_classes,
                device=args.device,
                max_per_class=args.base_bank_max,
            )
            support_features = encode_subset(
                model.encoder,
                target_support_dataset,
                support_indices,
                device=args.device,
                batch_size=args.support_batch_size,
                num_workers=args.num_workers,
            )
            base_stats = model.update_base_stats(base_features_by_class)
            cost = low_level_softmin_cost(
                z_query=support_features,
                base_samples_by_class=base_features_by_class,
                eps_sample=args.eps_sample,
            )
            a = torch.full((cost.size(0),), 1.0 / cost.size(0), device=args.device)
            b = torch.full((cost.size(1),), 1.0 / cost.size(1), device=args.device)
            transport = sinkhorn_log(cost, eps_class=args.eps_class, a=a, b=b, iters=args.sinkhorn_iters)
            w_bc = class_weights_from_transport(transport, support_class_indices)
            priors = build_priors_from_hot(base_stats, w_bc, cov_inflation=args.cov_inflation)

            entropy, top1, perplexity = summarize_wbc(w_bc)
            print(f"[w_bc] entropy={entropy:.3f} | top1={top1:.3f} | perp≈{perplexity:.1f}")

            if epoch >= args.ema_warmup and args.ema_proto > 0.0:
                proto_head = l2_normalize(w_bc.t() @ base_stats.mu_b, dim=1)
                old_head = l2_normalize(model.head.weight.data, dim=1)
                model.head.weight.data = l2_normalize(
                    (1.0 - args.ema_proto) * old_head + args.ema_proto * proto_head,
                    dim=1,
                )
                if model.head.bias is not None:
                    model.head.bias.data.zero_()

        model.train()
        if frozen:
            set_encoder_bn_mode(model.encoder, training=False, freeze_affine=True)
        else:
            set_encoder_bn_mode(model.encoder, training=True, freeze_affine=False)

        running_loss = 0.0
        running_correct = 0
        running_total = 0
        total_steps = args.repeat_factor * len(support_train_loader)
        completed_steps = 0

        for repeat_index in range(args.repeat_factor):
            for images, labels in support_train_loader:
                images = images.to(args.device, non_blocking=True)
                labels = labels.to(args.device, non_blocking=True)

                optimizer.zero_grad()
                loss, _stats = model.forward_batch_loss(
                    images,
                    labels,
                    priors=priors,
                    newton_steps=args.newton_steps,
                )
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    logits = robust_logits_from_priors(model, images, priors, args.newton_steps)
                    predictions = logits.argmax(dim=1)
                    running_total += labels.size(0)
                    running_correct += (predictions == labels).sum().item()

                completed_steps += 1
                running_loss += float(loss)
                avg_loss = running_loss / completed_steps
                train_acc = 100.0 * running_correct / max(1, running_total)
                print(
                    f"\r[Epoch {epoch + 1}/{args.epochs}] rep {repeat_index + 1}/{args.repeat_factor} "
                    f"| step {completed_steps}/{total_steps} "
                    f"| train_loss={avg_loss:.4f} | train_acc={train_acc:.2f}%   ",
                    end="",
                    flush=True,
                )
        print()

        test_acc = evaluate(
            model,
            test_loader,
            priors=priors,
            newton_steps=args.newton_steps,
            eval_noise=args.eval_noise,
            noise_eps=args.noise_eps,
            device=args.device,
        )
        history.append(test_acc)
        print(f"[Epoch {epoch + 1}/{args.epochs}] test_acc[{args.eval_noise}] = {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_test_acc": best_acc,
                    "support_indices": support_indices,
                    "base_dataset": args.base_dataset,
                    "target_dataset": args.target_dataset,
                    "eval_noise": args.eval_noise,
                },
                os.path.join(save_dir, "best.pth"),
            )
            print(f"saved best checkpoint: {best_acc:.2f}%")

    last_k = min(5, len(history))
    tail = np.array(history[-last_k:], dtype=np.float32)
    print(f"\nLast {last_k} epochs: mean acc = {float(tail.mean()):.2f}%, std = {float(tail.std(ddof=0)):.2f}%")
    print(f"Best test accuracy: {best_acc:.2f}%")
    return best_acc


def parse_args():
    parser = argparse.ArgumentParser(description="PG-DRO image experiment runner")
    parser.add_argument("--base-dataset", type=str, required=True, choices=DATASET_CHOICES)
    parser.add_argument("--target-dataset", type=str, required=True, choices=DATASET_CHOICES)
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--feature-dim", type=int, default=512)

    parser.set_defaults(pretrained=True)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--freeze-epochs", type=int, default=0)
    parser.add_argument("--repeat-factor", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--support-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--shots-per-class", type=int, default=5)
    parser.add_argument("--eps-sample", type=float, default=0.1)
    parser.add_argument("--eps-class", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--cov-inflation", type=float, default=1.5)
    parser.add_argument("--sinkhorn-iters", type=int, default=100)
    parser.add_argument("--newton-steps", type=int, default=5)
    parser.add_argument("--base-bank-max", type=int, default=256)
    parser.add_argument("--ema-proto", type=float, default=0.1)
    parser.add_argument("--ema-warmup", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)

    parser.add_argument("--eval-noise", type=str, default="clean", choices=["clean", "gauss", "laplace"])
    parser.add_argument("--noise-eps", type=float, default=2.0)

    parser.add_argument("--cifar-root", type=str, default="/home/sun1321/src/DRO/data")
    parser.add_argument("--tiny-root", type=str, default="/home/sun1321/src/DRO/data")
    parser.add_argument(
        "--mini-root",
        type=str,
        default="/scratch/gilbreth/sun1321/kaggle/data/mini-imagenet_raw/mini-imagenet",
    )
    parser.add_argument("--tiered-root", type=str, default="/scratch/gilbreth/sun1321")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    return parser.parse_args()


if __name__ == "__main__":
    train_pg_dro(parse_args())
