# PG-DRO

This is the Official Code for "Robust Generalization with Adaptive Optimal Transport Priors for Decision-Focused Learning" (AISTATS 2026)

For numerical experiments, please run the files in the `numerical` folder.

This directory keeps only the PG-DRO image experiment code path.

Included datasets:
- `cifar10`
- `cifar100`
- `tinyimagenet`
- `miniimagenet`
- `tieredimagenet`

Main entry:

```bash
python train.py \
  --base-dataset tinyimagenet \
  --target-dataset cifar100 \
  --shots-per-class 5 \
  --epochs 50 \
  --lr 1e-3 \
  --rho 0.5 \
  --eval-noise gauss \
  --noise-eps 2.0
```

Other paper-style runs:

```bash
python train.py --base-dataset miniimagenet --target-dataset cifar100 --shots-per-class 5 --epochs 50 --lr 1e-4 --rho 0.5 --eval-noise laplace --noise-eps 2.0
python train.py --base-dataset tieredimagenet --target-dataset cifar100 --shots-per-class 5 --epochs 50 --lr 1e-4 --rho 0.5 --eval-noise gauss --noise-eps 2.0
python train.py --base-dataset tieredimagenet --target-dataset miniimagenet --shots-per-class 5 --epochs 50 --lr 1e-4 --rho 0.5 --eval-noise laplace --noise-eps 2.0
python train.py --base-dataset cifar100 --target-dataset cifar10 --shots-per-class 5 --epochs 50 --lr 1e-4 --rho 0.5 --eval-noise clean
```

Checkpoints are written to `./checkpoints/<base>_to_<target>_<noise>/best.pth` unless `--save-dir` is provided.
