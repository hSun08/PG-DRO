import os
from typing import Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


DATASET_CHOICES = (
    "cifar10",
    "cifar100",
    "tinyimagenet",
    "miniimagenet",
    "tieredimagenet",
)


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_transform, eval_transform


class TinyImageNet(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None):
        super().__init__()
        self.root = os.path.join(root, "tiny-imagenet-200")
        self.transform = transform
        self.samples = []

        wnids_path = os.path.join(self.root, "wnids.txt")
        with open(wnids_path, "r", encoding="utf-8") as handle:
            wnids = [line.strip() for line in handle if line.strip()]
        self.class_to_idx = {wnid: index for index, wnid in enumerate(wnids)}

        if split == "train":
            for wnid in wnids:
                image_dir = os.path.join(self.root, "train", wnid, "images")
                for file_name in sorted(os.listdir(image_dir)):
                    if file_name.endswith(".JPEG"):
                        self.samples.append((os.path.join(image_dir, file_name), self.class_to_idx[wnid]))
        elif split == "val":
            annotation_path = os.path.join(self.root, "val", "val_annotations.txt")
            with open(annotation_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    file_name, wnid = parts[0], parts[1]
                    self.samples.append(
                        (os.path.join(self.root, "val", "images", file_name), self.class_to_idx[wnid])
                    )
        else:
            raise ValueError(f"Unsupported TinyImageNet split: {split}")

        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class MiniImageNet(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None):
        super().__init__()
        self.root = os.path.join(root, split)
        self.transform = transform
        self.samples = []
        self.targets = []

        classes = [
            class_name
            for class_name in sorted(os.listdir(self.root))
            if os.path.isdir(os.path.join(self.root, class_name))
        ]
        self.class_to_idx = {class_name: index for index, class_name in enumerate(classes)}

        for class_name in classes:
            class_dir = os.path.join(self.root, class_name)
            label = self.class_to_idx[class_name]
            for file_name in sorted(os.listdir(class_dir)):
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, file_name), label))
                    self.targets.append(label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class TieredImageNet(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None):
        super().__init__()
        self.root = os.path.join(root, split)
        self.transform = transform
        self.samples = []
        self.targets = []

        classes = [
            class_name
            for class_name in sorted(os.listdir(self.root))
            if os.path.isdir(os.path.join(self.root, class_name))
        ]
        self.class_to_idx = {class_name: index for index, class_name in enumerate(classes)}

        for class_name in classes:
            class_dir = os.path.join(self.root, class_name)
            label = self.class_to_idx[class_name]
            for file_name in sorted(os.listdir(class_dir)):
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, file_name), label))
                    self.targets.append(label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def dataset_num_classes(dataset_name: str, dataset: Dataset) -> int:
    if dataset_name == "cifar10":
        return 10
    if dataset_name == "cifar100":
        return 100
    return len(set(dataset.targets))


def target_test_split(dataset_name: str) -> str:
    if dataset_name in {"cifar10", "cifar100"}:
        return "test"
    if dataset_name == "tinyimagenet":
        return "val"
    if dataset_name in {"miniimagenet", "tieredimagenet"}:
        return "test"
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_dataset(
    dataset_name: str,
    split: str,
    transform,
    cifar_root: str,
    tiny_root: str,
    mini_root: str,
    tiered_root: str,
):
    if dataset_name == "cifar10":
        return datasets.CIFAR10(
            root=cifar_root,
            train=(split == "train"),
            download=True,
            transform=transform,
        )
    if dataset_name == "cifar100":
        return datasets.CIFAR100(
            root=cifar_root,
            train=(split == "train"),
            download=True,
            transform=transform,
        )
    if dataset_name == "tinyimagenet":
        mapped_split = "train" if split == "train" else "val"
        return TinyImageNet(root=tiny_root, split=mapped_split, transform=transform)
    if dataset_name == "miniimagenet":
        return MiniImageNet(root=mini_root, split=split, transform=transform)
    if dataset_name == "tieredimagenet":
        return TieredImageNet(root=tiered_root, split=split, transform=transform)
    raise ValueError(f"Unsupported dataset: {dataset_name}")

