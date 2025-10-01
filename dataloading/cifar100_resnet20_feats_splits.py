import argparse
import copy
import os
import shutil
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# ------------------------------
# Core: model loading (trainable ResNet20)
# ------------------------------

def load_resnet20_for_training(
        device="cuda",
        pretrained: bool = False,
        num_classes: int = 100,
):
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar100_resnet20",
        pretrained=pretrained,
        trust_repo=True,
    ).to(device)

    head_name = "linear" if hasattr(model, "linear") else ("fc" if hasattr(model, "fc") else None)
    if head_name is None:
        raise AttributeError("Can't find final linear layer on model.")
    head = getattr(model, head_name)
    if head.out_features != num_classes:
        setattr(model, head_name, nn.Linear(head.in_features, num_classes).to(device))
    return model


# ------------------------------
# Transforms / helpers
# ------------------------------

# CIFAR-100 stats per the linked codebase
CIFAR100_MEAN = (0.5070, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2761)


def make_transform(augment: bool = True):
    # Training: crop+flip; Eval: normalize only
    if augment:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])


def _normalize_class_spec(
        cls_spec: Optional[Union[Sequence[int], Sequence[str]]],
        class_to_idx: Dict[str, int]
) -> Optional[List[int]]:
    if cls_spec is None:
        return None
    if len(cls_spec) == 0:
        return []
    if isinstance(cls_spec[0], str):
        return [class_to_idx[name] for name in cls_spec]
    return list(map(int, cls_spec))


class RemappedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int], target_map: Optional[Dict[int, int]] = None):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.target_map = target_map

    def __getitem__(self, idx: int):
        x, y = self.dataset[int(self.indices[idx])]
        if self.target_map is not None:
            y = self.target_map[int(y)]
        return x, y

    def __len__(self):
        return self.indices.shape[0]

    @property
    def classes(self):
        if hasattr(self.dataset, "classes"):
            return self.dataset.classes
        return None


# ------------------------------
# Splitting logic
# ------------------------------

def _targets_from_ds(ds: Dataset) -> np.ndarray:
    if hasattr(ds, "targets"):
        return np.asarray(ds.targets, dtype=np.int64)
    ys = []
    for _, y in ds:
        ys.append(int(y))
    return np.asarray(ys, dtype=np.int64)


def stratified_fraction_split(
        ds: Dataset,
        frac_per_class: float,
        seed: int = 0,
        only_classes: Optional[Iterable[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    assert 0.0 < frac_per_class < 1.0, "Use a fraction in (0,1)."
    rng = np.random.RandomState(seed)
    targets = _targets_from_ds(ds)

    if only_classes is None:
        selected = sorted(set(int(t) for t in targets))
    else:
        selected = sorted(set(int(c) for c in only_classes))

    keep, left = [], []
    for c in selected:
        idx = np.flatnonzero(targets == c)
        rng.shuffle(idx)
        k = int(np.floor(len(idx) * frac_per_class))
        keep.extend(idx[:k].tolist())
        left.extend(idx[k:].tolist())

    if only_classes is not None:
        others = [i for i, y in enumerate(targets) if y not in selected]
        left.extend(others)

    return np.array(keep, dtype=np.int64), np.array(left, dtype=np.int64)


def split_by_class_membership(
        ds: Dataset,
        include_classes: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray]:
    targets = _targets_from_ds(ds)
    include = set(int(c) for c in include_classes)
    keep = np.array([i for i, y in enumerate(targets) if y in include], dtype=np.int64)
    left = np.array([i for i, y in enumerate(targets) if y not in include], dtype=np.int64)
    return keep, left


def make_cifar100_split_datasets(
        root: str = "./cifar",
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        frac_per_class: Optional[float] = None,
        selected_classes: Optional[Union[Sequence[int], Sequence[str]]] = None,
        relabel: bool = False,
        seed: int = 0,
        existing_target_map: Optional[Dict[int, int]] = None,
) -> Tuple[RemappedSubset, RemappedSubset, Optional[Dict[int, int]]]:
    """
    Build primary and complement datasets for CIFAR-100 according to either:
      A) frac_per_class in (0,1): per-class fraction,
      B) selected_classes: keep exactly these classes (all their samples).
    If both are provided, apply the fraction within the selected classes.
    If relabel=True and existing_target_map is given, use it (e.g., test uses train's map).

    Returns: (primary_ds, complement_ds, target_map_used_for_primary_or_None)
    """
    is_train = (split == "train")
    ds_full = datasets.CIFAR100(root=root, train=is_train, download=True, transform=transform)
    cls_map = _normalize_class_spec(selected_classes, ds_full.class_to_idx) if selected_classes is not None else None

    if frac_per_class is not None and (frac_per_class <= 0.0 or frac_per_class >= 1.0):
        raise ValueError("--frac_per_class must be in (0,1) when provided.")

    if cls_map is not None:
        keep_idx, left_idx = split_by_class_membership(ds_full, cls_map)
        if frac_per_class is not None:
            keep_frac_idx, left_frac_idx = stratified_fraction_split(ds_full, frac_per_class, seed,
                                                                     only_classes=cls_map)
            keep_idx = np.array(sorted(set(keep_idx.tolist()) & set(keep_frac_idx.tolist())), dtype=np.int64)
            left_idx = np.array(sorted(set(left_idx.tolist()) | set(left_frac_idx.tolist())), dtype=np.int64)
    elif frac_per_class is not None:
        keep_idx, left_idx = stratified_fraction_split(ds_full, frac_per_class, seed, only_classes=None)
    else:
        raise ValueError("Must provide either frac_per_class or selected_classes.")

    target_map = None
    if relabel:
        if existing_target_map is not None:
            target_map = existing_target_map
        else:
            targets = _targets_from_ds(ds_full)
            labels_present = sorted({int(targets[i]) for i in keep_idx.tolist()})
            target_map = {orig: new for new, orig in enumerate(labels_present)}

    primary = RemappedSubset(ds_full, keep_idx, target_map=target_map)
    complement = RemappedSubset(ds_full, left_idx, target_map=None)
    return primary, complement, target_map


def make_loaders_from_datasets(
        ds_primary: Dataset,
        ds_complement: Optional[Dataset],
        batch_size: int = 256,
        workers: int = 4,
        shuffle_primary: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    dl_primary = DataLoader(
        ds_primary, batch_size=batch_size, shuffle=shuffle_primary,
        num_workers=workers, pin_memory=True, drop_last=False
    )
    dl_complement = None
    if ds_complement is not None:
        dl_complement = DataLoader(
            ds_complement, batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, drop_last=False
        )
    return dl_primary, dl_complement


# ------------------------------
# Training / eval
# ------------------------------

def train(
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
        epochs: int = 200,
        base_lr: float = 0.1,
        weight_decay: float = 5e-4,
        momentum: float = 0.9,
        num_classes: int = 100,
        pretrained_backbone: bool = False,
        grad_clip: Optional[float] = None,
        use_amp: bool = False,
        basic_bs: int = 256,
) -> nn.Module:
    torch.backends.cudnn.benchmark = True

    model = load_resnet20_for_training(
        device=device, pretrained=pretrained_backbone, num_classes=num_classes
    )

    # LR scaling to emulate CustomSGD.basic_bs=256
    scaled_lr = base_lr * (train_loader.batch_size / float(basic_bs))

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=scaled_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0.0
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    criterion = nn.CrossEntropyLoss()

    best = {"epoch": -1, "acc": -1.0}
    for ep in range(epochs):
        model.train()
        running_loss, seen = 0.0, 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * x.size(0)
            seen += x.size(0)

        lr_sched.step()
        tr_loss = running_loss / max(1, seen)

        if val_loader is not None:
            acc = evaluate(model, val_loader, device=device)
            if acc > best["acc"]:
                best = {"epoch": ep, "acc": acc}
            print(
                f"[{ep + 1:03d}/{epochs}] train_loss={tr_loss:.4f}  val_acc={acc:.2f}%  best={best['acc']:.2f}%@{best['epoch'] + 1}")
        else:
            print(f"[{ep + 1:03d}/{epochs}] train_loss={tr_loss:.4f}")

    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str = "cuda") -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += y.numel()
    return 100.0 * correct / max(1, total)


def make_feature_model_from_trained(trained: nn.Module) -> nn.Module:
    m = copy.deepcopy(trained).eval()
    if hasattr(m, "linear"):
        m.linear = nn.Identity()
    elif hasattr(m, "fc"):
        m.fc = nn.Identity()
    else:
        raise AttributeError("Can't find final linear layer on model.")
    return m


@torch.no_grad()
def dump_with_trained_model(
        trained_model: nn.Module,
        loader: torch.utils.data.DataLoader,
        out_path: str,
        device: str = "cuda",
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    f = make_feature_model_from_trained(trained_model).to(device).eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = f(x)  # penultimate features from the TRAINED net
        feats.append(z.detach().cpu().numpy())
        labels.append(y.numpy())
    np.savez_compressed(out_path,
                        feats=np.concatenate(feats, 0),
                        labels=np.concatenate(labels, 0))


# ------------------------------
# Full test loader (unsplit) helper
# ------------------------------

def make_full_test_loader(
        root: str,
        batch_size: int,
        workers: int,
        transform,
        classes_list: Optional[Sequence[int]],
        relabel: bool,
        target_map: Optional[Dict[int, int]],
):
    """
    Build an UNSPLIT test loader.
    - If classes_list is None and not relabeling to fewer classes: use the full 10k test set.
    - If classes_list is provided and relabel=True: filter to those classes and apply target_map (K-way eval).
    - If classes_list is provided and relabel=False: filter to those classes but keep original labels (100-way labels).
    Returns (loader, indices_array)
    """
    base = datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
    if classes_list is None:
        indices = np.arange(len(base), dtype=np.int64)
        ds = RemappedSubset(base, indices, target_map=None)
    else:
        keep_idx, _ = split_by_class_membership(base, classes_list)
        ds = RemappedSubset(base, keep_idx, target_map=target_map if relabel else None)
        indices = keep_idx

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False)
    return dl, indices


# ------------------------------
# Argparse
# ------------------------------

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="../features_cifar100_split")
    parser.add_argument("--model", default="cifar100_resnet20")

    parser.add_argument("--root", default="./cifar_split")
    parser.add_argument("--batch_size", type=int, default=256)  # match repo
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    # Split controls
    parser.add_argument("--frac_per_class", type=float, default=None,
                        help="e.g., 0.5 to keep half per class (PRIMARY). Complement gets the rest.")
    parser.add_argument("--classes", type=str, default=None,
                        help="Comma-separated class names or indices, e.g. 'apple,bear' or '1,5,42'.")
    parser.add_argument("--relabel", action="store_true",
                        help="Relabel PRIMARY labels to [0..K-1]; used consistently for train+test if set.")

    # Augmentation / training
    parser.add_argument("--augment", action="store_true",  # default True (repo aug on)
                        help="Use CIFAR crop+flip for training.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)  # base LR before scaling
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--pretrained_backbone", action="store_true")
    parser.add_argument("--amp", action="store_true",  # repo CIFAR configs typically off
                        help="Enable mixed precision (disabled by default to mirror recipe).")

    # Validation choice
    parser.add_argument("--val_full_test", action="store_true",
                        help="Validate on the UNSPLIT test set instead of the split test_primary.")

    # Save indices
    parser.add_argument("--save_split_indices", type=str, default=None,
                        help="Saves npz with train/test primary & complement indices (+ full test indices).")
    return parser


# ------------------------------
# CLI run
# ------------------------------

def run(args):
    # Parse class list (names or indices)
    classes_list = None
    if args.classes is not None:
        raw = [s.strip() for s in args.classes.split(",") if s.strip()]
        tmp = datasets.CIFAR100(root=args.root, train=True, download=True)
        classes_list = _normalize_class_spec(raw, tmp.class_to_idx)

    # ---- Build TRAIN splits
    tfm_train = make_transform(augment=args.augment)  # default True
    ds_tr_primary, ds_tr_complement, target_map = make_cifar100_split_datasets(
        root=args.root,
        split="train",
        transform=tfm_train,
        frac_per_class=args.frac_per_class,
        selected_classes=classes_list,
        relabel=args.relabel,
        seed=args.seed,
        existing_target_map=None,
    )
    dl_tr_primary, dl_tr_complement = make_loaders_from_datasets(
        ds_tr_primary, ds_tr_complement,
        batch_size=args.batch_size, workers=args.workers,
        shuffle_primary=True
    )

    # ---- Build TEST splits with the SAME rules (and SAME label map if relabel)
    tfm_test = make_transform(augment=False)
    ds_te_primary, ds_te_complement, _ = make_cifar100_split_datasets(
        root=args.root,
        split="test",
        transform=tfm_test,
        frac_per_class=args.frac_per_class,
        selected_classes=classes_list,
        relabel=args.relabel,
        seed=args.seed,
        existing_target_map=target_map,
    )
    dl_te_primary, dl_te_complement = make_loaders_from_datasets(
        ds_te_primary, ds_te_complement,
        batch_size=args.batch_size, workers=args.workers,
        shuffle_primary=False
    )

    # ---- Build UNSPLIT TEST loader (for optional validation + dumping)
    dl_te_full, te_full_idx = make_full_test_loader(
        root=args.root,
        batch_size=args.batch_size,
        workers=args.workers,
        transform=tfm_test,
        classes_list=classes_list,
        relabel=args.relabel,
        target_map=target_map,
    )

    # ---- Train; validation chooser
    if args.relabel and classes_list is not None:
        num_classes = len(classes_list)
    elif args.frac_per_class is None and classes_list is None:
        num_classes = 100
    else:
        num_classes = 100 if not args.relabel else (len(classes_list) if classes_list is not None else 100)

    val_loader = dl_te_full if args.val_full_test else dl_te_primary
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train(
        dl_tr_primary,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        base_lr=args.lr,
        weight_decay=args.wd,
        momentum=args.momentum,
        num_classes=num_classes,
        pretrained_backbone=args.pretrained_backbone,
        use_amp=args.amp,
        basic_bs=256,
    )

    # ---- Dump TRAINED features for all four split parts + FULL test
    base = os.path.join(args.out_dir, "trained_features")
    os.makedirs(base, exist_ok=True)
    dump_with_trained_model(model, dl_tr_primary, os.path.join(base, "train_primary.npz"), device=device)
    dump_with_trained_model(model, dl_tr_complement, os.path.join(base, "train_complement.npz"), device=device)
    dump_with_trained_model(model, dl_te_primary, os.path.join(base, "test_primary.npz"), device=device)
    dump_with_trained_model(model, dl_te_complement, os.path.join(base, "test_complement.npz"), device=device)
    dump_with_trained_model(model, dl_te_full, os.path.join(base, "test_full.npz"), device=device)

    print(f"Trained-model features saved under: {base}")

    # Save checkpoint
    os.makedirs("./checkpoints", exist_ok=True)
    ckpt_path = os.path.join("./checkpoints", "resnet20_cifar100_customsplit.pt")
    torch.save({"model": model.state_dict()}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # Save indices if requested (train+test)
    if args.save_split_indices is not None:
        np.savez_compressed(
            args.save_split_indices,
            train_primary_idx=ds_tr_primary.indices,
            train_complement_idx=ds_tr_complement.indices,
            test_primary_idx=ds_te_primary.indices,
            test_complement_idx=ds_te_complement.indices,
            test_full_idx=te_full_idx,
        )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)  # if argv=None, uses sys.argv[1:]
    run(args)


if __name__ == "__main__":
    out_dir = "../features_cifar100_split"
    main([
        "--frac_per_class", "0.5",
        "--epochs", "200",
        "--seed", "0",
        "--val_full_test",  # comment this out to validate on split test_primary
        # "--amp",                # enable if you want AMP (off by default to mirror recipe)
        "--augment",  # augment is True in the original training setup;
        "--out_dir", out_dir
    ])
    shutil.copy2(os.path.join(out_dir, "trained_features", "train_complement.npz"), os.path.join(out_dir, "train.npz"))
    shutil.copy2(os.path.join(out_dir, "trained_features", "test_full.npz"), os.path.join(out_dir, "test.npz"))
