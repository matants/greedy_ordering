import argparse
import os

import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms


def load_embedder(device="cuda", model_name="cifar100_resnet20"):
    # Hub entry: 'cifar100_resnet20' (CIFAR-100, 32x32)
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        model_name,
        pretrained=True,
        trust_repo=True,
    ).to(device).eval()
    # remove classifier robustly: name can be 'linear' or 'fc'
    if hasattr(model, "linear"):
        model.linear = nn.Identity()
    elif hasattr(model, "fc"):
        model.fc = nn.Identity()
    else:
        raise AttributeError("Can't find final linear layer on model.")
    return model


def make_loader(split, batch_size=512, workers=4):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean, std)])
    ds = datasets.CIFAR100(root="./cifar", train=(split == "train"),
                           download=True, transform=tfm)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True
    )
    return dl


@torch.no_grad()
def dump(split, out_dir="../features", device="cuda", model_name="cifar100_resnet20"):
    os.makedirs(out_dir, exist_ok=True)
    model = load_embedder(device, model_name)
    dl = make_loader(split)
    feats, labels = [], []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        z = model(x)  # shape: [B, 64] for CIFAR ResNet20
        feats.append(z.cpu().numpy())
        labels.append(y.numpy())
    feats = np.concatenate(feats, 0)
    labels = np.concatenate(labels, 0)
    np.savez_compressed(os.path.join(out_dir, f"{split}.npz"),
                        feats=feats, labels=labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--out_dir", default="../features_cifar100")
    parser.add_argument("--model", default="cifar100_resnet20")
    args = parser.parse_args()
    dump(args.split, args.out_dir, args.model)
