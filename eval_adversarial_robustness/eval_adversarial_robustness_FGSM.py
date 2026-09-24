#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_FGSM_fair_v2.py

Update Points:
- Added `replknet31B` support.
- Unified local weight loading: `--local_weight_path` seamlessly handles BOTH 
  official PyTorch base weights and PyTorch Lightning distilled checkpoints.
- Updated Preprocessing: ConvNeXt and RepLKNet use Resize(256) -> CenterCrop(224) with BICUBIC.

Usage Examples:

1. Standard Torchvision Model
   python eval_FGSM_fair_v2.py --model_tag resnet34 --torchvision --stats_type imagenet

2. RepLKNet Official Base Model (Model B stats)
   python eval_FGSM_fair_v2.py --model_tag replknet31B --local_weight_path ./replknet31B_imagenet.pth --stats_type imagenet

3. RepLKNet Distilled Model (Model A stats)
   python eval_FGSM_fair_v2.py --model_tag replknet31B --local_weight_path ./runs/.../32-0.0000.ckpt --stats_type custom
"""

import os
import sys
import argparse
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import v2 as T

# RepLKNetのインポート（環境に合わせてパスを調整してください）
sys.path.append("../../DiST")
try:
    from real_cnn_model.models.replknet import create_RepLKNet31B
except ImportError:
    try:
        from models.replknet import create_RepLKNet31B
    except ImportError:
        from replknet import create_RepLKNet31B

# --------------------------
# Wrapper for On-Model Normalization
# --------------------------
class NormalizedModel(nn.Module):
    """
    入力を受け取り、指定されたmean/stdで正規化してから
    ベースモデルに渡すラッパークラス。
    これにより、攻撃(FGSM)は [0, 1] の画像空間で行われ、
    モデルに入力される直前に正規化が掛かるようになる。
    """
    def __init__(self, model, mean, std):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        
    def forward(self, x):
        normalized_x = (x - self.mean) / self.std
        return self.model(normalized_x)


# --------------------------
# Model Instantiation Helper
# --------------------------
def get_model_info(model_tag):
    """
    Returns (Model Class, Weights Enum) for torchvision models.
    """
    if model_tag == "resnet34":
        return models.resnet34, models.ResNet34_Weights.DEFAULT
    elif model_tag == "resnet50":
        return models.resnet50, models.ResNet50_Weights.DEFAULT
    elif model_tag == "vgg16_bn":
        return models.vgg16_bn, models.VGG16_BN_Weights.DEFAULT
    elif model_tag == "convnext_tiny":
        return models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT
    elif model_tag == "convnext_base":
        return models.convnext_base, models.ConvNeXt_Base_Weights.DEFAULT
    else:
        raise ValueError(f"Unsupported torchvision model tag: {model_tag}")

def instantiate_empty_model(model_tag, num_classes=1000):
    if model_tag == "replknet31B":
        # RepLKNetの初期化 (checkpointは不要なのでFalse)
        return create_RepLKNet31B(num_classes=num_classes, use_checkpoint=False)
    else:
        model_cls, _ = get_model_info(model_tag)
        return model_cls(weights=None, num_classes=num_classes)


# --------------------------
# ckpt utilities
# --------------------------
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "net", "model_state", "state_dict_ema", "ema"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt

def _strip_prefixes_repeat(k: str, prefixes: Tuple[str, ...]) -> str:
    while True:
        changed = False
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                changed = True
        if not changed:
            break
    return k

def load_model_safely_from_ckpt(
    ckpt_path: str,
    device: str,
    model_tag: str,
    num_classes: int = 1000,
    prefer_student: bool = True,
) -> torch.nn.Module:
    """
    Lightning ckpt or Standard .pth loader
    """
    # 1. Initialize empty model structure
    model = instantiate_empty_model(model_tag, num_classes=num_classes)

    # 2. Load Checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd_raw = _extract_state_dict(ckpt)

    student_prefixes = (
        "student.", "model.student.", "net.student.", "student.net.", "module.student.",
    )
    generic_strip = (
        "state_dict.", "module.", "model.", "net.", "backbone.", "resnet.", "vgg.", "convnext.",
    )

    # Lightningの生徒モデルか、公式のプレーンな重みかを自動判定
    has_student = False
    if prefer_student:
        has_student = any(any(k.startswith(p) for p in student_prefixes) for k in sd_raw.keys())

    target = model.state_dict()
    filtered = {}

    used = 0
    drop_teacher = 0

    for k, v in sd_raw.items():
        if k.startswith("teacher.") or ".teacher." in k:
            drop_teacher += 1
            continue

        k2 = k
        if has_student:
            if not any(k2.startswith(p) for p in student_prefixes):
                continue
            k2 = _strip_prefixes_repeat(k2, student_prefixes)

        k2 = _strip_prefixes_repeat(k2, generic_strip)

        if k2 in target and hasattr(v, "shape") and v.shape == target[k2].shape:
            filtered[k2] = v
            used += 1

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    
    if used < 5:
        print(f"[WARN] Loaded keys very low: {used}. Check prefix logic.")
    
    print(f"[LOAD LOCAL] {ckpt_path}")
    print(f"             Tag={model_tag} Used={used} DropTeacher={drop_teacher}")
    print(f"             Missing={len(missing)} Unexpected={len(unexpected)}")
    
    return model


# --------------------------
# FGSM (Fairness Preserved)
# --------------------------
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    # Clip to [0, 1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def evaluate_robustness(model, dataloader, device, epsilons):
    accuracies = []
    
    # 念のためモデル全体をEvaluationモードに
    model.eval()

    for eps in epsilons:
        correct = 0
        total = 0

        loop = tqdm(dataloader, desc=f"Eps {eps:.4f}", leave=False)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            images.requires_grad = True

            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            model.zero_grad(set_to_none=True)
            loss.backward()
            data_grad = images.grad.data

            if eps == 0:
                perturbed_images = images.detach()
            else:
                perturbed_images = fgsm_attack(images, eps, data_grad).detach()

            with torch.no_grad():
                outputs_adv = model(perturbed_images)
                predicted = outputs_adv.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = correct / max(total, 1)
        accuracies.append(acc)
        print(f"  Epsilon: {eps:.4f} -> Accuracy: {acc:.4f}")

    return accuracies


# --------------------------
# Args & Main
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fair FGSM Robustness")
    parser.add_argument("--data_root", type=str, default="/datasets/imagenet/val",
                        help="Path to ImageNet validation dataset root.")
    parser.add_argument("--model_tag", type=str, default="resnet34",
                        choices=["resnet34", "resnet50", "vgg16_bn", "convnext_tiny", "convnext_base", "replknet31B"],
                        help="Model architecture.")
    
    parser.add_argument("--stats_type", type=str, default="imagenet",
                        choices=["imagenet", "custom"],
                        help="'imagenet' for standard stats, 'custom' for Model A type stats.")

    parser.add_argument("--torchvision", action="store_true",
                        help="If True, load standard pretrained weights from torchvision.")
    
    # --distilled_path をより汎用的な --local_weight_path に変更 (互換性のためエイリアスとして残す)
    parser.add_argument("--local_weight_path", "--distilled_path", type=str, default=None, dest="local_weight_path",
                        help="Path to local weights (.pth or .ckpt). Handles both official base weights and distilled models.")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./results")
    return parser.parse_args()


def get_preprocess_without_normalization(model_tag):
    # ConvNeXtとRepLKNetは標準でResize(256) -> CenterCrop(224) (BICUBIC) を使用
    if "convnext" in model_tag or "replknet" in model_tag.lower():
        return T.Compose([
            T.Resize(256, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Define Stats
    if args.stats_type == "custom":
        # Model A Stats
        mean = [0.48235, 0.45882, 0.40784]
        std = [0.00392156862745098, 0.00392156862745098, 0.00392156862745098]
    else:
        # Model B / ImageNet Stats
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    print(f"Stats Type: {args.stats_type}")
    print(f"Mean: {mean}")
    print(f"Std : {std}")

    # 2. Dataset (No Normalization here)
    preprocess = get_preprocess_without_normalization(args.model_tag)
    full_dataset = torchvision.datasets.ImageFolder(root=args.data_root, transform=preprocess)
    
    if args.num_images:
        dataset = Subset(full_dataset, torch.arange(min(args.num_images, len(full_dataset))))
    else:
        dataset = full_dataset
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    epsilons = [0, 0.25/255, 0.5/255, 1/255, 2/255, 3/255, 4/255]
    epsilons = [0, 0.25/255, 0.5/255, 1/255]
    # epsilons = [1/255]
    epsilons = [0]

    # 3. Model Loading Logic
    base_model = None

    if args.torchvision:
        if args.model_tag == "replknet31B":
            print("Error: RepLKNet31B does not have torchvision weights.")
            print("       Please use --local_weight_path to load the official .pth file.")
            sys.exit(1)
            
        print(f"\n=== Loading Standard Torchvision Model: {args.model_tag} ===")
        model_cls, weights_enum = get_model_info(args.model_tag)
        base_model = model_cls(weights=weights_enum)
        
        if args.stats_type == "custom":
            print("WARNING: You are using a standard torchvision model with CUSTOM stats.")
            print("         Standard models usually expect ImageNet stats.")
            
    elif args.local_weight_path:
        print(f"\n=== Loading Local Model: {args.model_tag} ===")
        base_model = load_model_safely_from_ckpt(
            args.local_weight_path,
            str(device),
            model_tag=args.model_tag,
            num_classes=1000
        )
    else:
        print("Error: You must specify either --torchvision OR --local_weight_path")
        sys.exit(1)

    # 4. Apply Normalization Wrapper & Evaluate
    model_wrapped = NormalizedModel(base_model, mean, std).to(device)
    model_wrapped.eval()

    print("\nStarting FGSM Evaluation...")
    accuracies = evaluate_robustness(model_wrapped, dataloader, device, epsilons)
    
    # 5. Output Result
    print(f"\nFinal Results ({args.model_tag}, {args.stats_type}):")
    for eps, acc in zip(epsilons, accuracies):
        print(f"  Eps {eps:.4f}: {acc:.4f}")

if __name__ == "__main__":
    main()