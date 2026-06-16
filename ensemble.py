#!/usr/bin/env python3
"""
Soft-voting ensemble inference script.

Supplementary material for the publication.

Author:
    Matheus M. Batista
    Universidade Federal de Itajubá

What this script does:
- Loads two segmentation model checkpoints.
- Runs inference on the test split of a tiled remote-sensing dataset.
- Combines model probabilities with user-defined soft-voting weights.
- Optionally applies test-time augmentation.
- Saves per-image metrics, summary metrics, and predicted masks.

Expected dataset layout:
    DATA_ROOT/
        test/
            images/*.tif
            masks/*_mask.tif

Example:
    python ensemble.py --data-root dataset/PLANET/P2/512 \
        --output-dir results/PLANET/P4_ESSEMBLE_TTA \
        --ckpt-model-a results/PLANET/P4_ESSEMBLE_TTA/best_segformer_mit_b2_Dice.pth \
        --ckpt-model-b results/PLANET/P4_ESSEMBLE_TTA/best_unetpp_resnet34_Dice.pth \
        --weight-a 0.5 --weight-b 0.5 --tta
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import rasterio
import segmentation_models_pytorch as smp
import torch
import torchmetrics
import ttach as tta
from torch.utils.data import DataLoader, Dataset

BATCH_SIZE = 4
THR = 0.5
NUM_WORKERS = 4
SAMPLE_CHANNELS = 4
EPS = 1e-7

HAS_VIS = False
VIS_CLIP_RANGE = (-1, 1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARCHITECTURES = {
    "unetpp_resnet34": {
        "cls": smp.UnetPlusPlus,
        "kwargs": {
            "encoder_name": "resnet34",
            "decoder_attention_type": None,
        },
    },
    "segformer_mit_b2": {
        "cls": smp.Segformer,
        "kwargs": {
            "encoder_name": "mit_b2",
        },
    },
    "manet_resnet34": {
        "cls": smp.MAnet,
        "kwargs": {
            "encoder_name": "resnet34",
            "decoder_pab_channels": 64,
        },
    },
}


class CoffeeDataset(Dataset):
    def __init__(
        self,
        file_list: Sequence[Path],
        has_vis: bool = HAS_VIS,
        vis_clip: tuple[float, float] = VIS_CLIP_RANGE,
    ) -> None:
        self.file_list = list(file_list)
        self.has_vis = has_vis
        self.vis_clip = vis_clip

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        img_path = self.file_list[idx]
        mask_dir = img_path.parent.parent / "masks"
        mask_path = mask_dir / f"{img_path.stem}_mask.tif"

        with rasterio.open(img_path) as src:
            raw_data = src.read().astype("float32")

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype("float32")

        raw_bands = raw_data[0:4, :, :]
        spectral_bands = raw_bands[[2, 1, 0, 3], :, :]
        spectral_bands = spectral_bands / 10000.0
        spectral_bands = np.clip(spectral_bands, 0.0, 1.0)

        if self.has_vis:
            vis_band = raw_data[4:, :, :]
            vis_band = np.clip(vis_band, self.vis_clip[0], self.vis_clip[1])
            image = np.concatenate([spectral_bands, vis_band], axis=0)
        else:
            image = spectral_bands

        image = np.nan_to_num(image, nan=0.0)
        mask = np.nan_to_num(mask, nan=0.0)

        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        return image_tensor, mask_tensor, str(img_path)


def build_model(model_name: str) -> torch.nn.Module:
    if model_name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {model_name}")

    config = ARCHITECTURES[model_name]
    model = config["cls"](
        **config["kwargs"],
        encoder_weights=None,
        in_channels=SAMPLE_CHANNELS,
        classes=1,
        activation=None,
    )

    return model


def load_checkpoint_model(
    checkpoint_path: Path,
    use_tta: bool,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model_name = checkpoint["model_name"]

    model = build_model(model_name)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE)
    model.eval()

    if use_tta:
        model = tta.SegmentationTTAWrapper(
            model,
            tta.aliases.d4_transform(),
            merge_mode="mean",
        )
        model.eval()

    return model, checkpoint


def save_prediction_mask(
    reference_img_path: str,
    output_path: str,
    pred_mask: np.ndarray,
) -> None:
    with rasterio.open(reference_img_path) as src:
        profile = src.profile.copy()

    profile.update(count=1, dtype=rasterio.uint8)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(pred_mask.astype(np.uint8), 1)


def build_metrics() -> dict[str, torchmetrics.Metric]:
    metrics = {
        "dataset_iou": torchmetrics.JaccardIndex(task="binary", threshold=THR).to(DEVICE),
        "dataset_f1": torchmetrics.F1Score(task="binary", threshold=THR).to(DEVICE),
        "dataset_accuracy": torchmetrics.Accuracy(task="binary", threshold=THR).to(DEVICE),
        "dataset_recall": torchmetrics.Recall(task="binary", threshold=THR).to(DEVICE),
        "dataset_precision": torchmetrics.Precision(task="binary", threshold=THR).to(DEVICE),
    }

    for metric in metrics.values():
        metric.reset()

    return metrics


def compute_sample_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    true_positive = (prediction * target).sum().item()
    pred_positive = prediction.sum().item()
    target_positive = target.sum().item()
    union = pred_positive + target_positive - true_positive

    false_positive = pred_positive - true_positive
    false_negative = target_positive - true_positive
    total_pixels = target.numel()
    true_negative = total_pixels - true_positive - false_positive - false_negative

    sample_iou = (true_positive + EPS) / (union + EPS)
    sample_f1 = (2 * true_positive + EPS) / (pred_positive + target_positive + EPS)
    sample_accuracy = (true_positive + true_negative + EPS) / (total_pixels + EPS)
    sample_recall = (true_positive + EPS) / (true_positive + false_negative + EPS)
    sample_precision = (true_positive + EPS) / (true_positive + false_positive + EPS)

    return {
        "iou": sample_iou,
        "f1": sample_f1,
        "accuracy": sample_accuracy,
        "recall": sample_recall,
        "precision": sample_precision,
        "pixels_gt": target_positive,
        "pixels_pred": pred_positive,
    }


def format_weight(weight: float) -> str:
    return f"{weight:.4f}".rstrip("0").rstrip(".")


def build_experiment_id(
    model_name_a: str,
    model_name_b: str,
    weight_a: float,
    weight_b: float,
    use_tta: bool,
) -> str:
    experiment_id = (
        f"ensemble_{model_name_a}_{format_weight(weight_a)}"
        f"_vs_{model_name_b}_{format_weight(weight_b)}"
    )

    if use_tta:
        experiment_id = f"{experiment_id}_TTA"

    return experiment_id


def build_summary(
    experiment_id: str,
    checkpoint_path_a: Path,
    checkpoint_path_b: Path,
    model_name_a: str,
    model_name_b: str,
    weight_a: float,
    weight_b: float,
    use_tta: bool,
    metrics: dict[str, torchmetrics.Metric],
    sample_results: pd.DataFrame,
    test_files: Sequence[Path],
) -> dict[str, Any]:
    summary = {
        "experiment_id": experiment_id,
        "model_name_a": model_name_a,
        "model_name_b": model_name_b,
        "checkpoint_path_a": str(checkpoint_path_a),
        "checkpoint_path_b": str(checkpoint_path_b),
        "weight_a": weight_a,
        "weight_b": weight_b,
        "tta_enabled": use_tta,
        "num_test_samples": len(test_files),
    }

    metric_name_map = {
        "iou": "dataset_iou",
        "f1": "dataset_f1",
        "accuracy": "dataset_accuracy",
        "recall": "dataset_recall",
        "precision": "dataset_precision",
    }

    for sample_metric_name, dataset_metric_name in metric_name_map.items():
        dataset_value = metrics[dataset_metric_name].compute().item()
        per_sample_avg = sample_results[sample_metric_name].mean()
        delta = per_sample_avg - dataset_value

        summary[dataset_metric_name] = dataset_value
        summary[f"per_sample_avg_{sample_metric_name}"] = per_sample_avg
        summary[f"delta_{sample_metric_name}"] = delta
        summary[f"abs_delta_{sample_metric_name}"] = abs(delta)

    return summary


def print_model_summary(summary: dict[str, Any]) -> None:
    row_key_map = {
        "Dataset": "dataset",
        "Per-sample": "per_sample_avg",
        "Delta": "delta",
    }

    print(f"Experiment: {summary['experiment_id']}")
    print(
        f"Model A: {summary['model_name_a']} | "
        f"weight={summary['weight_a']} | "
        f"ckpt={summary['checkpoint_path_a']}"
    )
    print(
        f"Model B: {summary['model_name_b']} | "
        f"weight={summary['weight_b']} | "
        f"ckpt={summary['checkpoint_path_b']}"
    )
    print(f"TTA enabled: {summary['tta_enabled']}")

    for label, key_prefix in row_key_map.items():
        row = (
            f"{label:<11}"
            f"IoU={summary[f'{key_prefix}_iou']:.4f} | "
            f"F1={summary[f'{key_prefix}_f1']:.4f} | "
            f"Acc={summary[f'{key_prefix}_accuracy']:.4f} | "
            f"Recall={summary[f'{key_prefix}_recall']:.4f} | "
            f"Precision={summary[f'{key_prefix}_precision']:.4f}"
        )
        print(row)

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run soft-voting ensemble inference on the test split.")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset root containing test/images and test/masks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where ensemble outputs will be saved.",
    )
    parser.add_argument(
        "--ckpt-model-a",
        type=Path,
        required=True,
        help="Path to the Model A checkpoint (.pth).",
    )
    parser.add_argument(
        "--ckpt-model-b",
        type=Path,
        required=True,
        help="Path to the Model B checkpoint (.pth).",
    )
    parser.add_argument(
        "--weight-a",
        type=float,
        default=0.5,
        help="Weight applied to Model A probabilities.",
    )
    parser.add_argument(
        "--weight-b",
        type=float,
        default=0.5,
        help="Weight applied to Model B probabilities.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable test-time augmentation with D4 spatial transforms for both models.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.data_root}")

    if not args.ckpt_model_a.exists():
        raise FileNotFoundError(f"Checkpoint for Model A not found: {args.ckpt_model_a}")

    if not args.ckpt_model_b.exists():
        raise FileNotFoundError(f"Checkpoint for Model B not found: {args.ckpt_model_b}")

    if args.weight_a < 0 or args.weight_b < 0:
        raise ValueError("Ensemble weights must be non-negative.")

    if args.weight_a == 0 and args.weight_b == 0:
        raise ValueError("At least one ensemble weight must be greater than zero.")


def evaluate_ensemble(
    checkpoint_path_a: Path,
    checkpoint_path_b: Path,
    output_dir: Path,
    weight_a: float,
    weight_b: float,
    use_tta: bool,
    test_loader: DataLoader,
    test_files: Sequence[Path],
) -> dict[str, Any]:
    model_a, checkpoint_a = load_checkpoint_model(checkpoint_path_a, use_tta)
    model_b, checkpoint_b = load_checkpoint_model(checkpoint_path_b, use_tta)

    model_name_a = checkpoint_a["model_name"]
    model_name_b = checkpoint_b["model_name"]
    experiment_id = build_experiment_id(
        model_name_a=model_name_a,
        model_name_b=model_name_b,
        weight_a=weight_a,
        weight_b=weight_b,
        use_tta=use_tta,
    )

    metrics = build_metrics()
    exp_output_dir = output_dir / experiment_id
    masks_output_dir = exp_output_dir / "pred_masks"

    exp_output_dir.mkdir(parents=True, exist_ok=True)
    masks_output_dir.mkdir(parents=True, exist_ok=True)

    raw_results: list[dict[str, Any]] = []

    with torch.no_grad():
        batch_size = test_loader.batch_size or 1

        for batch_idx, (imgs, masks, img_paths) in enumerate(test_loader):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            logits_a = model_a(imgs)
            logits_b = model_b(imgs)

            probs_a = torch.sigmoid(logits_a)
            probs_b = torch.sigmoid(logits_b)
            probs_ensemble = (probs_a * weight_a) + (probs_b * weight_b)
            pred_binary = (probs_ensemble > THR).float()

            for metric in metrics.values():
                metric.update(probs_ensemble, masks)

            for i in range(imgs.size(0)):
                global_idx = batch_idx * batch_size + i
                if global_idx >= len(test_files):
                    break

                img_path = img_paths[i]
                file_name = Path(img_path).name
                stem = Path(img_path).stem

                prediction = pred_binary[i, 0]
                target = masks[i, 0]
                sample_metrics = compute_sample_metrics(prediction, target)

                raw_results.append(
                    {
                        "image_id": file_name,
                        **sample_metrics,
                    }
                )

                pred_mask_np = prediction.cpu().numpy().astype(np.uint8)
                out_mask_path = masks_output_dir / f"{stem}_pred.tif"
                save_prediction_mask(img_path, str(out_mask_path), pred_mask_np)

    df_results = pd.DataFrame(raw_results)
    df_results.to_csv(exp_output_dir / f"results_{experiment_id}.csv", index=False)

    summary = build_summary(
        experiment_id=experiment_id,
        checkpoint_path_a=checkpoint_path_a,
        checkpoint_path_b=checkpoint_path_b,
        model_name_a=model_name_a,
        model_name_b=model_name_b,
        weight_a=weight_a,
        weight_b=weight_b,
        use_tta=use_tta,
        metrics=metrics,
        sample_results=df_results,
        test_files=test_files,
    )

    pd.DataFrame([summary]).to_csv(
        exp_output_dir / f"summary_{experiment_id}.csv",
        index=False,
    )

    del model_a
    del model_b
    torch.cuda.empty_cache()
    gc.collect()

    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    test_img_dir = args.data_root / "test" / "images"
    test_mask_dir = args.data_root / "test" / "masks"

    if not test_img_dir.exists():
        raise FileNotFoundError(f"Directory not found: {test_img_dir}")

    if not test_mask_dir.exists():
        raise FileNotFoundError(f"Directory not found: {test_mask_dir}")

    test_files = sorted(test_img_dir.glob("*.tif"))
    if not test_files:
        raise RuntimeError("No .tif images were found in the test split.")

    test_dataset = CoffeeDataset(test_files, has_vis=HAS_VIS)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print("=" * 60)
    print("TEST SET ENSEMBLE INFERENCE")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint A: {args.ckpt_model_a}")
    print(f"Checkpoint B: {args.ckpt_model_b}")
    print(f"Weight A: {args.weight_a}")
    print(f"Weight B: {args.weight_b}")
    print(f"TTA enabled: {args.tta}")
    print(f"Dataset root: {args.data_root}")
    print(f"Output directory: {output_dir}")
    print(f"Number of test images: {len(test_files)}")
    print("=" * 60)

    summary = evaluate_ensemble(
        checkpoint_path_a=args.ckpt_model_a,
        checkpoint_path_b=args.ckpt_model_b,
        output_dir=output_dir,
        weight_a=args.weight_a,
        weight_b=args.weight_b,
        use_tta=args.tta,
        test_loader=test_loader,
        test_files=test_files,
    )

    print("\nSUMMARY")
    print_model_summary(summary)
    print(f"Results saved to: {(output_dir / summary['experiment_id']).resolve()}")


if __name__ == "__main__":
    main()
