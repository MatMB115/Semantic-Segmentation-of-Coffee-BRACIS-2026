#!/usr/bin/env python3
"""
Single-model inference and evaluation script.

Supplementary material for the publication.

Author:
    Matheus M. Batista
    Universidade Federal de Itajubá

What this script does:
- Loads one or more segmentation model checkpoints from a checkpoint directory.
- Runs inference on the test split of a tiled remote-sensing dataset.
- Optionally applies test-time augmentation.
- Saves predicted masks, per-image metrics, summary metrics, and optional
  municipality-level metrics.

Expected dataset layout:
    DATA_ROOT/
        test/
            images/*.tif
            masks/*_mask.tif

Example:
    python inference.py --data-root dataset/PLANET/P1/256 \
        --checkpoint-dir results/PLANET/P1_BASELINE/working_09_03_2026_06_58_19 \
        --output-dir results/PLANET/P1_INFERENCE \
        --per-sample-agg mean --municipality-metrics
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any, Literal, Sequence

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

PerSampleAggregation = Literal["mean", "median"]


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


def extract_municipality(image_path: str | Path) -> str:
    stem = Path(image_path).stem
    prefix = "tile_"
    coord_marker = "_tile_x"

    if stem.startswith(prefix) and coord_marker in stem:
        return stem[len(prefix) : stem.index(coord_marker)]

    return "unknown"


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
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "total_pixels": total_pixels,
    }


def compute_aggregated_metrics(stats: pd.Series) -> dict[str, float]:
    true_positive = stats["true_positive"]
    false_positive = stats["false_positive"]
    false_negative = stats["false_negative"]
    true_negative = stats["true_negative"]
    total_pixels = stats["total_pixels"]
    pred_positive = stats["pixels_pred"]
    target_positive = stats["pixels_gt"]
    union = true_positive + false_positive + false_negative

    return {
        "dataset_iou": (true_positive + EPS) / (union + EPS),
        "dataset_f1": (2 * true_positive + EPS) / (pred_positive + target_positive + EPS),
        "dataset_accuracy": (true_positive + true_negative + EPS) / (total_pixels + EPS),
        "dataset_recall": (true_positive + EPS) / (true_positive + false_negative + EPS),
        "dataset_precision": (true_positive + EPS) / (true_positive + false_positive + EPS),
    }


def build_municipality_summary(
    experiment_id: str,
    model_name: str,
    loss_name: str,
    checkpoint_path: Path,
    sample_results: pd.DataFrame,
    per_sample_aggregation: PerSampleAggregation,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stat_columns = [
        "pixels_gt",
        "pixels_pred",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "total_pixels",
    ]
    metric_columns = ["iou", "f1", "accuracy", "recall", "precision"]

    for municipality, municipality_results in sample_results.groupby("municipality", sort=True):
        stats = municipality_results[stat_columns].sum()
        aggregated_metrics = compute_aggregated_metrics(stats)

        row: dict[str, Any] = {
            "experiment_id": experiment_id,
            "model_name": model_name,
            "loss_name": loss_name,
            "municipality": municipality,
            "num_test_samples": len(municipality_results),
            "checkpoint_path": str(checkpoint_path),
            "per_sample_aggregation": per_sample_aggregation,
            **aggregated_metrics,
            **{column: stats[column] for column in stat_columns},
        }

        for metric_name in metric_columns:
            if per_sample_aggregation == "mean":
                per_sample_value = municipality_results[metric_name].mean()
            elif per_sample_aggregation == "median":
                per_sample_value = municipality_results[metric_name].median()
            else:
                raise ValueError(f"Unsupported per-sample aggregation: {per_sample_aggregation}")

            dataset_value = row[f"dataset_{metric_name}"]
            delta = per_sample_value - dataset_value
            row[f"per_sample_{metric_name}"] = per_sample_value
            row[f"delta_{metric_name}"] = delta
            row[f"abs_delta_{metric_name}"] = abs(delta)

        rows.append(row)

    return pd.DataFrame(rows)


def build_summary(
    experiment_id: str,
    model_name: str,
    loss_name: str,
    checkpoint_path: Path,
    metrics: dict[str, torchmetrics.Metric],
    sample_results: pd.DataFrame,
    test_files: Sequence[Path],
    per_sample_aggregation: PerSampleAggregation,
) -> dict[str, Any]:
    summary = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "loss_name": loss_name,
        "num_test_samples": len(test_files),
        "checkpoint_path": str(checkpoint_path),
        "per_sample_aggregation": per_sample_aggregation,
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
        if per_sample_aggregation == "mean":
            per_sample_value = sample_results[sample_metric_name].mean()
        elif per_sample_aggregation == "median":
            per_sample_value = sample_results[sample_metric_name].median()
        else:
            raise ValueError(f"Unsupported per-sample aggregation: {per_sample_aggregation}")

        delta = per_sample_value - dataset_value

        summary[dataset_metric_name] = dataset_value
        summary[f"per_sample_{sample_metric_name}"] = per_sample_value
        summary[f"delta_{sample_metric_name}"] = delta
        summary[f"abs_delta_{sample_metric_name}"] = abs(delta)

    return summary


def print_model_summary(summary: dict[str, Any]) -> None:
    per_sample_label = f"Per-sample ({summary['per_sample_aggregation']})"
    row_key_map = {
        "Dataset": "dataset",
        per_sample_label: "per_sample",
        "Delta": "delta",
    }

    print(f"Model: {summary['experiment_id']}")

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


def evaluate_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    test_loader: DataLoader,
    test_files: Sequence[Path],
    use_tta: bool = False,
    per_sample_aggregation: PerSampleAggregation = "mean",
    municipality_metrics: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model_name = checkpoint["model_name"]
    loss_name = checkpoint.get("loss_name", "unknown")

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

    metrics = build_metrics()

    experiment_id = checkpoint_path.stem.replace("best_", "")
    if use_tta:
        experiment_id = f"{experiment_id}_TTA"

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

            logits = model(imgs)
            probs = torch.sigmoid(logits)
            pred_binary = (probs > THR).float()

            for metric in metrics.values():
                metric.update(probs, masks)

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
                        "municipality": extract_municipality(img_path),
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
        model_name=model_name,
        loss_name=loss_name,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
        sample_results=df_results,
        test_files=test_files,
        per_sample_aggregation=per_sample_aggregation,
    )

    pd.DataFrame([summary]).to_csv(
        exp_output_dir / f"summary_{experiment_id}_{per_sample_aggregation}.csv",
        index=False,
    )

    municipality_summary = None
    if municipality_metrics:
        municipality_summary = build_municipality_summary(
            experiment_id=experiment_id,
            model_name=model_name,
            loss_name=loss_name,
            checkpoint_path=checkpoint_path,
            sample_results=df_results,
            per_sample_aggregation=per_sample_aggregation,
        )
        municipality_summary.to_csv(
            exp_output_dir / f"municipality_metrics_{experiment_id}_{per_sample_aggregation}.csv",
            index=False,
        )

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return summary, municipality_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on the test split.")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset root containing test/images and test/masks.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing best_*.pth checkpoint files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where inference outputs will be saved.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable test-time augmentation with D4 spatial transforms.",
    )
    parser.add_argument(
        "--per-sample-agg",
        choices=("mean", "median"),
        default="mean",
        help="Aggregation used to summarize per-sample metrics.",
    )
    parser.add_argument(
        "--municipality-metrics",
        action="store_true",
        help="Compute and save metrics grouped by municipality parsed from image names.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.data_root}")

    if not args.checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint_dir}")

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

    checkpoint_files = sorted(args.checkpoint_dir.glob("best_*.pth"))
    if not checkpoint_files:
        raise RuntimeError(
            f"No checkpoints were found in: {args.checkpoint_dir}\n"
            "Expected something like: best_unetpp_resnet34_Dice.pth"
        )

    print("=" * 60)
    print("TEST SET INFERENCE")
    print(f"Device: {DEVICE}")
    print(f"TTA enabled: {args.tta}")
    print(f"Per-sample aggregation: {args.per_sample_agg}")
    print(f"Municipality metrics enabled: {args.municipality_metrics}")
    print(f"Dataset root: {args.data_root}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of test images: {len(test_files)}")
    print(f"Number of checkpoints: {len(checkpoint_files)}")
    print("=" * 60)

    all_summaries: list[dict[str, Any]] = []
    all_municipality_summaries: list[pd.DataFrame] = []

    for checkpoint_path in checkpoint_files:
        print(f"\nProcessing checkpoint: {checkpoint_path.name}")

        summary, municipality_summary = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            test_loader=test_loader,
            test_files=test_files,
            use_tta=args.tta,
            per_sample_aggregation=args.per_sample_agg,
            municipality_metrics=args.municipality_metrics,
        )
        all_summaries.append(summary)
        if municipality_summary is not None:
            all_municipality_summaries.append(municipality_summary)
        print_model_summary(summary)

    df_all = pd.DataFrame(all_summaries)
    df_all = df_all.sort_values(by="dataset_iou", ascending=False)
    summary_filename = (
        f"all_models_test_summary_{args.per_sample_agg}_tta.csv"
        if args.tta
        else f"all_models_test_summary_{args.per_sample_agg}.csv"
    )
    df_all.to_csv(output_dir / summary_filename, index=False)

    if all_municipality_summaries:
        df_all_municipalities = pd.concat(all_municipality_summaries, ignore_index=True)
        municipality_summary_filename = (
            f"all_models_municipality_summary_{args.per_sample_agg}_tta.csv"
            if args.tta
            else f"all_models_municipality_summary_{args.per_sample_agg}.csv"
        )
        df_all_municipalities = df_all_municipalities.sort_values(
            by=["municipality", "dataset_iou"],
            ascending=[True, False],
        )
        df_all_municipalities.to_csv(output_dir / municipality_summary_filename, index=False)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    for summary in df_all.to_dict(orient="records"):
        print_model_summary(summary)
    print("=" * 60)
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
