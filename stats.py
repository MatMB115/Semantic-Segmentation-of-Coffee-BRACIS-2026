#!/usr/bin/env python3
"""
Statistical analysis script for experiment phases.

Supplementary material for the publication.

Author:
    Matheus M. Batista
    Universidade Federal de Itajubá

What this script does:
- Reads a JSON configuration describing experiment phases, result CSV files,
  metrics, statistical design, tests, alpha level, and p-value adjustment.
- Loads model-level metric values from CSV files.
- Runs paired or independent non-parametric statistical tests.
- Writes a JSON report with global and pairwise test results.

Supported designs:
- paired
- independent

Supported tests:
Paired design:
- Friedman: global comparison for 3 or more related groups
- Wilcoxon: pairwise comparison for 2 related groups, or all related pairs

Independent design:
- Kruskal-Wallis: global comparison for 3 or more independent groups
- Mann-Whitney U: pairwise comparison for 2 independent groups, or all independent pairs

Multiple-comparison adjustment:
- none
- holm
- bonferroni

Notes:
- Wilcoxon is only valid for paired data
- Mann-Whitney U is only valid for independent data
- Friedman is only valid for paired data with at least 3 groups
- Kruskal-Wallis is only valid for independent data with at least 3 groups
- If there is only one pairwise comparison, no p-value adjustment is applied

Usage:
    python stats.py experiment_config.json
    python stats.py experiment_config.json --output analysis_results.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from scipy.stats import friedmanchisquare, kruskal, mannwhitneyu, wilcoxon


VALID_ADJUSTMENTS = {"none", "holm", "bonferroni"}
VALID_DESIGNS = {"paired", "independent"}
DEFAULT_ALPHA = 0.05
IDENTIFIER_COLUMNS = ("image_id",)


def holm_correction(p_values: List[float]) -> List[float]:
    """
    Apply Holm correction to a list of p-values.
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m

    running_max = 0.0
    for rank, (original_idx, p_value) in enumerate(indexed):
        adjusted_p = min((m - rank) * p_value, 1.0)
        running_max = max(running_max, adjusted_p)
        adjusted[original_idx] = running_max

    return adjusted


def bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Apply Bonferroni correction to a list of p-values.
    """
    m = len(p_values)
    return [min(p * m, 1.0) for p in p_values]


def apply_p_adjustment(p_values: List[float], method: str) -> List[float]:
    """
    Apply the selected multiple-comparison adjustment method.
    """
    if method == "none":
        return list(p_values)
    if method == "holm":
        return holm_correction(p_values)
    if method == "bonferroni":
        return bonferroni_correction(p_values)

    raise ValueError(
        f"Invalid p-value adjustment method: {method}. "
        f"Valid options: {sorted(VALID_ADJUSTMENTS)}"
    )


def validate_series_alignment(model_data: Dict[str, pd.Series]) -> None:
    """
    Ensure all series have the same number of paired samples.
    """
    lengths = {name: len(series) for name, series in model_data.items()}
    unique_lengths = set(lengths.values())

    if len(unique_lengths) != 1:
        raise ValueError(
            f"Paired analysis requires the same number of samples in every group. "
            f"Received: {lengths}"
        )


def build_order_assumed_pairing_info(reason: str) -> dict:
    return {
        "paired_by_identifier": False,
        "identifier_column": None,
        "reordered_by_identifier": False,
        "warning": reason,
    }


def find_common_identifier_column(model_frames: Dict[str, pd.DataFrame]) -> Optional[str]:
    for column in IDENTIFIER_COLUMNS:
        if all(column in df.columns for df in model_frames.values()):
            return column
    return None


def validate_and_align_by_identifier(
    phase_name: str,
    model_frames: Dict[str, pd.DataFrame],
    metric_name: str,
) -> Tuple[Dict[str, pd.Series], dict]:
    identifier_column = find_common_identifier_column(model_frames)

    if identifier_column is None:
        model_data = {
            model_name: df[metric_name].reset_index(drop=True)
            for model_name, df in model_frames.items()
        }
        return model_data, build_order_assumed_pairing_info(
            "No common identifier column was found; pairing was assumed by row order."
        )

    reference_model = next(iter(model_frames))
    reference_ids = model_frames[reference_model][identifier_column]

    if reference_ids.isna().any():
        raise ValueError(
            f"{phase_name}: missing identifiers found in column '{identifier_column}' "
            f"for model '{reference_model}'."
        )

    if reference_ids.duplicated().any():
        duplicates = reference_ids[reference_ids.duplicated()].unique().tolist()
        raise ValueError(
            f"{phase_name}: duplicate identifiers found in column '{identifier_column}' "
            f"for model '{reference_model}': {duplicates[:10]}"
        )

    reference_index = pd.Index(reference_ids.astype(str), name=identifier_column)
    reference_set = set(reference_index)
    model_data: Dict[str, pd.Series] = {}
    reordered = False

    for model_name, df in model_frames.items():
        ids = df[identifier_column]

        if ids.isna().any():
            raise ValueError(
                f"{phase_name}: missing identifiers found in column '{identifier_column}' "
                f"for model '{model_name}'."
            )

        if ids.duplicated().any():
            duplicates = ids[ids.duplicated()].unique().tolist()
            raise ValueError(
                f"{phase_name}: duplicate identifiers found in column '{identifier_column}' "
                f"for model '{model_name}': {duplicates[:10]}"
            )

        current_index = pd.Index(ids.astype(str), name=identifier_column)
        current_set = set(current_index)

        if current_set != reference_set:
            missing = sorted(reference_set - current_set)
            extra = sorted(current_set - reference_set)
            raise ValueError(
                f"{phase_name}: identifiers in column '{identifier_column}' do not match "
                f"between '{reference_model}' and '{model_name}'. "
                f"Missing from {model_name}: {missing[:10]}; extra in {model_name}: {extra[:10]}"
            )

        aligned = df.copy()
        aligned.index = current_index

        if list(current_index) != list(reference_index):
            reordered = True
            aligned = aligned.loc[reference_index]

        model_data[model_name] = aligned[metric_name].reset_index(drop=True)

    return model_data, {
        "paired_by_identifier": True,
        "identifier_column": identifier_column,
        "reordered_by_identifier": reordered,
        "warning": None,
    }


def load_phase_data(
    phase_name: str,
    phase_config: dict,
) -> Tuple[str, Dict[str, pd.Series], dict]:
    """
    Load metric data for one phase.

    Returns:
    metric_name: selected metric column
    model_data: dictionary mapping model name to metric series
    """
    metric_name = phase_config.get("metric", "iou")
    models = phase_config.get("models", {})

    if len(models) < 2:
        raise ValueError(
            f"{phase_name}: at least 2 models are required."
        )

    model_frames: Dict[str, pd.DataFrame] = {}

    for model_name, csv_path in models.items():
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"{phase_name}: file not found for model '{model_name}': {path}"
            )

        df = pd.read_csv(path)

        if metric_name not in df.columns:
            raise KeyError(
                f"{phase_name}: column '{metric_name}' not found in {path}. "
                f"Available columns: {list(df.columns)}"
            )

        series = df[metric_name]

        if series.isna().any():
            raise ValueError(
                f"{phase_name}: missing values found in metric '{metric_name}' "
                f"for model '{model_name}'."
            )

        model_frames[model_name] = df

    if phase_config.get("design", "paired").lower() == "paired":
        model_data, pairing_info = validate_and_align_by_identifier(
            phase_name=phase_name,
            model_frames=model_frames,
            metric_name=metric_name,
        )
    else:
        model_data = {
            model_name: df[metric_name].reset_index(drop=True)
            for model_name, df in model_frames.items()
        }
        pairing_info = {
            "paired_by_identifier": None,
            "identifier_column": None,
            "reordered_by_identifier": False,
            "warning": None,
        }

    return metric_name, model_data, pairing_info


def get_phase_alpha(phase_config: dict, global_alpha: float = DEFAULT_ALPHA) -> float:
    tests_config = phase_config.get("tests", {})
    alpha = phase_config.get("alpha", tests_config.get("alpha", global_alpha))
    alpha = float(alpha)

    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1. Received: {alpha}")

    return alpha


def validate_test_plan(
    phase_name: str,
    phase_config: dict,
    model_data: Dict[str, pd.Series],
    global_alpha: float = DEFAULT_ALPHA,
) -> dict:
    """
    Validate and normalize the statistical plan for one phase.
    """
    tests_config = phase_config.get("tests", {})
    design = phase_config.get("design", "paired").lower()

    if design not in VALID_DESIGNS:
        raise ValueError(
            f"{phase_name}: invalid design '{design}'. "
            f"Valid options: {sorted(VALID_DESIGNS)}"
        )

    normalized = {
        "design": design,
        "friedman": bool(tests_config.get("friedman", False)),
        "kruskal_wallis": bool(tests_config.get("kruskal_wallis", False)),
        "wilcoxon": bool(tests_config.get("wilcoxon", False)),
        "mann_whitney": bool(tests_config.get("mann_whitney", False)),
        "alpha": get_phase_alpha(phase_config, global_alpha),
        "run_pairwise_if_global_not_significant": bool(
            phase_config.get(
                "run_pairwise_if_global_not_significant",
                tests_config.get("run_pairwise_if_global_not_significant", False),
            )
        ),
        "p_adjustment": tests_config.get(
            "p_adjustment",
            tests_config.get("correction", "none")
        ).lower(),
    }

    if normalized["p_adjustment"] not in VALID_ADJUSTMENTS:
        raise ValueError(
            f"{phase_name}: invalid p_adjustment '{normalized['p_adjustment']}'. "
            f"Valid options: {sorted(VALID_ADJUSTMENTS)}"
        )

    n_models = len(model_data)

    if design == "paired":
        validate_series_alignment(model_data)

        if normalized["kruskal_wallis"] or normalized["mann_whitney"]:
            raise ValueError(
                f"{phase_name}: independent-sample tests were requested for paired data."
            )

        if normalized["friedman"] and n_models < 3:
            raise ValueError(
                f"{phase_name}: Friedman requires at least 3 related groups. "
                f"Received {n_models}."
            )

    if design == "independent":
        if normalized["friedman"] or normalized["wilcoxon"]:
            raise ValueError(
                f"{phase_name}: paired-sample tests were requested for independent data."
            )

        if normalized["kruskal_wallis"] and n_models < 3:
            raise ValueError(
                f"{phase_name}: Kruskal-Wallis requires at least 3 independent groups. "
                f"Received {n_models}."
            )

    return normalized


def run_friedman(model_data: Dict[str, pd.Series], alpha: float = DEFAULT_ALPHA) -> dict:
    """
    Run Friedman across 3 or more related groups.
    """
    ordered_names = list(model_data.keys())
    samples = [model_data[name] for name in ordered_names]
    statistic, p_value = friedmanchisquare(*samples)

    return {
        "test": "friedman",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant_at_0_05": bool(p_value < 0.05),
        "significant_at_alpha": bool(p_value < alpha),
        "alpha": alpha,
        "groups_order": ordered_names,
    }


def run_kruskal_wallis(model_data: Dict[str, pd.Series], alpha: float = DEFAULT_ALPHA) -> dict:
    """
    Run Kruskal-Wallis across 3 or more independent groups.
    """
    ordered_names = list(model_data.keys())
    samples = [model_data[name] for name in ordered_names]
    statistic, p_value = kruskal(*samples)

    return {
        "test": "kruskal_wallis",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant_at_0_05": bool(p_value < 0.05),
        "significant_at_alpha": bool(p_value < alpha),
        "alpha": alpha,
        "groups_order": ordered_names,
    }


def run_pairwise_wilcoxon(
    model_data: Dict[str, pd.Series],
    p_adjustment: str = "none",
    alpha: float = DEFAULT_ALPHA,
) -> List[dict]:
    """
    Run pairwise Wilcoxon tests for paired data.
    """
    pairs = list(itertools.combinations(model_data.keys(), 2))
    raw_results = []

    for model_a, model_b in pairs:
        series_a = model_data[model_a]
        series_b = model_data[model_b]

        statistic, p_value = wilcoxon(series_a, series_b)

        mean_a = float(series_a.mean())
        mean_b = float(series_b.mean())

        if mean_a > mean_b:
            higher_mean_group = model_a
        elif mean_b > mean_a:
            higher_mean_group = model_b
        else:
            higher_mean_group = "tie"

        raw_results.append({
            "test": "wilcoxon",
            "comparison": f"{model_a} vs {model_b}",
            "group_a": model_a,
            "group_b": model_b,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "mean_group_a": mean_a,
            "mean_group_b": mean_b,
            "higher_mean_group": higher_mean_group,
        })

    if len(raw_results) <= 1:
        for result in raw_results:
            result["p_adjustment"] = "none"
            result["adjusted_p_value"] = result["p_value"]
            result["significant_at_0_05"] = bool(result["adjusted_p_value"] < 0.05)
            result["significant_at_alpha"] = bool(result["adjusted_p_value"] < alpha)
            result["alpha"] = alpha
        return raw_results

    adjusted = apply_p_adjustment(
        [result["p_value"] for result in raw_results],
        p_adjustment,
    )

    for result, adjusted_p in zip(raw_results, adjusted):
        result["p_adjustment"] = p_adjustment
        result["adjusted_p_value"] = float(adjusted_p)
        result["significant_at_0_05"] = bool(adjusted_p < 0.05)
        result["significant_at_alpha"] = bool(adjusted_p < alpha)
        result["alpha"] = alpha

    return raw_results


def run_pairwise_mann_whitney(
    model_data: Dict[str, pd.Series],
    p_adjustment: str = "none",
    alpha: float = DEFAULT_ALPHA,
) -> List[dict]:
    """
    Run pairwise Mann-Whitney U tests for independent data.
    """
    pairs = list(itertools.combinations(model_data.keys(), 2))
    raw_results = []

    for model_a, model_b in pairs:
        series_a = model_data[model_a]
        series_b = model_data[model_b]

        statistic, p_value = mannwhitneyu(series_a, series_b, alternative="two-sided")

        mean_a = float(series_a.mean())
        mean_b = float(series_b.mean())

        if mean_a > mean_b:
            higher_mean_group = model_a
        elif mean_b > mean_a:
            higher_mean_group = model_b
        else:
            higher_mean_group = "tie"

        raw_results.append({
            "test": "mann_whitney_u",
            "comparison": f"{model_a} vs {model_b}",
            "group_a": model_a,
            "group_b": model_b,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "mean_group_a": mean_a,
            "mean_group_b": mean_b,
            "higher_mean_group": higher_mean_group,
        })

    if len(raw_results) <= 1:
        for result in raw_results:
            result["p_adjustment"] = "none"
            result["adjusted_p_value"] = result["p_value"]
            result["significant_at_0_05"] = bool(result["adjusted_p_value"] < 0.05)
            result["significant_at_alpha"] = bool(result["adjusted_p_value"] < alpha)
            result["alpha"] = alpha
        return raw_results

    adjusted = apply_p_adjustment(
        [result["p_value"] for result in raw_results],
        p_adjustment,
    )

    for result, adjusted_p in zip(raw_results, adjusted):
        result["p_adjustment"] = p_adjustment
        result["adjusted_p_value"] = float(adjusted_p)
        result["significant_at_0_05"] = bool(adjusted_p < 0.05)
        result["significant_at_alpha"] = bool(adjusted_p < alpha)
        result["alpha"] = alpha

    return raw_results


def build_pairwise_decision(
    normalized_tests: dict,
    n_groups: int,
    friedman_result: Optional[dict],
) -> dict:
    design = normalized_tests["design"]
    alpha = normalized_tests["alpha"]
    wilcoxon_requested = normalized_tests["wilcoxon"]
    override = normalized_tests["run_pairwise_if_global_not_significant"]

    decision: dict[str, Any] = {
        "alpha": alpha,
        "global_test_required": False,
        "global_test_name": None,
        "global_test_significant": None,
        "pairwise_wilcoxon_requested": wilcoxon_requested,
        "pairwise_wilcoxon_executed": False,
        "pairwise_wilcoxon_skipped": False,
        "skip_reason": None,
        "run_pairwise_if_global_not_significant": override,
    }

    if design != "paired" or not wilcoxon_requested:
        return decision

    if n_groups == 2:
        decision["pairwise_wilcoxon_executed"] = True
        return decision

    decision["global_test_required"] = True
    decision["global_test_name"] = "friedman"

    if friedman_result is None:
        if override:
            decision["pairwise_wilcoxon_executed"] = True
            decision["override_reason"] = (
                "Wilcoxon was run without a Friedman result because "
                "run_pairwise_if_global_not_significant is true."
            )
        else:
            decision["pairwise_wilcoxon_skipped"] = True
            decision["skip_reason"] = (
                "Friedman test was required before pairwise Wilcoxon but was not run."
            )
        return decision

    decision["global_test_significant"] = friedman_result["p_value"] < alpha

    if decision["global_test_significant"]:
        decision["pairwise_wilcoxon_executed"] = True
    elif override:
        decision["pairwise_wilcoxon_executed"] = True
        decision["override_reason"] = (
            f"Friedman test was not significant at alpha = {alpha}, "
            "but run_pairwise_if_global_not_significant is true."
        )
    else:
        decision["pairwise_wilcoxon_skipped"] = True
        decision["skip_reason"] = (
            f"Friedman test was not significant at alpha = {alpha}."
        )

    return decision


def summarize_phase(
    phase_name: str,
    metric_name: str,
    model_data: Dict[str, pd.Series],
    phase_config: dict,
    pairing_info: Optional[dict] = None,
    global_alpha: float = DEFAULT_ALPHA,
) -> dict:
    """
    Build the complete summary for one phase.
    """
    means = {name: float(series.mean()) for name, series in model_data.items()}
    stds = {name: float(series.std()) for name, series in model_data.items()}
    medians = {name: float(series.median()) for name, series in model_data.items()}
    sample_sizes = {name: int(len(series)) for name, series in model_data.items()}

    normalized_tests = validate_test_plan(
        phase_name=phase_name,
        phase_config=phase_config,
        model_data=model_data,
        global_alpha=global_alpha,
    )
    alpha = normalized_tests["alpha"]

    friedman_result: Optional[dict] = None
    kruskal_result: Optional[dict] = None
    wilcoxon_results: List[dict] = []
    mann_whitney_results: List[dict] = []

    if normalized_tests["friedman"]:
        friedman_result = run_friedman(model_data, alpha=alpha)

    if normalized_tests["kruskal_wallis"]:
        kruskal_result = run_kruskal_wallis(model_data, alpha=alpha)

    wilcoxon_decision = build_pairwise_decision(
        normalized_tests=normalized_tests,
        n_groups=len(model_data),
        friedman_result=friedman_result,
    )

    if wilcoxon_decision["pairwise_wilcoxon_executed"]:
        wilcoxon_results = run_pairwise_wilcoxon(
            model_data=model_data,
            p_adjustment=normalized_tests["p_adjustment"],
            alpha=alpha,
        )

    if normalized_tests["mann_whitney"]:
        mann_whitney_results = run_pairwise_mann_whitney(
            model_data=model_data,
            p_adjustment=normalized_tests["p_adjustment"],
            alpha=alpha,
        )

    ranking = sorted(means.items(), key=lambda x: x[1], reverse=True)
    global_test_result = friedman_result or kruskal_result
    global_test_significant = (
        global_test_result["significant_at_alpha"]
        if global_test_result is not None
        else wilcoxon_decision["global_test_significant"]
    )

    return {
        "phase": phase_name,
        "metric": metric_name,
        "design": normalized_tests["design"],
        "alpha": alpha,
        "n_groups": len(model_data),
        "sample_sizes": sample_sizes,
        "pairing": pairing_info or {},
        "means": means,
        "stds": stds,
        "medians": medians,
        "ranking_by_mean": ranking,
        "tests_requested": normalized_tests,
        "global_test_executed": global_test_result is not None,
        "global_test_required": wilcoxon_decision["global_test_required"],
        "global_test_significant": global_test_significant,
        "pairwise_wilcoxon_executed": wilcoxon_decision["pairwise_wilcoxon_executed"],
        "pairwise_wilcoxon_skipped": wilcoxon_decision["pairwise_wilcoxon_skipped"],
        "skip_reason": wilcoxon_decision["skip_reason"],
        "pairwise_decision": wilcoxon_decision,
        "friedman": friedman_result,
        "kruskal_wallis": kruskal_result,
        "pairwise_wilcoxon": wilcoxon_results,
        "pairwise_mann_whitney": mann_whitney_results,
    }


def print_phase_report(phase_result: dict) -> None:
    """
    Print a readable terminal report for one phase.
    """
    print("=" * 80)
    print(f"Phase: {phase_result['phase']}")
    print(f"Metric: {phase_result['metric']}")
    print(f"Design: {phase_result['design']}")
    print(f"Alpha: {phase_result['alpha']}")
    print(f"Number of groups: {phase_result['n_groups']}")
    pairing_warning = phase_result.get("pairing", {}).get("warning")
    if pairing_warning:
        print(f"Pairing warning: {pairing_warning}")
    print("Sample sizes by group:")
    for group, n_value in phase_result["sample_sizes"].items():
        print(f"  {group}: {n_value}")
    print()

    print("Mean metric by group:")
    for group, value in phase_result["ranking_by_mean"]:
        print(f"  {group}: {value:.6f}")
    print()

    if phase_result["friedman"] is not None:
        item = phase_result["friedman"]
        print("Friedman test:")
        print(f"  Statistic: {item['statistic']:.6f}")
        print(f"  p-value: {item['p_value']:.6e}")
        print(f"  Significant at 0.05: {item['significant_at_0_05']}")
        print(f"  Significant at alpha: {item['significant_at_alpha']}")
        print()

    if phase_result["kruskal_wallis"] is not None:
        item = phase_result["kruskal_wallis"]
        print("Kruskal-Wallis test:")
        print(f"  Statistic: {item['statistic']:.6f}")
        print(f"  p-value: {item['p_value']:.6e}")
        print(f"  Significant at 0.05: {item['significant_at_0_05']}")
        print(f"  Significant at alpha: {item['significant_at_alpha']}")
        print()

    if phase_result.get("pairwise_wilcoxon_skipped"):
        print("Pairwise Wilcoxon tests: skipped")
        print(f"  Reason: {phase_result.get('skip_reason')}")
        print()

    if phase_result["pairwise_wilcoxon"]:
        print("Pairwise Wilcoxon tests:")
        for item in phase_result["pairwise_wilcoxon"]:
            print(
                f"  {item['comparison']}: "
                f"raw p={item['p_value']:.6e}, "
                f"adjusted p={item['adjusted_p_value']:.6e}, "
                f"adjustment={item['p_adjustment']}, "
                f"significant={item['significant_at_alpha']}, "
                f"higher mean={item['higher_mean_group']}"
            )
        print()

    if phase_result["pairwise_mann_whitney"]:
        print("Pairwise Mann-Whitney U tests:")
        for item in phase_result["pairwise_mann_whitney"]:
            print(
                f"  {item['comparison']}: "
                f"raw p={item['p_value']:.6e}, "
                f"adjusted p={item['adjusted_p_value']:.6e}, "
                f"adjustment={item['p_adjustment']}, "
                f"significant={item['significant_at_alpha']}, "
                f"higher mean={item['higher_mean_group']}"
            )
        print()


def get_global_alpha(config: dict) -> float:
    alpha = float(config.get("alpha", DEFAULT_ALPHA))

    if not 0 < alpha < 1:
        raise ValueError(f"Global alpha must be between 0 and 1. Received: {alpha}")

    return alpha


def iter_phase_configs(config: dict) -> Dict[str, dict]:
    if "phases" in config:
        phases = config["phases"]
        if not isinstance(phases, dict):
            raise ValueError("'phases' must be an object mapping phase names to configurations.")
        return phases

    return {
        phase_name: phase_config
        for phase_name, phase_config in config.items()
        if isinstance(phase_config, dict)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run statistical analysis for experiment phases."
    )
    parser.add_argument(
        "config_json",
        type=str,
        help="Path to the JSON configuration file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis_results.json",
        help="Path to the output JSON results file."
    )
    args = parser.parse_args()

    config_path = Path(args.config_json)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    global_alpha = get_global_alpha(config)
    phases_config = iter_phase_configs(config)
    phase_results = []

    for phase_name, phase_config in phases_config.items():
        if not phase_config:
            print(f"{phase_name}: empty configuration. Skipping.")
            continue

        models = phase_config.get("models", {})
        if not models:
            print(f"{phase_name}: no groups defined. Skipping.")
            continue

        metric_name, model_data, pairing_info = load_phase_data(phase_name, phase_config)
        phase_result = summarize_phase(
            phase_name=phase_name,
            metric_name=metric_name,
            model_data=model_data,
            phase_config=phase_config,
            pairing_info=pairing_info,
            global_alpha=global_alpha,
        )
        phase_results.append(phase_result)
        print_phase_report(phase_result)

    final_output = {
        "config_file": str(config_path),
        "alpha": global_alpha,
        "results": phase_results,
    }

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
