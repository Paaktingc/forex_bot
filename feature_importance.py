"""
feature_importance.py

Change summary:
- Adds an out-of-sample permutation-importance pipeline for the signal-time
  meta-label dataset.
- Injects synthetic noise features, runs a label-shuffle leakage test, and
  removes features that do not beat the noise baseline.
- Uses purged and embargoed time-series folds so diagnostics stay causally
  aligned with the research pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight


@dataclass
class PipelineConfig:
    """
    Tunable parameters for the feature-importance diagnostic.
    """

    n_folds: int = 5
    purge_bars: int = 10
    embargo_bars: int = 20

    model_type: str = "gbm"
    model_params: Dict[str, object] = field(
        default_factory=lambda: {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "min_samples_leaf": 20,
            "random_state": 42,
        }
    )

    n_permutations: int = 10
    n_noise_features: int = 5
    n_shuffle_trials: int = 5

    max_features_to_keep: int = 10
    min_importance_vs_noise: float = 1.5
    max_correlation: float = 0.7


class PurgedTimeSeriesCV:
    """
    Sequential folds with an explicit purge and embargo gap between train and test.
    """

    def __init__(self, n_splits: int, purge: int, embargo: int):
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo

    def split(self, X: pd.DataFrame | np.ndarray):
        n = len(X)
        if n < 50:
            return

        fold_size = max(10, n // (self.n_splits + 1))
        for fold in range(1, self.n_splits + 1):
            split_pos = fold_size * fold
            train_end = max(0, split_pos - self.purge)
            test_start = split_pos + self.embargo
            test_end = min(test_start + fold_size, n)

            if train_end <= 0 or test_start >= n or test_end - test_start < 10:
                continue

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx


class FeatureImportancePipeline:
    """
    Permutation-importance and leakage-detection workflow for signal-time features.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None,
        config: Optional[PipelineConfig] = None,
    ):
        if X is None or X.empty:
            raise ValueError("Feature matrix X is empty.")
        if y is None or len(y) == 0:
            raise ValueError("Target vector y is empty.")

        self.cfg = config or PipelineConfig()
        self.X = X.copy()
        self.y = y.copy().astype(int)
        self.feature_names = list(feature_names or self.X.columns)

        self._inject_noise()

    def _inject_noise(self) -> None:
        np.random.seed(42)
        for idx in range(self.cfg.n_noise_features):
            column = f"__noise_{idx}__"
            self.X[column] = np.random.randn(len(self.X))
            self.feature_names.append(column)

    def _get_model(self):
        if self.cfg.model_type == "gbm":
            return GradientBoostingClassifier(**self.cfg.model_params)
        if self.cfg.model_type == "rf":
            params = dict(self.cfg.model_params)
            params.pop("learning_rate", None)
            return RandomForestClassifier(**params)
        raise ValueError(f"Unknown model_type: {self.cfg.model_type}")

    def _permutation_importance(self) -> Dict[str, object]:
        cv = PurgedTimeSeriesCV(self.cfg.n_folds, self.cfg.purge_bars, self.cfg.embargo_bars)
        X_array = self.X.values
        y_array = self.y.values

        feature_scores = {feature: [] for feature in self.feature_names}
        baseline_scores: list[float] = []
        fold_details: list[dict[str, object]] = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(self.X), start=1):
            model = self._get_model()
            model.fit(X_array[train_idx], y_array[train_idx])

            baseline = accuracy_score(y_array[test_idx], model.predict(X_array[test_idx]))
            baseline_scores.append(float(baseline))
            fold_details.append(
                {
                    "fold": fold_idx,
                    "train_size": int(len(train_idx)),
                    "test_size": int(len(test_idx)),
                    "baseline_acc": round(float(baseline) * 100, 1),
                }
            )

            for feature_idx, feature_name in enumerate(self.feature_names):
                drops: list[float] = []
                for _ in range(self.cfg.n_permutations):
                    X_perm = X_array[test_idx].copy()
                    np.random.shuffle(X_perm[:, feature_idx])
                    perm_score = accuracy_score(y_array[test_idx], model.predict(X_perm))
                    drops.append(float(baseline - perm_score))
                feature_scores[feature_name].append(float(np.mean(drops)))

        importance = {}
        for feature_name, scores in feature_scores.items():
            importance[feature_name] = {
                "mean_drop": float(np.mean(scores)) if scores else 0.0,
                "std_drop": float(np.std(scores)) if scores else 0.0,
                "per_fold": scores,
            }

        return {
            "importance": importance,
            "baseline_scores": baseline_scores,
            "fold_details": fold_details,
        }

    def _label_shuffle_test(self) -> Dict[str, object]:
        cv = PurgedTimeSeriesCV(self.cfg.n_folds, self.cfg.purge_bars, self.cfg.embargo_bars)
        X_array = self.X.values
        y_array = self.y.values

        old_shuffle_scores: list[float] = []
        balanced_shuffle_scores: list[float] = []
        for _ in range(self.cfg.n_shuffle_trials):
            shuffled = y_array.copy()
            np.random.shuffle(shuffled)

            old_fold_scores: list[float] = []
            balanced_fold_scores: list[float] = []
            for train_idx, test_idx in cv.split(self.X):
                old_model = self._get_model()
                old_model.fit(X_array[train_idx], shuffled[train_idx])
                old_fold_scores.append(float(accuracy_score(shuffled[test_idx], old_model.predict(X_array[test_idx]))))

                balanced_model = self._get_model()
                weights = compute_sample_weight("balanced", shuffled[train_idx])
                balanced_model.fit(X_array[train_idx], shuffled[train_idx], sample_weight=weights)
                balanced_fold_scores.append(
                    float(accuracy_score(shuffled[test_idx], balanced_model.predict(X_array[test_idx])))
                )

            if old_fold_scores:
                old_shuffle_scores.append(float(np.mean(old_fold_scores)))
            if balanced_fold_scores:
                balanced_shuffle_scores.append(float(np.mean(balanced_fold_scores)))

        old_mean = float(np.mean(old_shuffle_scores)) if old_shuffle_scores else 0.0
        old_std = float(np.std(old_shuffle_scores)) if old_shuffle_scores else 0.0
        balanced_mean = float(np.mean(balanced_shuffle_scores)) if balanced_shuffle_scores else 0.0
        balanced_std = float(np.std(balanced_shuffle_scores)) if balanced_shuffle_scores else 0.0
        return {
            "mean_shuffle_acc": balanced_mean,
            "std_shuffle_acc": balanced_std,
            "per_trial": balanced_shuffle_scores,
            "old_mean_shuffle_acc": old_mean,
            "old_std_shuffle_acc": old_std,
            "old_per_trial": old_shuffle_scores,
            "balanced_mean_shuffle_acc": balanced_mean,
            "balanced_std_shuffle_acc": balanced_std,
            "balanced_per_trial": balanced_shuffle_scores,
            "leakage_suspected": balanced_mean > 0.54,
        }

    def _correlation_analysis(self) -> Dict[str, object]:
        real_features = [feature for feature in self.feature_names if not feature.startswith("__noise_")]
        corr = self.X[real_features].corr().abs()

        clusters: list[list[str]] = []
        seen: set[str] = set()
        for idx, feature_left in enumerate(real_features):
            if feature_left in seen:
                continue
            cluster = [feature_left]
            for jdx, feature_right in enumerate(real_features):
                if idx == jdx or feature_right in seen:
                    continue
                if float(corr.loc[feature_left, feature_right]) > self.cfg.max_correlation:
                    cluster.append(feature_right)
                    seen.add(feature_right)
            if len(cluster) > 1:
                clusters.append(cluster)
            seen.add(feature_left)

        return {
            "clusters": clusters,
            "n_redundant": sum(len(cluster) - 1 for cluster in clusters),
        }

    def _select_features(self, perm_results: Dict[str, object]) -> Dict[str, object]:
        importance = perm_results["importance"]
        noise_scores = [importance[feature]["mean_drop"] for feature in self.feature_names if feature.startswith("__noise_")]
        noise_median = float(np.median(noise_scores)) if noise_scores else 0.0
        noise_max = float(np.max(noise_scores)) if noise_scores else 0.0
        threshold = max(noise_median * self.cfg.min_importance_vs_noise, noise_max)
        noise_reference = max(abs(noise_median), noise_max, 1e-10)

        ranked = [
            (feature, importance[feature]["mean_drop"], importance[feature]["std_drop"])
            for feature in self.feature_names
            if not feature.startswith("__noise_")
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)

        selected: list[dict[str, object]] = []
        rejected: list[dict[str, object]] = []
        for feature_name, mean_drop, std_drop in ranked:
            record = {
                "feature": feature_name,
                "mean_importance": round(float(mean_drop), 6),
                "std": round(float(std_drop), 6),
            }
            if mean_drop > threshold and len(selected) < self.cfg.max_features_to_keep:
                record["signal_to_noise"] = round(float(mean_drop / noise_reference), 2)
                selected.append(record)
            else:
                record["reason"] = "below_noise" if mean_drop <= threshold else "max_features"
                rejected.append(record)

        return {
            "selected": selected,
            "rejected": rejected,
            "noise_median": round(noise_median, 6),
            "noise_max": round(noise_max, 6),
            "threshold": round(float(threshold), 6),
        }

    def run(self) -> Dict[str, object]:
        print("Step 1/4: Computing permutation importance (OOS)...")
        perm_results = self._permutation_importance()

        print("Step 2/4: Running label shuffle test...")
        shuffle_results = self._label_shuffle_test()

        print("Step 3/4: Analysing feature correlations...")
        corr_results = self._correlation_analysis()

        print("Step 4/4: Selecting robust features...")
        selection = self._select_features(perm_results)

        return {
            "permutation_importance": perm_results,
            "label_shuffle": shuffle_results,
            "correlation": corr_results,
            "feature_selection": selection,
        }

    def print_report(self, results: Dict[str, object]) -> None:
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE & LEAKAGE DETECTION REPORT")
        print("=" * 80)

        perm = results["permutation_importance"]
        print(f"\n{'MODEL BASELINE (OOS)':-^60}")
        for detail in perm["fold_details"]:
            print(
                f"  Fold {detail['fold']}: train={detail['train_size']}, "
                f"test={detail['test_size']}, OOS_acc={detail['baseline_acc']}%"
            )
        avg_baseline = float(np.mean(perm["baseline_scores"])) if perm["baseline_scores"] else 0.0
        print(f"  Average OOS Accuracy: {avg_baseline * 100:.1f}%")

        shuffle = results["label_shuffle"]
        print(f"\n{'LABEL SHUFFLE TEST':-^60}")
        print(f"  Legacy shuffle accuracy:   {shuffle['old_mean_shuffle_acc'] * 100:.1f}% ± {shuffle['old_std_shuffle_acc'] * 100:.1f}%")
        print(f"  Balanced shuffle accuracy: {shuffle['balanced_mean_shuffle_acc'] * 100:.1f}% ± {shuffle['balanced_std_shuffle_acc'] * 100:.1f}%")
        if shuffle["leakage_suspected"]:
            print("  WARNING: balanced shuffled labels still produce >54% accuracy. Investigate leakage.")
        else:
            print("  No leakage signal from balanced label shuffle test.")

        corr = results["correlation"]
        print(f"\n{'CORRELATION CLUSTERS (|r| > %.2f)' % self.cfg.max_correlation:-^60}")
        if corr["clusters"]:
            for idx, cluster in enumerate(corr["clusters"], start=1):
                print(f"  Cluster {idx}: {' <-> '.join(cluster)}")
            print(f"  Redundant features to review: {corr['n_redundant']}")
        else:
            print("  No highly correlated feature clusters found.")

        selection = results["feature_selection"]
        print(f"\n{'FEATURE RANKING (Permutation Importance)':-^60}")
        print(f"  Noise baseline (median): {selection['noise_median']:.6f}")
        print(f"  Noise baseline (max):    {selection['noise_max']:.6f}")
        print(f"  Selection threshold:     {selection['threshold']:.6f}")
        print(f"\n  {'Rank':<5} {'Feature':<32} {'Importance':>12} {'+/-Std':>10} {'S/N':>8}")
        print(f"  {'-' * 74}")
        for rank, feature in enumerate(selection["selected"], start=1):
            print(
                f"  {rank:<5} {feature['feature']:<32} {feature['mean_importance']:>12.6f} "
                f"{feature['std']:>10.6f} {feature['signal_to_noise']:>8.1f}x"
            )

        if selection["rejected"]:
            print("\n  REJECTED:")
            for feature in selection["rejected"][:10]:
                print(
                    f"    - {feature['feature']:<32} imp={feature['mean_importance']:.6f} "
                    f"({feature['reason']})"
                )
            if len(selection["rejected"]) > 10:
                print(f"    ... and {len(selection['rejected']) - 10} more")

        print("\n" + "=" * 80)
        if not selection["selected"]:
            print("NO FEATURES beat the noise baseline.")
        else:
            print(f"Selected features ({len(selection['selected'])}):")
            for feature in selection["selected"]:
                print(f"  - {feature['feature']}")
        print("=" * 80)

    def plot(self, results: Dict[str, object], save_path: str = "feature_importance.png") -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch
        except ImportError:
            print("matplotlib not available - skipping feature importance plot.")
            return

        selection = results["feature_selection"]
        importance = results["permutation_importance"]["importance"]

        real_features = [
            (feature, importance[feature]["mean_drop"], importance[feature]["std_drop"])
            for feature in self.feature_names
            if not feature.startswith("__noise_")
        ]
        noise_features = [
            (feature, importance[feature]["mean_drop"], importance[feature]["std_drop"])
            for feature in self.feature_names
            if feature.startswith("__noise_")
        ]
        all_features = sorted(real_features + noise_features, key=lambda item: item[1], reverse=True)

        names = [item[0] for item in all_features]
        means = [item[1] for item in all_features]
        stds = [item[2] for item in all_features]
        selected_names = {item["feature"] for item in selection["selected"]}

        colors = []
        for name in names:
            if name.startswith("__noise_"):
                colors.append("#95a5a6")
            elif name in selected_names:
                colors.append("#2ecc71")
            else:
                colors.append("#e74c3c")

        fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.35)))
        y_pos = np.arange(len(names))
        ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean Accuracy Drop (Permutation Importance)")
        ax.set_title("Feature Importance vs Noise Baseline", fontweight="bold")
        ax.axvline(x=selection["threshold"], color="blue", linestyle=":", alpha=0.7)
        ax.axvline(x=0.0, color="black", linewidth=0.5)
        ax.legend(
            handles=[
                Patch(facecolor="#2ecc71", label="Selected"),
                Patch(facecolor="#e74c3c", label="Rejected"),
                Patch(facecolor="#95a5a6", label="Noise"),
            ],
            loc="lower right",
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {save_path}")
        plt.close(fig)
