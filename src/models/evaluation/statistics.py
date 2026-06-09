import numpy as np
import scikit_posthocs as sp

from scipy import stats
from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon
from itertools import combinations
from typing import Dict, List, Optional, Sequence


def _validate_scores(a: np.ndarray, b: np.ndarray, name: str = "scores") -> None:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(
            f"{name}: both arrays must have the same shape, "
            f"got {a.shape} vs {b.shape}."
        )
    if a.ndim != 1:
        raise ValueError(f"{name}: expected 1-D arrays, got shape {a.shape}.")
    if len(a) < 2:
        raise ValueError(f"{name}: need at least 2 observations.")


def _extract_metric(
    cv_results: Dict[str, np.ndarray],
    metric: str = "test_f1",
) -> np.ndarray:
    """Pull a per-fold metric array from cross_validate output dict."""
    if metric not in cv_results:
        available = [k for k in cv_results if k.startswith("test_")]
        raise KeyError(
            f"Metric '{metric}' not found in cv_results. "
            f"Available test metrics: {available}"
        )
    return np.asarray(cv_results[metric], dtype=float)


class StatisticalEvaluator:
    """
    Compare pairs (or groups) of models using their per-fold CV scores.

    Parameters
    ----------
    alpha : float
        Significance level (default 0.05).
    default_metric : str
        Key used when extracting scores from cv_results dicts
        (default ``"test_f1"``).
    """

    def __init__(self, alpha: float = 0.05, default_metric: str = "test_f1") -> None:
        self.alpha = alpha
        self.default_metric = default_metric

    def _to_array(self, scores: np.ndarray | Dict[str, np.ndarray], metric: Optional[str] = None) -> np.ndarray:
        metric = metric or self.default_metric
        if isinstance(scores, dict):
            return _extract_metric(scores, metric)
        return np.asarray(scores, dtype=float)

    def _report(
        self,
        test_name: str,
        model_a: str,
        model_b: str,
        statistic: float,
        p_value: float,
        extra: str = "",
    ) -> Dict:
        significant = p_value < self.alpha
        result = {
            "test": test_name,
            "model_a": model_a,
            "model_b": model_b,
            "statistic": round(statistic, 6),
            "p_value": round(p_value, 6),
            "significant": significant,
        }
        direction = (
            f"  {model_a} > {model_b}"
            if np.mean(self._last_a) > np.mean(self._last_b)
            else f"  {model_b} > {model_a}"
        )
        sig_str = "SIGNIFICANT " if significant else "not significant"
        print(
            f"\n[{test_name}] {model_a} vs {model_b}\n"
            f"  statistic={statistic:.4f}  p={p_value:.4f}"
            f"  {sig_str}\n"
            f"  mean scores: {model_a}={np.mean(self._last_a):.4f}"
            f"  {model_b}={np.mean(self._last_b):.4f}"
            f"{direction}"
            + (f"\n  {extra}" if extra else "")
        )
        return result

    def wilcoxon(
        self,
        scores_a: np.ndarray | Dict,
        scores_b: np.ndarray | Dict,
        model_a: str,
        model_b: str,
        metric: Optional[str] = None,
        alternative: str = "two-sided",
    ) -> Dict:
        """
        Wilcoxon signed-rank test on paired per-fold scores.
        """
        a = self._to_array(scores_a, metric)
        b = self._to_array(scores_b, metric)
        _validate_scores(a, b, "wilcoxon")
        self._last_a, self._last_b = a, b

        diff = a - b
        if np.all(diff == 0):
            return {"test": "Wilcoxon", "statistic": 0.0, "p_value": 1.0,
                    "significant": False}

        stat, p = wilcoxon(a, b, alternative=alternative)
        return self._report("Wilcoxon signed-rank", model_a, model_b, stat, p)

    def t_test(
        self,
        scores_a: np.ndarray | Dict,
        scores_b: np.ndarray | Dict,
        model_a: str,
        model_b: str,
        metric: Optional[str] = None,
        alternative: str = "two-sided",
    ) -> Dict:
        """
        Paired (dependent-samples) t-test on per-fold scores.
        """
        a = self._to_array(scores_a, metric)
        b = self._to_array(scores_b, metric)
        _validate_scores(a, b, "t_test")
        self._last_a, self._last_b = a, b

        stat, p = ttest_rel(a, b, alternative=alternative)

        # Normality check on differences (informational only)
        diff = a - b
        extra = ""
        if len(diff) >= 3:
            _, p_norm = stats.shapiro(diff)
            if p_norm < 0.05:
                extra = (
                    f"Shapiro-Wilk on differences: p={p_norm:.4f} "
                    f"(normality assumption may be violated — consider Wilcoxon)"
                )

        return self._report("Paired t-test", model_a, model_b, stat, p, extra)


    def mc_nemar(
        self,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray,
        y_true: np.ndarray,
        model_a: str,
        model_b: str,
        exact: bool = False,
    ) -> Dict:
        """
        McNemar's test comparing two classifiers on the *same* test set.
        """
        y_pred_a = np.asarray(y_pred_a)
        y_pred_b = np.asarray(y_pred_b)
        y_true   = np.asarray(y_true)

        if not (y_pred_a.shape == y_pred_b.shape == y_true.shape):
            raise ValueError("y_pred_a, y_pred_b, and y_true must have the same shape.")

        correct_a = (y_pred_a == y_true)
        correct_b = (y_pred_b == y_true)

        b01 = np.sum( correct_a & ~correct_b)   # A right, B wrong
        b10 = np.sum(~correct_a &  correct_b)   # A wrong, B right
        b00 = np.sum(~correct_a & ~correct_b)
        b11 = np.sum( correct_a &  correct_b)

        total_disagree = b01 + b10
        acc_a = correct_a.mean()
        acc_b = correct_b.mean()

        # Dummy store for _report direction
        self._last_a = np.array([acc_a])
        self._last_b = np.array([acc_b])

        if total_disagree == 0:
            print(
                f"\n[McNemar] {model_a} vs {model_b}\n"
                f"  Contingency: b01={b01} b10={b10} (no disagreements)\n"
                f"  p=1.0  → not significant"
            )
            return {"test": "McNemar", "b01": 0, "b10": 0,
                    "statistic": 0.0, "p_value": 1.0,
                    "significant": False}

        if exact or total_disagree < 25:
            # Exact binomial: H0: P(b01) = 0.5
            result_binom = stats.binomtest(b01, total_disagree, p=0.5)
            stat = float(b01)
            p    = result_binom.pvalue
            method = "exact binomial"
        else:
            # Chi-squared approximation (with continuity correction)
            stat = (abs(b01 - b10) - 1.0) ** 2 / (b01 + b10)
            p    = stats.chi2.sf(stat, df=1)
            method = "chi-squared"

        significant = p < self.alpha
        extra = (
            f"Contingency: b01={b01}  b10={b10} "
            f"  b00={b00}  b11={b11}\n"
            f"  method={method}"
            f"  acc_A={acc_a:.4f}  acc_B={acc_b:.4f}"
        )
        print(
            f"\n[McNemar] {model_a} vs {model_b}\n"
            f"  statistic={stat:.4f}  p={p:.4f}"
            f"  {'SIGNIFICANT' if significant else 'not significant'}\n"
            f"  {extra}"
        )
        return {
            "test": "McNemar",
            "model_a": model_a,
            "model_b": model_b,
            "b01": int(b01),
            "b10": int(b10),
            "b00": int(b00),
            "b11": int(b11),
            "statistic": round(stat, 6),
            "p_value": round(p, 6),
            "significant": significant,
            "method": method,
        }

    def friedman(
        self,
        cv_results_list: List[np.ndarray | Dict],
        model_names: List[str],
        metric: Optional[str] = None,
    ) -> Dict:
        """
        Friedman test: non-parametric one-way repeated-measures ANOVA.
        """
        if len(cv_results_list) < 3:
            raise ValueError("Friedman test requires at least 3 models.")

        scores = [self._to_array(s, metric) for s in cv_results_list]

        # Validate shapes
        n_folds = len(scores[0])
        for i, s in enumerate(scores):
            if len(s) != n_folds:
                raise ValueError(
                    f"All models must have the same number of folds. "
                    f"{model_names[i]} has {len(s)}, expected {n_folds}."
                )

        stat, p = friedmanchisquare(*scores)
        significant = p < self.alpha

        print(
            f"\n[Friedman] {len(scores)} models over {n_folds} folds\n"
            f"  models: {model_names}\n"
            f"  statistic={stat:.4f}  p={p:.4f}"
            f"  {'SIGNIFICANT (run post-hoc Nemenyi)' if significant else 'not significant'}"
        )
        for name, s in zip(model_names, scores):
            print(f"    {name}: mean={np.mean(s):.4f}  std={np.std(s):.4f}")

        return {
            "test": "Friedman",
            "models": model_names,
            "statistic": round(stat, 6),
            "p_value": round(p, 6),
            "significant": significant,
            "mean_scores": {n: round(float(np.mean(s)), 6) for n, s in zip(model_names, scores)},
        }

    def nemenyi(
        self,
        cv_results_list: List[np.ndarray | Dict],
        model_names: List[str],
        metric: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Nemenyi post-hoc test: pairwise comparisons after a significant
        Friedman test.
        """
        scores = [self._to_array(s, metric) for s in cv_results_list]

        print(f"\n[Nemenyi post-hoc] pairwise comparisons (o={self.alpha})")

        # scikit-posthocs expects shape (n_folds, n_models)
        data = np.column_stack(scores)
        p_matrix = sp.posthoc_nemenyi_friedman(data)
        p_matrix.index   = model_names
        p_matrix.columns = model_names

        results = {}
        for i, j in combinations(range(len(model_names)), 2):
            na, nb = model_names[i], model_names[j]
            p = float(p_matrix.iloc[i, j])
            sig = p < self.alpha
            results[f"{na}_vs_{nb}"] = {
                "p_value": round(p, 6),
                "significant": sig
            }
            print(
                f"  {'SIGNIFICANT' if sig else 'not significant'}"
            )
        return results
    

    def compare_all(
        self,
        cv_results_list: List[np.ndarray | Dict],
        model_names: List[str],
        metric: Optional[str] = None,
        tests: Sequence[str] = ("wilcoxon", "t_test"),
    ) -> Dict[str, List[Dict]]:
        """
        Run pairwise statistical tests for every pair of models.
        """
        all_results: Dict[str, List[Dict]] = {t: [] for t in tests}

        print(
            f"\n{'='*60}\n"
            f"Pairwise statistical comparison ({len(cv_results_list)} models)\n"
            f"Metric: {metric or self.default_metric} o={self.alpha}\n"
            f"{'='*60}"
        )

        for i, j in combinations(range(len(cv_results_list)), 2):
            na, nb = model_names[i], model_names[j]
            if "wilcoxon" in tests:
                res = self.wilcoxon(
                    cv_results_list[i], cv_results_list[j],
                    model_a=na, model_b=nb, metric=metric
                )
                all_results["wilcoxon"].append(res)
            if "t_test" in tests:
                res = self.t_test(
                    cv_results_list[i], cv_results_list[j],
                    model_a=na, model_b=nb, metric=metric
                )
                all_results["t_test"].append(res)

        # If ≥3 models, run Friedman + Nemenyi
        if len(cv_results_list) >= 3:
            friedman_res = self.friedman(cv_results_list, model_names, metric)
            all_results["friedman"] = [friedman_res]
            if friedman_res["significant"]:
                nemenyi_res = self.nemenyi(cv_results_list, model_names, metric)
                all_results["nemenyi"] = [nemenyi_res]

        return all_results
    
    
    # TODO: I dont know if we want to compare models between datasets only with Wilcoxon - verify pls
    def compare_datasets(
        self,
        cv_results: List[np.ndarray | Dict],
        cv_partial_results: List[np.ndarray | Dict],
        model_names: List[str],
        metric: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """
        Compare each model trained on the full dataset vs the partial dataset.
 
        For each model at index i, runs a Wilcoxon signed-rank test between
        cv_results[i] and cv_partial_results[i]. Folds must be identical
        (same random_state) for the pairing to be valid.
        """
        if len(cv_results) != len(cv_partial_results):
            raise ValueError(
                f"cv_results ({len(cv_results)}) and cv_partial_results "
                f"({len(cv_partial_results)}) must have the same length."
            )
 
        metric = metric or self.default_metric
 
        print(
            f"\n{'='*60}\n"
            f"Dataset comparison: full vs partial features\n"
            f"Metric: {metric}   o={self.alpha}\n"
            f"{'='*60}"
        )
 
        all_results: Dict[str, Dict] = {}
 
        for i, name in enumerate(model_names):
            full    = self._to_array(cv_results[i], metric)
            partial = self._to_array(cv_partial_results[i], metric)
            self._last_a, self._last_b = full, partial
 
            diff = full - partial
            mean_full    = float(np.mean(full))
            mean_partial = float(np.mean(partial))
            delta        = mean_full - mean_partial
            direction    = "full > partial" if delta > 0 else "partial > full" if delta < 0 else "tie"
 
            if np.all(diff == 0):
                result = {
                    "statistic": 0.0, "p_value": 1.0,
                    "significant": False,
                    "mean_full": mean_full, "mean_partial": mean_partial,
                    "delta": 0.0, "direction": "tie",
                }
            else:
                stat, p = ttest_rel(full, partial, alternative="two-sided")
                significant = p < self.alpha
                result = {
                    "statistic": round(stat, 6),
                    "p_value": round(p, 6),
                    "significant": significant,
                    "mean_full": round(mean_full, 6),
                    "mean_partial": round(mean_partial, 6),
                    "delta": round(delta, 6),
                    "direction": direction,
                }
 
            all_results[name] = result
 
            sig_str = "SIGNIFICANT" if result["significant"] else "not significant"
            print(
                f"\n[{name}] full vs partial\n"
                f"  full={mean_full:.4f}  partial={mean_partial:.4f}"
                f"  delta={delta:+.4f}  ({direction})\n"
                f"  statistic={result['statistic']:.4f}  "
                f"   p={result['p_value']:.4f}"
                f"  {sig_str}"
            )
 
        # Summary table
        print(f"\n{'='*60}")
        print(f"  Summary")
        print(f"{'='*60}")
        print(f"  {'Model':<25} {'Full':>8} {'Partial':>8} {'delta':>8}  {'Sig':>4}")
        print(f"  {'-'*55}")
        for name, r in all_results.items():
            print(
                f"  {name:<25} {r['mean_full']:>8.4f} {r['mean_partial']:>8.4f}"
                f" {r['delta']:>+8.4f}"
            )
        print(f"{'='*60}\n")
 
        return all_results