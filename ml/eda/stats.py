"""Statistical calculations, bootstrap intervals, and event study helpers for EDA."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from ml.eda.config import BOOTSTRAP_ITERATIONS, SEED

if TYPE_CHECKING:
    from collections.abc import Sequence


def bootstrap_ci(
    values: Sequence[float] | np.ndarray | pd.Series,
    stat: Callable[[np.ndarray], float] = np.mean,
    n: int = BOOTSTRAP_ITERATIONS,
    alpha: float = 0.05,
    seed: int = SEED,
) -> tuple[float, float]:
    """Compute percentile bootstrap confidence interval for a statistic.

    Returns:
        (lower_bound, upper_bound)
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n_samples = len(arr)
    if n_samples == 0:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)

    if stat is np.mean or stat == np.mean:  # Vectorized mean bootstrap for speed
        indices = rng.integers(0, n_samples, size=(n, n_samples))
        boot_stats = arr[indices].mean(axis=1)
    else:
        boot_stats = np.empty(n, dtype=np.float64)
        for i in range(n):
            sample = rng.choice(arr, size=n_samples, replace=True)
            boot_stats[i] = stat(sample)

    lower = float(np.percentile(boot_stats, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return (lower, upper)


def block_bootstrap_ci(
    series: Sequence[float] | np.ndarray | pd.Series,
    block_len: int,
    stat: Callable[[np.ndarray], float] = np.mean,
    n: int = BOOTSTRAP_ITERATIONS,
    alpha: float = 0.05,
    seed: int = SEED,
) -> tuple[float, float]:
    """Compute moving-block bootstrap confidence interval for autocorrelated series.

    Returns:
        (lower_bound, upper_bound)
    """
    arr = np.asarray(series, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n_samples = len(arr)
    if n_samples == 0:
        return (np.nan, np.nan)

    b = max(1, min(int(block_len), n_samples))
    if b == 1:
        return bootstrap_ci(arr, stat=stat, n=n, alpha=alpha, seed=seed)

    num_blocks = int(np.ceil(n_samples / b))
    max_start = n_samples - b + 1
    rng = np.random.default_rng(seed)

    boot_stats = np.empty(n, dtype=np.float64)
    for i in range(n):
        starts = rng.integers(0, max_start, size=num_blocks)
        resampled_parts = [arr[s : s + b] for s in starts]
        resampled = np.concatenate(resampled_parts)[:n_samples]
        boot_stats[i] = stat(resampled)

    lower = float(np.percentile(boot_stats, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return (lower, upper)


def cohens_d(
    a: Sequence[float] | np.ndarray | pd.Series,
    b: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """Compute Cohen's d effect size between two groups."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    arr_a = arr_a[~np.isnan(arr_a)]
    arr_b = arr_b[~np.isnan(arr_b)]

    n_a, n_b = len(arr_a), len(arr_b)
    if n_a < 2 or n_b < 2:
        return 0.0

    var_a = np.var(arr_a, ddof=1)
    var_b = np.var(arr_b, ddof=1)
    pooled_sd = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_sd == 0:
        return 0.0

    return float((np.mean(arr_a) - np.mean(arr_b)) / pooled_sd)


def ratio_with_ci(
    a: Sequence[float] | np.ndarray | pd.Series,
    b: Sequence[float] | np.ndarray | pd.Series,
    n: int = BOOTSTRAP_ITERATIONS,
    alpha: float = 0.05,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Compute ratio of means (mean(a) / mean(b)) with bootstrap confidence interval.

    Returns:
        (point_ratio, lower_bound, upper_bound)
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    arr_a = arr_a[~np.isnan(arr_a)]
    arr_b = arr_b[~np.isnan(arr_b)]

    mean_a = np.mean(arr_a)
    mean_b = np.mean(arr_b)
    if mean_b == 0:
        return (np.nan, np.nan, np.nan)

    point_ratio = float(mean_a / mean_b)

    rng = np.random.default_rng(seed)
    boot_ratios = np.empty(n, dtype=np.float64)

    for i in range(n):
        sa = rng.choice(arr_a, size=len(arr_a), replace=True)
        sb = rng.choice(arr_b, size=len(arr_b), replace=True)
        m_sb = np.mean(sb)
        boot_ratios[i] = (np.mean(sa) / m_sb) if m_sb != 0 else np.nan

    valid_ratios = boot_ratios[~np.isnan(boot_ratios)]
    if len(valid_ratios) == 0:
        return (point_ratio, np.nan, np.nan)

    lower = float(np.percentile(valid_ratios, 100 * (alpha / 2)))
    upper = float(np.percentile(valid_ratios, 100 * (1 - alpha / 2)))
    return (point_ratio, lower, upper)


def ecdf(
    values: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical cumulative distribution function (ECDF).

    Returns:
        (sorted_values, cumulative_probabilities)
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (np.array([]), np.array([]))

    sorted_vals = np.sort(arr)
    probs = np.arange(1, len(sorted_vals) + 1, dtype=np.float64) / len(sorted_vals)
    return (sorted_vals, probs)


def robust_z(
    series: Sequence[float] | np.ndarray | pd.Series,
    window: int = 288,
    min_periods: int | None = None,
) -> pd.Series:
    """Compute trailing median and MAD robust z-score without future leakage.

    The rolling baseline uses trailing observations strictly prior to the current
    point (shifted by 1 interval), preventing temporal data leakage.
    """
    if min_periods is None:
        min_periods = max(1, int(window * 0.5))

    s = pd.Series(series, dtype=np.float64)
    # Strictly past values
    s_past = s.shift(1)
    med = s_past.rolling(window=window, min_periods=min_periods).median()
    abs_dev = (s_past - med).abs()
    mad = abs_dev.rolling(window=window, min_periods=min_periods).median()

    norm_mad = 1.4826 * mad
    norm_mad = norm_mad.replace(0, 1e-6)

    return (s - med) / norm_mad


def event_study(
    df: pd.DataFrame,
    event_times: Sequence[Any],
    signal: str,
    pre_hours: int = 72,
    post_hours: int = 24,
    baseline_window: tuple[int, int] | int = (-336, -72),
) -> pd.DataFrame:
    """Compute hourly baseline-normalized trajectory aligned to event occurrences.

    Args:
        df: DataFrame with 'timestamp' and the target signal column.
        event_times: Sequence of event timestamps.
        signal: Name of the signal column.
        pre_hours: Number of hours before event start.
        post_hours: Number of hours after event start.
        baseline_window: (start_hour, end_hour) relative to event start for baseline normalization.

    Returns:
        DataFrame indexed by relative_hour (-pre_hours to post_hours),
        with one column per event, plus 'median', 'min', 'max'.
    """
    if signal not in df.columns:
        raise KeyError(f"Signal '{signal}' not in DataFrame columns.")

    data = df[["timestamp", signal]].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    hourly = data.set_index("timestamp")[signal].resample("1h").mean()

    if isinstance(baseline_window, (int, float)):
        b_start, b_end = -int(baseline_window), 0
    else:
        b_start, b_end = baseline_window

    rel_hours = np.arange(-pre_hours, post_hours + 1)
    event_cols: dict[str, pd.Series] = {}

    for idx, e_time in enumerate(event_times):
        t0 = pd.to_datetime(e_time, utc=True).floor("h")

        # Baseline window
        base_start_t = t0 + pd.Timedelta(hours=b_start)
        base_end_t = t0 + pd.Timedelta(hours=b_end)
        base_series = hourly.loc[(hourly.index >= base_start_t) & (hourly.index <= base_end_t)]

        base_val = float(base_series.median()) if len(base_series) > 0 else float(hourly.median())
        if np.isnan(base_val) or base_val == 0:
            base_val = 1.0

        # Extract values for relative hours
        vals = []
        for h in rel_hours:
            target_t = t0 + pd.Timedelta(hours=int(h))
            if target_t in hourly.index:
                val = hourly.loc[target_t]
            else:
                val = np.nan
            vals.append(val / base_val)

        col_name = f"event_{idx+1}"
        event_cols[col_name] = pd.Series(vals, index=rel_hours)

    result_df = pd.DataFrame(event_cols, index=rel_hours)
    result_df.index.name = "relative_hour"

    if not result_df.empty:
        result_df["median"] = result_df.median(axis=1)
        result_df["min"] = result_df.drop(columns=["median"]).min(axis=1)
        result_df["max"] = result_df.drop(columns=["median", "min"]).max(axis=1)

    return result_df


def acf_values(series: Sequence[float] | np.ndarray | pd.Series, nlags: int = 48) -> np.ndarray:
    """Compute autocorrelation values for a series up to nlags."""
    s = pd.Series(series, dtype=np.float64).dropna()
    if len(s) <= nlags:
        nlags = max(1, len(s) - 1)
    return acf(s, nlags=nlags, fft=True)


__all__ = [
    "acf_values",
    "block_bootstrap_ci",
    "bootstrap_ci",
    "cohens_d",
    "ecdf",
    "event_study",
    "ratio_with_ci",
    "robust_z",
]
