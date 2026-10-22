"""Unit tests for EDA helper modules (ml/eda/)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ml.eda.config import EXPLORATION_FRACTION, SEED
from ml.eda.plotting import apply_style, save_fig
from ml.eda.splits import (
    exploration_cutoff,
    get_sensor_dates,
    is_exploration,
    is_holdout,
    make_ai4i_split,
)
from ml.eda.stats import (
    block_bootstrap_ci,
    bootstrap_ci,
    ecdf,
    event_study,
    robust_z,
)


def test_split_masks():
    """Exploration and holdout masks must be disjoint, chronological, and cover all timestamps."""
    dates = get_sensor_dates()
    assert len(dates) > 0, "No dates found in sensor dataset."

    # Create dummy timestamps across the entire span
    timestamps = pd.date_range(dates[0], dates[-1] + pd.Timedelta(hours=23), freq="1h", tz="UTC")

    exp_mask = is_exploration(timestamps)
    hold_mask = is_holdout(timestamps)

    # Disjoint
    assert not (exp_mask & hold_mask).any(), "Exploration and holdout masks must be disjoint."

    # Together cover all timestamps
    assert (exp_mask | hold_mask).all(), "Exploration and holdout must partition the full timeline."

    # Chronological
    exp_times = timestamps[exp_mask]
    hold_times = timestamps[hold_mask]
    assert len(exp_times) > 0 and len(hold_times) > 0
    assert exp_times.max() < hold_times.min(), (
        f"Max exploration timestamp ({exp_times.max()}) must be earlier than "
        f"min holdout timestamp ({hold_times.min()})."
    )


def test_exploration_cutoff():
    """Cutoff must fall at 00:00 UTC and fraction of days in exploration within one day of 0.7."""
    cutoff = exploration_cutoff()
    dates = get_sensor_dates()
    n_days = len(dates)

    assert cutoff.hour == 0 and cutoff.minute == 0 and cutoff.second == 0, (
        f"Cutoff must be 00:00 UTC, got {cutoff}"
    )
    assert str(cutoff.tz) == "UTC", f"Cutoff timezone must be UTC, got {cutoff.tz}"

    # Calculate actual fraction of days prior to cutoff
    days_in_exp = sum(1 for d in dates if d < cutoff)
    actual_fraction = days_in_exp / n_days

    # Within 1 day tolerance of EXPLORATION_FRACTION (0.7)
    expected_days = n_days * EXPLORATION_FRACTION
    assert abs(days_in_exp - expected_days) <= 1.0, (
        f"Expected ~{expected_days} days in exploration, got {days_in_exp} out of {n_days} "
        f"({actual_fraction:.3f})"
    )


def test_ai4i_split():
    """AI4I split must be stratified, disjoint, deterministic, and 80/20 size."""
    split1 = make_ai4i_split()
    split2 = make_ai4i_split()

    # Determinism
    pd.testing.assert_frame_equal(split1, split2)

    # Disjoint & complete
    train_udis = set(split1.loc[split1["partition"] == "train", "udi"])
    test_udis = set(split1.loc[split1["partition"] == "test", "udi"])
    assert len(train_udis.intersection(test_udis)) == 0, "Partitions must be disjoint."
    assert len(train_udis) + len(test_udis) == len(split1), "Partitions must sum to total."

    # 80/20 sizes
    n_total = len(split1)
    train_share = len(train_udis) / n_total
    assert abs(train_share - 0.8) < 0.01, f"Expected 80% train, got {train_share:.3%}"

    # Stratification check on machine_failure rate
    from ml.eda.load import load_ai4i

    full_df = load_ai4i(partition=None)
    merged = full_df.merge(split1, on="udi")

    train_rate = merged.loc[merged["partition"] == "train", "machine_failure"].mean()
    test_rate = merged.loc[merged["partition"] == "test", "machine_failure"].mean()

    # Failure share difference must be within 0.5 percentage points (0.005)
    diff = abs(train_rate - test_rate)
    assert diff <= 0.005, (
        f"Stratification mismatch: train failure rate {train_rate:.4f} vs test {test_rate:.4f} "
        f"(diff: {diff:.4f} > 0.005)"
    )


def test_bootstrap_ci():
    """Bootstrap CI is deterministic with fixed seed and contains true mean for normal sample."""
    rng = np.random.default_rng(SEED)
    true_mean = 50.0
    sample = rng.normal(loc=true_mean, scale=5.0, size=500)

    ci1 = bootstrap_ci(sample, stat=np.mean, n=1000, seed=123)
    ci2 = bootstrap_ci(sample, stat=np.mean, n=1000, seed=123)

    assert ci1 == ci2, "Bootstrap CI must be deterministic with fixed seed."
    assert ci1[0] <= true_mean <= ci1[1], (
        f"CI [{ci1[0]:.2f}, {ci1[1]:.2f}] does not contain true mean {true_mean}"
    )


def test_block_bootstrap_autocorrelated():
    """Block bootstrap yields a wider CI than iid bootstrap for autocorrelated series."""
    # Synthesize AR(1) process with high autocorrelation
    n = 600
    phi = 0.90
    rng = np.random.default_rng(SEED)
    noise = rng.normal(0, 1, size=n)
    ar_series = np.zeros(n)
    for t in range(1, n):
        ar_series[t] = phi * ar_series[t - 1] + noise[t]

    iid_lower, iid_upper = bootstrap_ci(ar_series, stat=np.mean, n=1500, seed=42)
    block_lower, block_upper = block_bootstrap_ci(
        ar_series, block_len=20, stat=np.mean, n=1500, seed=42
    )

    iid_width = iid_upper - iid_lower
    block_width = block_upper - block_lower

    assert block_width > iid_width, (
        f"Block bootstrap width ({block_width:.3f}) must be wider than iid bootstrap width "
        f"({iid_width:.3f}) for strongly autocorrelated data."
    )


def test_ecdf():
    """ECDF must be monotonic and end at 1.0."""
    vals = [5.2, 1.1, 3.4, 9.8, 2.5, 4.0]
    x, y = ecdf(vals)

    # Monotonic
    assert np.all(np.diff(x) >= 0), "ECDF x values must be sorted."
    assert np.all(np.diff(y) >= 0), "ECDF probabilities must be non-decreasing."

    # Ends at 1.0
    assert pytest.approx(y[-1], 1e-6) == 1.0


def test_robust_z_spike_and_no_leakage():
    """Robust z must detect spike and have no leakage from future observations."""
    n = 200
    base = np.full(n, 10.0)
    # Introduce spike at index 100
    base[100] = 50.0

    z_scores = robust_z(base, window=50, min_periods=10)

    # Spike at index 100 should have a large z-score (> 10)
    assert z_scores.iloc[100] > 10.0, f"Expected large robust z at spike, got {z_scores.iloc[100]}"

    # Future values change test (no leakage from future)
    base_modified = base.copy()
    base_modified[150:] = 1000.0  # Massive future shift after index 100

    z_scores_mod = robust_z(base_modified, window=50, min_periods=10)

    # Prior to the modification at index 150, z_scores must be identical
    np.testing.assert_allclose(
        z_scores.iloc[:150].fillna(0).to_numpy(),
        z_scores_mod.iloc[:150].fillna(0).to_numpy(),
        err_msg="Robust z at earlier time steps must not be affected by future observations.",
    )


def test_event_study():
    """Event study on crafted step series shows step at hour 0 and ~1.0 before event."""
    # Create 20 days hourly series
    start = pd.Timestamp("2026-05-01 00:00:00", tz="UTC")
    ts = pd.date_range(start, periods=24 * 20, freq="1h")

    # Flat baseline = 10.0, step to 25.0 starting at event
    event_t = pd.Timestamp("2026-05-10 12:00:00", tz="UTC")
    values = np.where(ts >= event_t, 25.0, 10.0)

    df = pd.DataFrame({"timestamp": ts, "signal": values})

    study = event_study(
        df,
        event_times=[event_t],
        signal="signal",
        pre_hours=24,
        post_hours=12,
        baseline_window=(-72, -24),
    )

    # Relative hour index
    assert -24 in study.index and 12 in study.index

    # Values before event (h < 0) should be ~1.0
    pre_vals = study.loc[study.index < 0, "median"]
    assert np.allclose(pre_vals, 1.0), f"Pre-event normalized values must be ~1.0, got {pre_vals}"

    # Value at step (h >= 0) should be 25.0 / 10.0 = 2.5
    post_vals = study.loc[study.index >= 0, "median"]
    assert np.allclose(post_vals, 2.5), f"Post-event values must be 2.5, got {post_vals}"


def test_save_fig(tmp_path, monkeypatch):
    """save_fig writes the PNG, returns the path, and file is non-empty."""
    monkeypatch.setattr("ml.eda.plotting.FIGURES", tmp_path)
    apply_style()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([1, 2, 3], [4, 5, 6])
    ax.set_title("Test Plot")

    out_path = save_fig(fig, "test_plot.png")
    plt.close(fig)

    assert out_path.exists(), f"Figure file not found at {out_path}"
    assert out_path.stat().st_size > 0, "Saved figure file is empty."
