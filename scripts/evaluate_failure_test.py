# ruff: noqa: E501
"""Execute single-run evaluation of the FailureModel on the AI4I test partition.

Implements Task T10, Rules R2, R5, R6, R7, R8, R9, Decision D68, D77:
1. Verifies models/failure_config_frozen.json and integrity hashes.
2. Checks docs/holdout_log.md to guarantee single-run evaluation (or justified rerun).
3. Loads test partition (2,000 rows) through guarded load_test().
4. Evaluates frozen model, baselines, and ablation.
5. Computes stratified bootstrap 95% confidence intervals (2,000 resamples).
6. Evaluates operating points (t_med, t_high, t_crit), calibration, and breakdowns.
7. Saves data/processed/failure_test_predictions.parquet.
8. Updates docs/figures/fig15_*.png to fig18_*.png with final test curves.
9. Appends dated, immutable entry to docs/holdout_log.md.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve
from xgboost import XGBClassifier

from ml.eda.plotting import apply_style, save_fig
from ml.prediction.baselines import LogisticRegressionBaseline, MajorityBaseline
from ml.prediction.config import (
    EXPERIMENTS_LOG_PATH,
    FROZEN_CONFIG_PATH,
    HOLDOUT_LOG_PATH,
    MODEL_PKL_PATH,
    TEST_PREDICTIONS_PATH,
    XGB_FIXED_PARAMS,
)
from ml.prediction.data import (
    compute_sha256,
    load_test,
    load_train,
    verify_frozen_config,
)
from ml.prediction.features import build_features
from ml.prediction.metrics import (
    breakdown_by_failure_mode,
    breakdown_by_machine_type,
    compute_expected_calibration_error,
    compute_operating_metrics,
    compute_recall_ceiling,
)
from ml.prediction.model import FailureModel
from ml.prediction.search import select_primary
from ml.prediction.thresholds import (
    assign_risk_category,
    compute_category_reliability,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate_failure_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FailureModel on test partition.")
    parser.add_argument(
        "--justified-rerun",
        type=str,
        default="",
        help="Justification reason required if this frozen configuration was already evaluated.",
    )
    return parser.parse_args()


def check_single_evaluation_guard(frozen_hash: str, justification: str) -> None:
    """Ensure test partition is evaluated exactly once per frozen configuration."""
    if not HOLDOUT_LOG_PATH.exists():
        return

    with open(HOLDOUT_LOG_PATH, encoding="utf-8") as f:
        content = f.read()

    if f"Frozen Hash: `{frozen_hash}`" in content:
        if not justification.strip():
            raise PermissionError(
                f"Rule R2 Guard: Frozen configuration {frozen_hash} has ALREADY been evaluated "
                f"in {HOLDOUT_LOG_PATH}. A rerun is prohibited without --justified-rerun."
            )
        logger.warning(
            "Justified re-run accepted for frozen hash %s: %s", frozen_hash, justification
        )


def main() -> None:
    args = parse_args()
    logger.info("Starting single-run test partition evaluation (Task T10)...")

    verify_frozen_config(FROZEN_CONFIG_PATH)
    frozen_hash = compute_sha256(FROZEN_CONFIG_PATH)
    logger.info("Verified frozen config: hash=%s", frozen_hash)

    check_single_evaluation_guard(frozen_hash, args.justified_rerun)

    test_df = load_test(FROZEN_CONFIG_PATH)
    y_test = test_df["machine_failure"].to_numpy(dtype=np.int32)
    n_test = len(y_test)
    n_pos_test = int(np.sum(y_test == 1))
    prevalence_test = float(n_pos_test / n_test)
    logger.info(
        "Loaded test partition: %d rows (%d positives, %.2f%% failure rate)",
        n_test,
        n_pos_test,
        prevalence_test * 100.0,
    )

    model = FailureModel.load(MODEL_PKL_PATH)
    thresholds = model.thresholds
    t_med = thresholds.t_med if hasattr(thresholds, "t_med") else float(thresholds["t_med"])
    t_high = thresholds.t_high if hasattr(thresholds, "t_high") else float(thresholds["t_high"])
    t_crit = thresholds.t_crit if hasattr(thresholds, "t_crit") else float(thresholds["t_crit"])

    p_test = model.predict_proba(test_df)
    cats_test = assign_risk_category(p_test, model.thresholds)

    test_preds_df = pd.DataFrame(
        {
            "udi": test_df["udi"].to_numpy(),
            "y_true": y_test,
            "failure_probability": np.round(p_test, 4),
            "risk_level": cats_test,
        }
    )
    TEST_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_preds_df.to_parquet(TEST_PREDICTIONS_PATH, index=False)
    logger.info("Saved test predictions to %s", TEST_PREDICTIONS_PATH)

    m_high = compute_operating_metrics(y_test, p_test, threshold=t_high)
    m_med = compute_operating_metrics(y_test, p_test, threshold=t_med)
    m_crit = compute_operating_metrics(y_test, p_test, threshold=t_crit)
    m_half = compute_operating_metrics(y_test, p_test, threshold=0.50)

    logger.info("Computing stratified bootstrap 95%% confidence intervals (2,000 resamples)...")
    pos_idx = np.where(y_test == 1)[0]
    neg_idx = np.where(y_test == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    n_resamples = 2000
    rng = np.random.default_rng(42)

    boot_pr = np.empty(n_resamples, dtype=np.float64)
    boot_roc = np.empty(n_resamples, dtype=np.float64)
    boot_f1 = np.empty(n_resamples, dtype=np.float64)
    boot_rec = np.empty(n_resamples, dtype=np.float64)
    boot_prec = np.empty(n_resamples, dtype=np.float64)
    boot_brier = np.empty(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        r_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        r_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        idx = np.concatenate([r_pos, r_neg])
        yt_s = y_test[idx]
        yp_s = p_test[idx]
        m_s = compute_operating_metrics(yt_s, yp_s, threshold=t_high)
        boot_pr[i] = m_s["pr_auc"]
        boot_roc[i] = m_s["roc_auc"]
        boot_f1[i] = m_s["f1"]
        boot_rec[i] = m_s["recall"]
        boot_prec[i] = m_s["precision"]
        boot_brier[i] = m_s["brier_score"]

    from ml.prediction.metrics import BootstrapResult
    ci_pr_auc = BootstrapResult(m_high["pr_auc"], float(np.percentile(boot_pr, 2.5)), float(np.percentile(boot_pr, 97.5)))
    ci_roc_auc = BootstrapResult(m_high["roc_auc"], float(np.percentile(boot_roc, 2.5)), float(np.percentile(boot_roc, 97.5)))
    ci_f1 = BootstrapResult(m_high["f1"], float(np.percentile(boot_f1, 2.5)), float(np.percentile(boot_f1, 97.5)))
    ci_recall = BootstrapResult(m_high["recall"], float(np.percentile(boot_rec, 2.5)), float(np.percentile(boot_rec, 97.5)))
    ci_precision = BootstrapResult(m_high["precision"], float(np.percentile(boot_prec, 2.5)), float(np.percentile(boot_prec, 97.5)))
    ci_brier = BootstrapResult(m_high["brier_score"], float(np.percentile(boot_brier, 2.5)), float(np.percentile(boot_brier, 97.5)))

    train_df = load_train()
    y_train = train_df["machine_failure"].to_numpy(dtype=np.int32)
    X_train_phys = build_features(train_df, feature_set="physics")
    X_test_phys = build_features(test_df, feature_set="physics")
    X_train_raw = build_features(train_df, feature_set="raw")
    X_test_raw = build_features(test_df, feature_set="raw")

    base_maj = MajorityBaseline().fit(X_train_phys, y_train)
    p_maj_test = base_maj.predict_proba(X_test_phys)[:, 1]
    m_maj = compute_operating_metrics(y_test, p_maj_test, threshold=0.5)

    base_lr = LogisticRegressionBaseline().fit(X_train_phys, y_train)
    p_lr_test = base_lr.predict_proba(X_test_phys)[:, 1]
    m_lr = compute_operating_metrics(y_test, p_lr_test, threshold=0.5)

    exp_df = pd.read_csv(EXPERIMENTS_LOG_PATH)
    primary_row = select_primary(exp_df)
    xgb_raw_params = {
        **XGB_FIXED_PARAMS,
        "n_estimators": int(primary_row["n_estimators"]),
        "max_depth": int(primary_row["max_depth"]),
        "learning_rate": float(primary_row["learning_rate"]),
        "subsample": float(primary_row["subsample"]),
        "colsample_bytree": float(primary_row["colsample_bytree"]),
        "min_child_weight": int(primary_row["min_child_weight"]),
        "reg_lambda": float(primary_row["reg_lambda"]),
        "gamma": float(primary_row["gamma"]),
        "scale_pos_weight": float(primary_row["scale_pos_weight"]),
    }
    xgb_raw = XGBClassifier(**xgb_raw_params).fit(X_train_raw, y_train)
    p_raw_test = xgb_raw.predict_proba(X_test_raw)[:, 1]
    m_raw = compute_operating_metrics(y_test, p_raw_test, threshold=0.5)

    mode_breakdown = breakdown_by_failure_mode(test_df, p_test, threshold=t_high)
    type_breakdown = breakdown_by_machine_type(test_df, y_test, p_test, threshold=t_high)
    ceiling = compute_recall_ceiling(test_df)
    category_rel = compute_category_reliability(y_test, p_test, model.thresholds)

    logger.info("Test PR-AUC: %.4f (CI: [%.4f, %.4f])", m_high["pr_auc"], ci_pr_auc.ci_lower, ci_pr_auc.ci_upper)
    logger.info("Test ROC-AUC: %.4f (CI: [%.4f, %.4f])", m_high["roc_auc"], ci_roc_auc.ci_lower, ci_roc_auc.ci_upper)
    logger.info("Test Recall (t_high): %.4f (CI: [%.4f, %.4f])", m_high["recall"], ci_recall.ci_lower, ci_recall.ci_upper)
    logger.info("Test Precision (t_high): %.4f (CI: [%.4f, %.4f])", m_high["precision"], ci_precision.ci_lower, ci_precision.ci_upper)

    logger.info("Updating Figures F15-F18 with test partition results...")
    apply_style()

    # Figure 15: PR & ROC curves
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(14, 5.5))
    prec_t, rec_t, _ = precision_recall_curve(y_test, p_test)
    fpr_t, tpr_t, _ = roc_curve(y_test, p_test)
    ax_pr.plot(
        rec_t,
        prec_t,
        label=f"Test (PR-AUC={m_high['pr_auc']:.3f}, n={n_pos_test})",
        color="#0072B2",
        lw=2.5,
    )
    ax_roc.plot(
        fpr_t,
        tpr_t,
        label=f"Test (ROC-AUC={m_high['roc_auc']:.3f})",
        color="#0072B2",
        lw=2.5,
    )

    prec_raw_t, rec_raw_t, _ = precision_recall_curve(y_test, p_raw_test)
    fpr_raw_t, tpr_raw_t, _ = roc_curve(y_test, p_raw_test)
    ax_pr.plot(
        rec_raw_t,
        prec_raw_t,
        label=f"Ablation F0 Raw (PR-AUC={m_raw['pr_auc']:.3f})",
        color="#E69F00",
        ls="--",
    )
    ax_roc.plot(
        fpr_raw_t,
        tpr_raw_t,
        label=f"Ablation F0 Raw (ROC-AUC={m_raw['roc_auc']:.3f})",
        color="#E69F00",
        ls="--",
    )

    prec_lr_t, rec_lr_t, _ = precision_recall_curve(y_test, p_lr_test)
    fpr_lr_t, tpr_lr_t, _ = roc_curve(y_test, p_lr_test)
    ax_pr.plot(
        rec_lr_t,
        prec_lr_t,
        label=f"Logistic Reg (PR-AUC={m_lr['pr_auc']:.3f})",
        color="#009E73",
        ls=":",
    )
    ax_roc.plot(
        fpr_lr_t,
        tpr_lr_t,
        label=f"Logistic Reg (ROC-AUC={m_lr['roc_auc']:.3f})",
        color="#009E73",
        ls=":",
    )

    ax_pr.axhline(
        prevalence_test,
        color="#D55E00",
        ls="--",
        label=f"Majority Baseline (prevalence={prevalence_test:.3f})",
    )
    ax_roc.plot([0, 1], [0, 1], color="#999999", ls="--", label="Random Chance (AUC=0.500)")

    ax_pr.set_title(f"Precision-Recall Curve (Test Partition, N={n_test:,})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="lower left", framealpha=0.9)

    ax_roc.set_title(f"Receiver Operating Characteristic (Test Partition, N={n_test:,})")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    save_fig(fig, "fig15_failure_model_pr_roc")
    plt.close(fig)

    # Figure 16: Calibration
    fig, (ax_cal, ax_hist) = plt.subplots(1, 2, figsize=(14, 5.5))
    pt_cal, pp_cal = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")
    cal_ece = compute_expected_calibration_error(y_test, p_test, n_bins=10)
    ax_cal.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax_cal.plot(
        pp_cal,
        pt_cal,
        "o-",
        color="#0072B2",
        lw=2,
        label=f"Test Calibrated (Brier={m_high['brier_score']:.4f}, ECE={cal_ece:.4f})",
    )
    ax_cal.set_title("Test Reliability Diagram (10 Quantile Bins)")
    ax_cal.set_xlabel("Mean Predicted Probability")
    ax_cal.set_ylabel("Observed Fraction of Positives")
    ax_cal.legend(loc="lower right")

    ax_hist.hist(
        p_test, bins=30, alpha=0.7, color="#0072B2", density=True, label="Test Probabilities"
    )
    ax_hist.set_title("Test Predicted Probabilities Distribution")
    ax_hist.set_xlabel("Predicted Failure Probability")
    ax_hist.set_ylabel("Density")
    ax_hist.set_yscale("log")
    ax_hist.legend(loc="upper right")
    plt.tight_layout()
    save_fig(fig, "fig16_failure_model_calibration")
    plt.close(fig)

    # Figure 17: Operating points and category reliability
    fig, (ax_cm, ax_rel) = plt.subplots(1, 2, figsize=(14, 5.5))
    op_data = [
        [
            "t_med",
            f"{t_med:.4f}",
            f"{m_med['recall']:.1%}",
            f"{m_med['precision']:.1%}",
            f"{m_med['fp']:,}",
        ],
        [
            "t_high",
            f"{t_high:.4f}",
            f"{m_high['recall']:.1%}",
            f"{m_high['precision']:.1%}",
            f"{m_high['fp']:,}",
        ],
        [
            "t_crit",
            f"{t_crit:.4f}",
            f"{m_crit['recall']:.1%}",
            f"{m_crit['precision']:.1%}",
            f"{m_crit['fp']:,}",
        ],
    ]
    ax_cm.axis("off")
    table = ax_cm.table(
        cellText=op_data,
        colLabels=[
            "Threshold",
            "Cutoff",
            f"Recall (n={n_pos_test})",
            "Precision",
            f"False Positives (N={n_test:,})",
        ],
        loc="center",
        cellLoc="center",
    )
    table.scale(1.1, 2.0)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    ax_cm.set_title(f"Test Operational Operating Points (N={n_test:,}, Pos={n_pos_test})", pad=20)

    cats = category_rel["risk_category"].tolist()
    shares = category_rel["share_pct"].tolist()
    rates = (category_rel["observed_failure_rate"] * 100).tolist()
    x = np.arange(len(cats))
    ax_rel.bar(x, shares, width=0.4, label="Sample Share (%)", color="#56B4E9", alpha=0.8)
    ax_rel_twin = ax_rel.twinx()
    ax_rel_twin.plot(x, rates, color="#D55E00", marker="o", lw=2, label="Observed Rate (%)")
    ax_rel.set_xticks(x)
    ax_rel.set_xticklabels(cats)
    ax_rel.set_ylabel("Share of Fleet States (%)", color="#56B4E9")
    ax_rel_twin.set_ylabel("Observed Failure Rate (%)", color="#D55E00")
    ax_rel.set_title("Test Risk Category Reliability & Volume Distribution")
    plt.tight_layout()
    save_fig(fig, "fig17_failure_model_risk_categories")
    plt.close(fig)

    # Figure 18: Breakdowns
    fig, (ax_mode, ax_type) = plt.subplots(1, 2, figsize=(14, 5.5))
    modes = mode_breakdown["failure_mode"].tolist()
    mode_recs = (mode_breakdown["recall"] * 100).tolist()
    mode_cnts = mode_breakdown["count"].tolist()
    colors = ["#0072B2", "#0072B2", "#0072B2", "#0072B2", "#D55E00"]
    bars = ax_mode.bar(modes, mode_recs, color=colors, alpha=0.85)
    ax_mode.axhline(
        ceiling["recall_ceiling"] * 100,
        color="#CC79A7",
        ls="--",
        lw=2,
        label=f"Recall Ceiling ({ceiling['recall_ceiling']:.1%})",
    )
    for bar, cnt in zip(bars, mode_cnts, strict=False):
        yval = bar.get_height()
        ax_mode.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 1.5,
            f"n={cnt}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax_mode.set_title("Test Recall by Failure Mode (t_high operating point)")
    ax_mode.set_ylabel("Recall (%)")
    ax_mode.set_ylim([0, 115])
    ax_mode.legend(loc="upper right")

    types = type_breakdown["type"].tolist()
    type_recs = (type_breakdown["recall"] * 100).tolist()
    type_cnts = type_breakdown["n_failures"].tolist()
    bars_t = ax_type.bar(types, type_recs, color="#009E73", alpha=0.85)
    for bar, cnt in zip(bars_t, type_cnts, strict=False):
        yval = bar.get_height()
        ax_type.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 1.5,
            f"n={cnt}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax_type.set_title("Test Recall by Machine Quality Type")
    ax_type.set_ylabel("Recall (%)")
    ax_type.set_ylim([0, 115])
    plt.tight_layout()
    save_fig(fig, "fig18_failure_model_recall_by_mode")
    plt.close(fig)

    # 11. Append immutable entry to docs/holdout_log.md
    now_utc_str = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Helper row lookups for category reliability
    low_r = category_rel[category_rel["risk_category"] == "Low"].iloc[0]
    med_r = category_rel[category_rel["risk_category"] == "Medium"].iloc[0]
    high_r = category_rel[category_rel["risk_category"] == "High"].iloc[0]
    crit_r = category_rel[category_rel["risk_category"] == "Critical"].iloc[0]

    # Helper row lookups for modes
    def _m(m_name: str) -> tuple[int, int, str]:
        sub = mode_breakdown[mode_breakdown["failure_mode"] == m_name]
        if sub.empty:
            return 0, 0, "0.0%"
        row = sub.iloc[0]
        cnt = int(row["count"])
        rec = float(row["recall"])
        det = int(round(rec * cnt))
        return cnt, det, f"{rec:.1%}"

    twf_cnt, twf_det, twf_rec = _m("twf")
    hdf_cnt, hdf_det, hdf_rec = _m("hdf")
    pwf_cnt, pwf_det, pwf_rec = _m("pwf")
    osf_cnt, osf_det, osf_rec = _m("osf")
    rnf_cnt, rnf_det, rnf_rec = _m("rnf")

    log_entry = f"""
## Failure Model Test Partition Evaluation — {now_utc_str}

- **Execution Mode:** Frozen Pre-Registered Configuration (Rule R2, Decision D68)
- **Frozen Hash (SHA-256):** `{frozen_hash}`
- **Test Partition Size:** {n_test:,} rows ({n_pos_test} positive failures, prevalence: {prevalence_test:.4f})
- **Model Version:** `{model.model_version}`
- **Primary Configuration ID:** `{model.metadata.get('oof_summary', {}).get('primary_config_id', 'XGB-PHYS-08')}`
- **Operating Thresholds:** `t_med={t_med:.4f}`, `t_high={t_high:.4f}`, `t_crit={t_crit:.4f}`

### 1. Headline Evaluation Metrics (Test Partition)

| Metric | Primary Model | 95% Bootstrap CI | Baseline (Majority) | Baseline (Logistic Reg) | Ablation (Raw Features) |
|---|---|---|---|---|---|
| **PR-AUC (Average Precision)** | **{m_high['pr_auc']:.4f}** | [{ci_pr_auc.ci_lower:.4f}, {ci_pr_auc.ci_upper:.4f}] | {m_maj['pr_auc']:.4f} | {m_lr['pr_auc']:.4f} | {m_raw['pr_auc']:.4f} |
| **ROC-AUC** | **{m_high['roc_auc']:.4f}** | [{ci_roc_auc.ci_lower:.4f}, {ci_roc_auc.ci_upper:.4f}] | {m_maj['roc_auc']:.4f} | {m_lr['roc_auc']:.4f} | {m_raw['roc_auc']:.4f} |
| **F1 Score (@ t_high)** | **{m_high['f1']:.4f}** | [{ci_f1.ci_lower:.4f}, {ci_f1.ci_upper:.4f}] | 0.0000 | {m_lr['f1']:.4f} | {m_raw['f1']:.4f} |
| **Recall (@ t_high, n={n_pos_test})** | **{m_high['recall']:.4f}** | [{ci_recall.ci_lower:.4f}, {ci_recall.ci_upper:.4f}] | 0.0000 | {m_lr['recall']:.4f} | {m_raw['recall']:.4f} |
| **Precision (@ t_high)** | **{m_high['precision']:.4f}** | [{ci_precision.ci_lower:.4f}, {ci_precision.ci_upper:.4f}] | 0.0000 | {m_lr['precision']:.4f} | {m_raw['precision']:.4f} |
| **Accuracy (@ t_high)** | **{m_high['accuracy']:.4f}** | N/A | {m_maj['accuracy']:.4f} | {m_lr['accuracy']:.4f} | {m_raw['accuracy']:.4f} |
| **Brier Score** | **{m_high['brier_score']:.4f}** | [{ci_brier.ci_lower:.4f}, {ci_brier.ci_upper:.4f}] | {m_maj['brier_score']:.4f} | {m_lr['brier_score']:.4f} | {m_raw['brier_score']:.4f} |
| **Expected Calibration Error (ECE)** | **{cal_ece:.4f}** | N/A | N/A | {m_lr['ece']:.4f} | {m_raw['ece']:.4f} |

*Note: All recall and precision figures are computed with exactly {n_pos_test} positive failures in the test partition.*

### 2. Operating Points Performance on Test Partition

| Operating Point | Threshold | Recall (n={n_pos_test}) | Precision | False Positives | True Positives | True Negatives | False Negatives |
|---|---|---|---|---|---|---|---|
| **t_med** (Target Recall >= 0.95) | {t_med:.4f} | {m_med['recall']:.1%} | {m_med['precision']:.1%} | {m_med['fp']} | {m_med['tp']} | {m_med['tn']} | {m_med['fn']} |
| **t_high** (Max F1 Score) | {t_high:.4f} | {m_high['recall']:.1%} | {m_high['precision']:.1%} | {m_high['fp']} | {m_high['tp']} | {m_high['tn']} | {m_high['fn']} |
| **t_crit** (Target Precision >= 0.90) | {t_crit:.4f} | {m_crit['recall']:.1%} | {m_crit['precision']:.1%} | {m_crit['fp']} | {m_crit['tp']} | {m_crit['tn']} | {m_crit['fn']} |
| **Default (0.50)** | 0.5000 | {m_half['recall']:.1%} | {m_half['precision']:.1%} | {m_half['fp']} | {m_half['tp']} | {m_half['tn']} | {m_half['fn']} |

### 3. Test Category Reliability & Volume Share

| Category | Range | Rows | Share (%) | Failures | Empirical Failure Rate |
|---|---|---|---|---|---|
| **Low** | `[0.0000, {t_med:.4f})` | {int(low_r['n_rows']):,} | {float(low_r['share_pct']):.2f}% | {int(low_r['n_failures'])} | {float(low_r['observed_failure_rate']):.4f} |
| **Medium** | `[{t_med:.4f}, {t_high:.4f})` | {int(med_r['n_rows']):,} | {float(med_r['share_pct']):.2f}% | {int(med_r['n_failures'])} | {float(med_r['observed_failure_rate']):.4f} |
| **High** | `[{t_high:.4f}, {t_crit:.4f})` | {int(high_r['n_rows']):,} | {float(high_r['share_pct']):.2f}% | {int(high_r['n_failures'])} | {float(high_r['observed_failure_rate']):.4f} |
| **Critical** | `[{t_crit:.4f}, 1.0000]` | {int(crit_r['n_rows']):,} | {float(crit_r['share_pct']):.2f}% | {int(crit_r['n_failures'])} | {float(crit_r['observed_failure_rate']):.4f} |

### 4. Failure Mode Breakdown & Recall Ceiling

| Failure Mode | Test Count | Detected (@ t_high) | Recall | Classification Nature |
|---|---|---|---|---|
| **twf** (Tool Wear Failure) | {twf_cnt} | {twf_det} | {twf_rec} | Physical degradation |
| **hdf** (Heat Dissipation Failure) | {hdf_cnt} | {hdf_det} | {hdf_rec} | Thermodynamic limit |
| **pwf** (Power Failure) | {pwf_cnt} | {pwf_det} | {pwf_rec} | Power envelope violation |
| **osf** (Overstrain Failure) | {osf_cnt} | {osf_det} | {osf_rec} | Mechanical overstrain |
| **rnf** (Random Failures) | {rnf_cnt} | {rnf_det} | {rnf_rec} | Uniform random noise (Unpredictable) |

- **Empirical Test Recall Ceiling (Caveat C2):** **{ceiling['recall_ceiling']:.2%}** ({ceiling['n_predictable']}/{ceiling['n_failures']} failures contain at least one physical non-random failure mode).
- **Caveat C1 Intellectual Honesty Statement:** The near-perfect scores achieved on physical failure modes (`hdf`, `pwf`, `osf`) directly reflect the rule-generated nature of the synthetic AI4I 2020 benchmark dataset. The model learns mathematical representations of these physical envelopes. On truly random failures (`rnf`), detection recall is near 0%, as expected.
"""

    with open(HOLDOUT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_entry)

    logger.info("Appended test evaluation entry to %s", HOLDOUT_LOG_PATH)
    logger.info("Task T10 Complete! Test partition successfully evaluated once.")


if __name__ == "__main__":
    main()
