"""Generate ml/notebooks/08_failure_model_experiments.ipynb and development figures F15-F18.

Implements Task T8:
- Runs strictly on AI4I training partition (8,000 rows) and OOF predictions.
- No access to test partition (Rule R2).
- Notebook outputs are stripped (Rule R12).
- Generates development figures: F15, F16, F17, F18.
"""

from __future__ import annotations

import logging
import sys

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from ml.eda.plotting import apply_style, save_fig
from ml.prediction.calibration import choose_calibrator
from ml.prediction.config import (
    EXPERIMENTS_LOG_PATH,
    FIGURES_DIR,
    NOTEBOOKS_DIR,
    OOF_PREDICTIONS_PATH,
)
from ml.prediction.data import load_train
from ml.prediction.metrics import (
    compute_expected_calibration_error,
    compute_operating_metrics,
    compute_recall_ceiling,
)
from ml.prediction.thresholds import (
    compute_category_reliability,
    compute_thresholds,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("build_nb08")


def generate_development_figures() -> None:
    """Generate development versions of figures F15 through F18 from training and OOF data."""
    apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_train()
    y_train = train_df["machine_failure"].to_numpy(dtype=np.int32)
    prevalence = float(np.mean(y_train))

    oof_df = pd.read_parquet(OOF_PREDICTIONS_PATH)
    rep1_oof = oof_df[oof_df["repeat"] == 1].reset_index(drop=True)
    y_oof = rep1_oof["y_true"].to_numpy()
    p_oof_raw = rep1_oof["y_prob"].to_numpy()

    exp_df = pd.read_csv(EXPERIMENTS_LOG_PATH)

    _, calibrator = choose_calibrator(
        p_oof_raw,
        y_oof,
        rep1_oof["fold"].to_numpy(),
    )
    p_oof_cal = calibrator.predict_proba(p_oof_raw)
    thresholds = compute_thresholds(y_oof, p_oof_cal)

    # Figure 15: Precision-Recall and ROC Curves
    logger.info("Generating fig15_failure_model_pr_roc.png...")
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(14, 5.5))

    prec, rec, _ = precision_recall_curve(y_oof, p_oof_cal)
    fpr, tpr, _ = roc_curve(y_oof, p_oof_cal)
    primary_pr = float(exp_df[exp_df["config_id"] == "XGB-PHYS-08"]["mean_pr_auc"].iloc[0])
    primary_roc = float(exp_df[exp_df["config_id"] == "XGB-PHYS-08"]["mean_roc_auc"].iloc[0])

    ax_pr.plot(
        rec,
        prec,
        label=f"XGB-PHYS-08 (OOF, PR-AUC={primary_pr:.3f})",
        color="#0072B2",
        lw=2,
    )
    ax_roc.plot(
        fpr,
        tpr,
        label=f"XGB-PHYS-08 (OOF, ROC-AUC={primary_roc:.3f})",
        color="#0072B2",
        lw=2,
    )

    if "ABL-RAW-01" in exp_df["config_id"].values:
        raw_row = exp_df[exp_df["config_id"] == "ABL-RAW-01"].iloc[0]
        ax_pr.plot(
            [],
            [],
            label=f"Ablation F0 Raw (PR-AUC={raw_row['mean_pr_auc']:.3f})",
            color="#E69F00",
            ls="--",
        )
        ax_roc.plot(
            [],
            [],
            label=f"Ablation F0 Raw (ROC-AUC={raw_row['mean_roc_auc']:.3f})",
            color="#E69F00",
            ls="--",
        )

    if "BASE-LOGREG-01" in exp_df["config_id"].values:
        lr_row = exp_df[exp_df["config_id"] == "BASE-LOGREG-01"].iloc[0]
        ax_pr.plot(
            [],
            [],
            label=f"Logistic Reg (PR-AUC={lr_row['mean_pr_auc']:.3f})",
            color="#009E73",
            ls=":",
        )
        ax_roc.plot(
            [],
            [],
            label=f"Logistic Reg (ROC-AUC={lr_row['mean_roc_auc']:.3f})",
            color="#009E73",
            ls=":",
        )

    ax_pr.axhline(
        prevalence,
        color="#D55E00",
        ls="--",
        label=f"Majority Baseline (prevalence={prevalence:.3f})",
    )
    ax_roc.plot([0, 1], [0, 1], color="#999999", ls="--", label="Random Chance (AUC=0.500)")

    ax_pr.set_title("Precision-Recall Curve (OOF Training)")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim([0.0, 1.02])
    ax_pr.set_ylim([0.0, 1.02])
    ax_pr.legend(loc="lower left", framealpha=0.9)

    ax_roc.set_title("Receiver Operating Characteristic (OOF Training)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_xlim([-0.02, 1.0])
    ax_roc.set_ylim([0.0, 1.02])
    ax_roc.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    save_fig(fig, "fig15_failure_model_pr_roc")
    plt.close(fig)

    # Figure 16: Reliability Diagram and Probability Histogram
    logger.info("Generating fig16_failure_model_calibration.png...")
    fig, (ax_cal, ax_hist) = plt.subplots(1, 2, figsize=(14, 5.5))

    pt_raw, pp_raw = calibration_curve(y_oof, p_oof_raw, n_bins=10, strategy="quantile")
    pt_cal, pp_cal = calibration_curve(y_oof, p_oof_cal, n_bins=10, strategy="quantile")

    raw_ece = compute_expected_calibration_error(y_oof, p_oof_raw, n_bins=10)
    cal_ece = compute_expected_calibration_error(y_oof, p_oof_cal, n_bins=10)
    raw_brier = float(np.mean((p_oof_raw - y_oof) ** 2))
    cal_brier = float(np.mean((p_oof_cal - y_oof) ** 2))

    ax_cal.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax_cal.plot(
        pp_raw,
        pt_raw,
        "s-",
        color="#E69F00",
        label=f"Uncalibrated (Brier={raw_brier:.4f}, ECE={raw_ece:.4f})",
    )
    ax_cal.plot(
        pp_cal,
        pt_cal,
        "o-",
        color="#0072B2",
        lw=2,
        label=f"Isotonic Calibrated (Brier={cal_brier:.4f}, ECE={cal_ece:.4f})",
    )

    ax_cal.set_title("Calibration Curve (Reliability Diagram)")
    ax_cal.set_xlabel("Mean Predicted Probability")
    ax_cal.set_ylabel("Observed Fraction of Positives")
    ax_cal.set_xlim([0.0, 1.0])
    ax_cal.set_ylim([0.0, 1.0])
    ax_cal.legend(loc="lower right")

    ax_hist.hist(
        p_oof_raw, bins=30, alpha=0.5, label="Raw Probabilities", color="#E69F00", density=True
    )
    ax_hist.hist(
        p_oof_cal, bins=30, alpha=0.5, label="Calibrated Probs", color="#0072B2", density=True
    )
    ax_hist.set_title("Distribution of Predicted Probabilities")
    ax_hist.set_xlabel("Predicted Failure Probability")
    ax_hist.set_ylabel("Density")
    ax_hist.set_yscale("log")
    ax_hist.legend(loc="upper right")

    plt.tight_layout()
    save_fig(fig, "fig16_failure_model_calibration")
    plt.close(fig)

    # Figure 17: Risk Categories and Operating Points
    logger.info("Generating fig17_failure_model_risk_categories.png...")
    fig, (ax_cm, ax_rel) = plt.subplots(1, 2, figsize=(14, 5.5))

    m_med = compute_operating_metrics(y_oof, p_oof_cal, threshold=thresholds.t_med)
    m_high = compute_operating_metrics(y_oof, p_oof_cal, threshold=thresholds.t_high)
    m_crit = compute_operating_metrics(y_oof, p_oof_cal, threshold=thresholds.t_crit)

    op_data = [
        [
            "t_med",
            f"{thresholds.t_med:.4f}",
            f"{m_med['recall']:.1%}",
            f"{m_med['precision']:.1%}",
            f"{m_med['fp']:,}",
        ],
        [
            "t_high",
            f"{thresholds.t_high:.4f}",
            f"{m_high['recall']:.1%}",
            f"{m_high['precision']:.1%}",
            f"{m_high['fp']:,}",
        ],
        [
            "t_crit",
            f"{thresholds.t_crit:.4f}",
            f"{m_crit['recall']:.1%}",
            f"{m_crit['precision']:.1%}",
            f"{m_crit['fp']:,}",
        ],
    ]
    ax_cm.axis("off")
    table = ax_cm.table(
        cellText=op_data,
        colLabels=["Threshold", "Cutoff", "Recall", "Precision", "False Positives"],
        loc="center",
        cellLoc="center",
    )
    table.scale(1.1, 2.0)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    ax_cm.set_title("Operational Operating Points (OOF Training)", pad=20)

    rel_df = compute_category_reliability(y_oof, p_oof_cal, thresholds)
    categories = rel_df["risk_category"].tolist()
    shares = rel_df["share_pct"].tolist()
    rates = (rel_df["observed_failure_rate"] * 100).tolist()

    x = np.arange(len(categories))
    ax_rel.bar(x, shares, width=0.4, label="Sample Share (%)", color="#56B4E9", alpha=0.8)
    ax_rel_twin = ax_rel.twinx()
    ax_rel_twin.plot(x, rates, color="#D55E00", marker="o", lw=2, label="Observed Rate (%)")

    ax_rel.set_xticks(x)
    ax_rel.set_xticklabels(categories)
    ax_rel.set_ylabel("Share of Fleet States (%)", color="#56B4E9")
    ax_rel_twin.set_ylabel("Observed Failure Rate (%)", color="#D55E00")
    ax_rel.set_title("Risk Category Reliability & Volume Distribution")

    plt.tight_layout()
    save_fig(fig, "fig17_failure_model_risk_categories")
    plt.close(fig)

    # Figure 18: Recall Breakdown by Failure Mode and Machine Type
    logger.info("Generating fig18_failure_model_recall_by_mode.png...")
    fig, (ax_mode, ax_type) = plt.subplots(1, 2, figsize=(14, 5.5))

    train_fail = train_df[train_df["machine_failure"] == 1].copy()
    fail_udis = train_fail["udi"].to_numpy()

    fail_oof = rep1_oof[rep1_oof["udi"].isin(fail_udis)].copy()
    fail_preds = (
        calibrator.predict_proba(fail_oof["y_prob"].to_numpy()) >= thresholds.t_high
    ).astype(int)
    fail_oof["pred_fail"] = fail_preds
    merged_fail = pd.merge(train_fail, fail_oof[["udi", "pred_fail"]], on="udi")

    modes = ["twf", "hdf", "pwf", "osf", "rnf"]
    mode_recalls = []
    mode_counts = []
    for m in modes:
        sub = merged_fail[merged_fail[m] == 1]
        cnt = len(sub)
        rec = float(sub["pred_fail"].mean()) if cnt > 0 else 0.0
        mode_recalls.append(rec)
        mode_counts.append(cnt)

    ceiling = compute_recall_ceiling(train_fail)

    colors = ["#0072B2", "#0072B2", "#0072B2", "#0072B2", "#D55E00"]
    bars = ax_mode.bar(modes, [r * 100 for r in mode_recalls], color=colors, alpha=0.85)
    ax_mode.axhline(
        ceiling["recall_ceiling"] * 100,
        color="#CC79A7",
        ls="--",
        lw=2,
        label=f"Recall Ceiling ({ceiling['recall_ceiling']:.1%})",
    )

    for bar, cnt in zip(bars, mode_counts, strict=False):
        yval = bar.get_height()
        ax_mode.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 1.5,
            f"n={cnt}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax_mode.set_title("OOF Recall by Failure Mode (t_high operating point)")
    ax_mode.set_ylabel("Recall (%)")
    ax_mode.set_ylim([0, 115])
    ax_mode.legend(loc="upper right")

    types = ["L", "M", "H"]
    type_recalls = []
    type_counts = []
    for t in types:
        sub = merged_fail[merged_fail["type"] == t]
        cnt = len(sub)
        rec = float(sub["pred_fail"].mean()) if cnt > 0 else 0.0
        type_recalls.append(rec)
        type_counts.append(cnt)

    bars_t = ax_type.bar(types, [r * 100 for r in type_recalls], color="#009E73", alpha=0.85)
    for bar, cnt in zip(bars_t, type_counts, strict=False):
        yval = bar.get_height()
        ax_type.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 1.5,
            f"n={cnt}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax_type.set_title("OOF Recall by Machine Quality Type")
    ax_type.set_ylabel("Recall (%)")
    ax_type.set_ylim([0, 115])

    plt.tight_layout()
    save_fig(fig, "fig18_failure_model_recall_by_mode")
    plt.close(fig)
    logger.info("All development figures generated successfully.")


def build_notebook_08() -> None:
    """Build ml/notebooks/08_failure_model_experiments.ipynb with outputs stripped."""
    nb = nbf.v4.new_notebook()
    cells = []

    c1_md = (
        "# Notebook 08: Failure Prediction Model Development & Experiments\n\n"
        "**Project:** AI Operations Intelligence Platform  \n"
        "**Phase:** Week 1 Day 6 (Failure Risk Model)  \n"
        "**Data Scope:** AI4I 2020 Dataset (Training Partition: 8,000 rows only)\n\n"
        "---\n\n"
        "## Governance Rules:\n"
        "- Rule R6 (Pre-Registration): Search grid and selection rule fixed.\n"
        "- Rule R3/R4 (No Leakage): twf, hdf, pwf, osf, rnf, udi, product_id excluded.\n"
        "- Rule R2 (Test Discipline): Test partition is never loaded.\n"
        "- Rule R12 (Clean Notebook): Outputs stripped.\n"
    )
    cells.append(nbf.v4.new_markdown_cell(c1_md))

    c2_code = (
        "# 0. Setup and Environment Verification\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from ml.prediction.config import SEED, EXPERIMENTS_LOG_PATH, OOF_PREDICTIONS_PATH\n"
        "from ml.prediction.data import load_train\n"
        "from ml.prediction.features import build_features\n"
        "from ml.prediction.calibration import choose_calibrator\n"
        "from ml.prediction.thresholds import compute_thresholds, compute_category_reliability\n"
        "from ml.prediction.controls import (\n"
        "    assert_forbidden_columns_absent,\n"
        "    run_feature_leakage_scan,\n"
        ")\n"
        "from ml.prediction.model import FailureModel\n"
        "from ml.eda.plotting import apply_style\n\n"
        "apply_style()\n"
        "print(f'Random seed set to {SEED}')\n"
    )
    cells.append(nbf.v4.new_code_cell(c2_code))

    c3_md = (
        "## 1. Training Partition Inspection\n"
        "We verify that the training partition contains 8,000 rows and inspect failure prevalence."
    )
    cells.append(nbf.v4.new_markdown_cell(c3_md))

    c4_code = (
        "# 1. Load Training Partition\n"
        "train_df = load_train()\n"
        "n_total = len(train_df)\n"
        "n_failures = int(train_df['machine_failure'].sum())\n"
        "prevalence = n_failures / n_total\n"
        "print(f'Training Rows: {n_total:,}')\n"
        "print(f'Failures: {n_failures} ({prevalence:.2%})')\n"
    )
    cells.append(nbf.v4.new_code_cell(c4_code))

    c5_md = (
        "## 2. Experiments Log & Ablations Audit\n"
        "We examine failure_experiments.csv comparing XGBoost against ablations and baselines."
    )
    cells.append(nbf.v4.new_markdown_cell(c5_md))

    c6_code = (
        "# 2. Experiments Log Summary\n"
        "exp_df = pd.read_csv(EXPERIMENTS_LOG_PATH)\n"
        "print(f'Total experiments recorded: {len(exp_df)}')\n"
        "summary = exp_df[exp_df['config_id'].isin([\n"
        "    'XGB-PHYS-08', 'ABL-RAW-01', 'BASE-LOGREG-01', 'BASE-MAJORITY-01'\n"
        "])]\n"
        "cols = ['config_id', 'model_type', 'feature_set', "
        "'mean_pr_auc', 'mean_roc_auc', 'mean_brier']\n"
        "print(summary[cols].to_string(index=False))\n"
    )
    cells.append(nbf.v4.new_code_cell(c6_code))

    c7_md = (
        "## 3. Calibration & Risk Thresholds\n"
        "We audit the chosen calibrator (isotonic) and derived operating risk thresholds."
    )
    cells.append(nbf.v4.new_markdown_cell(c7_md))

    c8_code = (
        "# 3. Calibration & Risk Categories\n"
        "oof_df = pd.read_parquet(OOF_PREDICTIONS_PATH)\n"
        "rep1 = oof_df[oof_df['repeat'] == 1].reset_index(drop=True)\n"
        "_, cal = choose_calibrator(\n"
        "    rep1['y_prob'].to_numpy(), rep1['y_true'].to_numpy(), rep1['fold'].to_numpy()\n"
        ")\n"
        "cal_p = cal.predict_proba(rep1['y_prob'].to_numpy())\n"
        "thresh = compute_thresholds(rep1['y_true'].to_numpy(), cal_p)\n"
        "print(f'Thresholds: {thresh.to_dict()}')\n"
        "rel = compute_category_reliability(rep1['y_true'].to_numpy(), cal_p, thresh)\n"
        "print(rel.to_string(index=False))\n"
    )
    cells.append(nbf.v4.new_code_cell(c8_code))

    c9_md = (
        "## 4. Negative Controls (Rule R8)\n"
        "Audit feature leakage scan and confirm zero forbidden columns exist in feature matrix."
    )
    cells.append(nbf.v4.new_markdown_cell(c9_md))

    c10_code = (
        "# 4. Negative Controls\n"
        "X_train = build_features(train_df, feature_set='physics')\n"
        "assert_forbidden_columns_absent(X_train)\n"
        "print('Forbidden columns check: PASSED (zero leakage features).')\n"
        "scan = run_feature_leakage_scan(X_train, train_df['machine_failure'].to_numpy())\n"
        "print(scan.to_string(index=False))\n"
    )
    cells.append(nbf.v4.new_code_cell(c10_code))

    nb.cells = cells
    nb_path = NOTEBOOKS_DIR / "08_failure_model_experiments.ipynb"
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    logger.info("Saved notebook to %s", nb_path)


def main() -> None:
    logger.info("Executing Task T8: Generating figures F15-F18 and building Notebook 08...")
    generate_development_figures()
    build_notebook_08()
    logger.info("Task T8 successfully completed!")


if __name__ == "__main__":
    main()
