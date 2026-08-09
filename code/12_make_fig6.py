"""
STEP 12 - Figure 6, rebuilt so it matches the tables exactly.

WHY THIS REPLACES 05_make_figures.py
    05 refits its own XGBoost to draw the curves. That model is fractionally different
    from the one behind Tables 4-5, so its labels read AUC 0.670 / PR-AUC 0.225 while the
    tables say 0.671 / 0.227. This script instead reads the SAME out-of-fold predictions
    the tables were computed from (results/oof_predictions.npz, written by
    09_delta_auc_clustered.py), so figure and table can never disagree.

WHY THE CALIBRATION CURVE CHANGES
    Equal-WIDTH probability bins leave the top bins nearly empty, which produced the wild
    spike in the isotonic curve (and the MCE of 0.80). This script uses equal-COUNT
    (quantile) bins, so every plotted point summarises the same number of patients. That
    is the standard choice for reliability curves on imbalanced outcomes and removes the
    artefact without hiding anything - the underlying predictions are untouched.

Outputs (results/):
    fig6a_roc.png
    fig6b_pr.png
    fig6c_calibration.png
    fig6d_calibration_curves.png     (four variants, quantile-binned)

Run:  python 12_make_fig6.py
Prerequisite: 09_delta_auc_clustered.py (creates results/oof_predictions.npz)
"""

import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score, brier_score_loss)
import xgboost as xgb
from xgboost import XGBClassifier

BASE = os.environ.get(
    "ENRICHMENT_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED = os.path.join(BASE, "DATA", "processed")
RESULTS   = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

TARGET, GROUP, N_SPLITS, SEED = "readmitted", "patient_nbr", 5, 42
CAT_AS_STR = ["admission_type_id", "discharge_disposition_id", "admission_source_id", "race"]
CACHE = os.path.join(RESULTS, "oof_predictions.npz")
NAVY = "#1f3864"

plt.rcParams.update({"font.family": "serif", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

def load_frame(path):
    df = pd.read_csv(path)
    if df[TARGET].dtype == object:
        df[TARGET] = (df[TARGET].astype(str) == "<30").astype(int)
    y = df[TARGET].astype(int).values; g = df[GROUP].values
    X = df.drop(columns=[TARGET, GROUP])
    for c in CAT_AS_STR:
        if c in X.columns: X[c] = X[c].astype(str)
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    return pd.get_dummies(X, columns=cat, dummy_na=False).astype(float), y, g

def quantile_calibration(y, p, bins=10):
    """Reliability points using equal-COUNT bins (every point = same n patients)."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    xs, ys, ns = [], [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < len(edges) - 2 else (p >= lo) & (p <= hi)
        if m.sum() == 0: continue
        xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)

def make_xgb(pos_weight):
    kw = dict(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
              colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0,
              scale_pos_weight=pos_weight, eval_metric="logloss",
              random_state=SEED, tree_method="hist")
    try:
        xgb.train({"tree_method": "hist", "device": "cuda"},
                  xgb.DMatrix(np.zeros((8, 2)), label=[0, 1] * 4), num_boost_round=1)
        kw["device"] = "cuda"
    except Exception:
        pass
    return XGBClassifier(**kw)

def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)

# ======================================================================
if __name__ == "__main__":
    X, y, g = load_frame(os.path.join(PROCESSED, "uci_baseline.csv"))

    if not os.path.exists(CACHE):
        raise SystemExit("results/oof_predictions.npz not found - run "
                         "09_delta_auc_clustered.py first (it caches the predictions).")
    oof = np.load(CACHE)["baseline"]
    auc, pra, bri = roc_auc_score(y, oof), average_precision_score(y, oof), brier_score_loss(y, oof)
    print(f"Using cached out-of-fold predictions: AUROC={auc:.4f}  "
          f"PR-AUC={pra:.4f}  Brier={bri:.4f}")
    print("These are the same numbers reported in Tables 4 and 5.\n")

    # ---- 6a ROC ----
    fpr, tpr, _ = roc_curve(y, oof)
    plt.figure(figsize=(4.6, 4.6))
    plt.plot(fpr, tpr, color=NAVY, label=f"XGBoost (AUROC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="#999")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curve"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "fig6a_roc.png"), dpi=300); plt.close()
    print("  fig6a_roc.png")

    # ---- 6b precision-recall ----
    pr, rc, _ = precision_recall_curve(y, oof)
    plt.figure(figsize=(4.6, 4.6))
    plt.plot(rc, pr, color=NAVY, label=f"PR-AUC = {pra:.3f}")
    plt.axhline(y.mean(), ls="--", color="#999", label=f"prevalence = {y.mean():.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall curve"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "fig6b_pr.png"), dpi=300); plt.close()
    print("  fig6b_pr.png")

    # ---- 6c calibration of the reported model ----
    xs, ys, ns = quantile_calibration(y, oof)
    plt.figure(figsize=(4.6, 4.6))
    plt.plot([0, 1], [0, 1], "--", color="#999", label="perfect calibration")
    plt.plot(xs, ys, "o-", color=NAVY, label="weighted XGBoost (as reported)")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
    plt.title(f"Calibration (Brier = {bri:.3f})")
    plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "fig6c_calibration.png"), dpi=300)
    plt.close(); print(f"  fig6c_calibration.png   (deciles of ~{ns.min()}-{ns.max()} patients)")

    # ---- 6d four variants, recomputed ----
    print("\nRecomputing the four calibration variants for panel 6d ...")
    Xv = X.values.astype(float)
    splits = list(StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(y)), y, g))
    inner = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    var = {k: np.zeros(len(y)) for k in ["unweighted", "isotonic", "platt"]}
    for k, (tr, te) in enumerate(splits, 1):
        sc = StandardScaler().fit(Xv[tr])
        Xtr, Xte = sc.transform(Xv[tr]), sc.transform(Xv[te])
        m0 = make_xgb(1.0); m0.fit(Xtr, y[tr])
        var["unweighted"][te] = m0.predict_proba(Xte)[:, 1]

        i_fit, i_cal = next(iter(inner.split(np.zeros(len(tr)), y[tr], g[tr])))
        fit_idx, cal_idx = tr[i_fit], tr[i_cal]
        assert len(set(g[cal_idx]) & set(g[te])) == 0, "CALIBRATION/TEST LEAK!"
        sc2 = StandardScaler().fit(Xv[fit_idx])
        pw2 = (y[fit_idx] == 0).sum() / max((y[fit_idx] == 1).sum(), 1)
        m2 = make_xgb(pw2); m2.fit(sc2.transform(Xv[fit_idx]), y[fit_idx])
        p_cal = m2.predict_proba(sc2.transform(Xv[cal_idx]))[:, 1]
        p_te = m2.predict_proba(sc2.transform(Xv[te]))[:, 1]
        var["isotonic"][te] = IsotonicRegression(out_of_bounds="clip").fit(
            p_cal, y[cal_idx]).predict(p_te)
        lr = LogisticRegression(C=1e10, max_iter=1000).fit(_logit(p_cal), y[cal_idx])
        var["platt"][te] = lr.predict_proba(_logit(p_te))[:, 1]
        print(f"   fold {k}/{len(splits)}")

    series = [("weighted (as reported)", oof, NAVY),
              ("unweighted", var["unweighted"], "#e07b39"),
              ("weighted + isotonic", var["isotonic"], "#2e7d32"),
              ("weighted + Platt", var["platt"], "#b3261e")]
    plt.figure(figsize=(5.2, 5.2))
    plt.plot([0, 1], [0, 1], "--", color="#999", label="perfect calibration")
    for label, p, col in series:
        xs, ys, _ = quantile_calibration(y, p)
        plt.plot(xs, ys, "o-", color=col, markersize=4, label=label)
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Calibration before and after recalibration")
    plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig6d_calibration_curves.png"), dpi=300); plt.close()
    print("  fig6d_calibration_curves.png")

    print("\nBrier scores:")
    for label, p, _ in series:
        print(f"   {label:<24s} {brier_score_loss(y, p):.4f}")
    print("\n05_make_figures.py is now superseded - use these panels instead.")
