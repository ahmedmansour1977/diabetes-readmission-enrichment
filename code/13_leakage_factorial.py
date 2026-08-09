"""
STEP 13 - Factorial decomposition of the leakage effect (Reviewer 2, Major Comment 2).

WHY
    The leakage demonstration in 06_leakage_demo.py applies BOTH leakage mechanisms at
    once - SMOTE before the split AND encounter-level (non-grouped) folds - so it cannot
    say which one does the damage. Encounter-level splitting alone is the more common
    error in the literature, so its isolated effect is of independent interest.

WHAT THIS DOES
    A 2 x 2 design over the same model, data, seed and hyperparameters:

                              no SMOTE            SMOTE before the split
        patient-grouped CV    (D) correct         (A) resampling leakage only
        encounter-level CV    (B) grouping only   (C) both = the "leaky protocol"

    Reports AUROC for each cell, the two main effects, and the interaction:

        main effect of encounter-level splitting = (B - D) and (C - A)
        main effect of pre-split SMOTE           = (A - D) and (C - B)
        interaction                              = (C - A) - (B - D)

    Cell C should reproduce the ~0.956 already reported; cell D should reproduce ~0.671.

IMPORTANT
    Cells A and C are deliberately WRONG protocols, included only to quantify the bias.
    Only cell D is a valid estimate of performance.

Outputs: results/leakage_factorial.txt , leakage_factorial.csv , fig2b_leakage_factorial.png

Run:  python 13_leakage_factorial.py
"""

import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
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
NAVY, RED, ORANGE, GREEN = "#1f3864", "#b3261e", "#e07b39", "#2e7d32"

XGB_VER = tuple(int(v) for v in xgb.__version__.split(".")[:2])
def _gpu():
    try:
        d = xgb.DMatrix(np.zeros((16, 3)), label=np.array([0, 1] * 8))
        p = {"tree_method": "hist", "device": "cuda"} if XGB_VER >= (2, 0) else {"tree_method": "gpu_hist"}
        xgb.train(p, d, num_boost_round=1); return True
    except Exception:
        return False
USE_GPU = _gpu()

def make_xgb(pos_weight):
    """Verbatim Table 2 hyperparameters. pos_weight=1.0 when SMOTE has balanced the data."""
    kw = dict(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
              colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0,
              scale_pos_weight=pos_weight, eval_metric="logloss",
              random_state=SEED, tree_method="hist")
    if USE_GPU and XGB_VER >= (2, 0): kw["device"] = "cuda"
    elif USE_GPU: kw["tree_method"] = "gpu_hist"
    return XGBClassifier(**kw)

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

# ----------------------------------------------------------------------
def run_cell(X, y, g, grouped, smote_before_split):
    """One cell of the 2x2. Returns (AUROC, accuracy).

    smote_before_split=True reproduces the canonical error: the minority class is
    oversampled on the FULL dataset, so synthetic points derived from rows that later
    land in the test fold also appear in training.
    """
    Xv = X.values.astype(float)
    if smote_before_split:
        Xr, yr = SMOTE(random_state=SEED).fit_resample(Xv, y)
        # synthetic rows have no patient identity; give each a unique one, which is
        # exactly what makes grouping ineffective once resampling has already happened
        gr = np.concatenate([g, np.arange(g.max() + 1, g.max() + 1 + (len(yr) - len(y)))])
        pos_w = 1.0
    else:
        Xr, yr, gr = Xv, y, g
        pos_w = (y == 0).sum() / max((y == 1).sum(), 1)

    if grouped:
        splits = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED).split(
            np.zeros(len(yr)), yr, gr)
    else:
        splits = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(
            np.zeros(len(yr)), yr)

    oof = np.zeros(len(yr))
    for tr, te in splits:
        sc = StandardScaler().fit(Xr[tr])
        m = make_xgb(pos_w); m.fit(sc.transform(Xr[tr]), yr[tr])
        oof[te] = m.predict_proba(sc.transform(Xr[te]))[:, 1]
    return roc_auc_score(yr, oof), accuracy_score(yr, (oof >= 0.5).astype(int))

# ======================================================================
if __name__ == "__main__":
    out = []
    def log(s): print(s); out.append(s)

    X, y, g = load_frame(os.path.join(PROCESSED, "uci_baseline.csv"))
    log(f"XGBoost {xgb.__version__} | GPU {'YES' if USE_GPU else 'no'}")
    log(f"{len(y)} encounters, {len(np.unique(g))} patients, prevalence {y.mean():.4f}\n")

    CELLS = [
        ("D", "patient-grouped CV, no SMOTE            (correct protocol)", True,  False),
        ("B", "encounter-level CV, no SMOTE            (grouping error only)", False, False),
        ("A", "patient-grouped CV, SMOTE before split  (resampling error only)", True,  True),
        ("C", "encounter-level CV, SMOTE before split  (both = 'leaky protocol')", False, True),
    ]
    res = {}
    log("Running the four cells (each is a full 5-fold XGBoost run) ...")
    for key, label, grouped, smote in CELLS:
        auc, acc = run_cell(X, y, g, grouped, smote)
        res[key] = auc
        log(f"  [{key}] {label:62s} AUROC={auc:.4f}  acc={acc:.4f}")

    D, B, A, C = res["D"], res["B"], res["A"], res["C"]
    log("\nDECOMPOSITION (change in AUROC relative to the correct protocol):")
    log(f"  encounter-level splitting alone      {B - D:+.4f}")
    log(f"  pre-split SMOTE alone                {A - D:+.4f}")
    log(f"  both together                        {C - D:+.4f}")
    log(f"  sum of the two individual effects    {(B - D) + (A - D):+.4f}")
    log(f"  interaction (super-additivity)       {(C - A) - (B - D):+.4f}")

    log("\nMAIN EFFECTS (averaged over the other factor):")
    log(f"  encounter-level splitting   {0.5 * ((B - D) + (C - A)):+.4f}")
    log(f"  pre-split SMOTE             {0.5 * ((A - D) + (C - B)):+.4f}")

    pd.DataFrame([
        dict(cell="D", grouping="patient", smote_before_split=False, auroc=round(D, 4)),
        dict(cell="B", grouping="encounter", smote_before_split=False, auroc=round(B, 4)),
        dict(cell="A", grouping="patient", smote_before_split=True, auroc=round(A, 4)),
        dict(cell="C", grouping="encounter", smote_before_split=True, auroc=round(C, 4)),
    ]).to_csv(os.path.join(RESULTS, "leakage_factorial.csv"), index=False)

    # ---- figure ----
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    labels = ["Correct\n(grouped,\nno SMOTE)", "Encounter-level\nsplitting only",
              "Pre-split\nSMOTE only", "Both\n(leaky protocol)"]
    vals = [D, B, A, C]
    cols = [NAVY, ORANGE, GREEN, RED]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar(range(4), vals, color=cols, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(D, ls=":", color=NAVY, lw=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("AUROC"); ax.set_ylim(0.4, 1.05)
    ax.set_title("Decomposition of the evaluation-leakage effect", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig2b_leakage_factorial.png"), dpi=300)
    plt.close()
    log("\nSaved -> results/fig2b_leakage_factorial.png , leakage_factorial.csv")

    with open(os.path.join(RESULTS, "leakage_factorial.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print("\nLog -> results/leakage_factorial.txt")
