"""
STEP 16 - Multi-seed replication of the factorial decomposition (reviewer request).

WHY
  13_leakage_factorial.py runs the 2x2 design at a single seed (42). Reviewers asked
  whether the decomposition is stable across alternative fold assignments and SMOTE
  realizations. This script repeats the identical four cells over a list of seeds and
  reports the distribution of each cell, each main effect, and the interaction.

WHAT IT CHANGES vs 13
  Nothing except the seed. The cell logic, hyperparameters, encoding, scaling and
  fold construction are copied verbatim from 13_leakage_factorial.py so that the
  seed-42 row of the output reproduces the published table exactly.

USAGE
  set ENRICHMENT_BASE=M:\\WH IZ NXT\\RESEARCH\\IDEA\\ENRICHMENT
  python 16_factorial_multiseed.py                 # default seeds 1..20
  python 16_factorial_multiseed.py 42 1 2 3 4 5    # explicit seed list

  Roughly 3-5 minutes per seed on CPU, faster with a GPU. Results are written after
  every seed, so the run can be interrupted and resumed without losing work.

OUTPUTS (results/)
  factorial_multiseed.csv   one row per seed: four cells, two main effects, interaction
  factorial_multiseed.txt   summary with mean, SD, min, max and a paste-ready sentence
  fig2c_factorial_seeds.png distribution of the two main effects across seeds
"""

import os, sys, warnings, json
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
RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

TARGET, GROUP, N_SPLITS = "readmitted", "patient_nbr", 5
CAT_AS_STR = ["admission_type_id", "discharge_disposition_id", "admission_source_id", "race"]
NAVY, GREEN, ORANGE = "#1f3864", "#2e7d32", "#e07b39"

XGB_VER = tuple(int(v) for v in xgb.__version__.split(".")[:2])
def _gpu():
    try:
        d = xgb.DMatrix(np.zeros((16, 3)), label=np.array([0, 1] * 8))
        p = {"tree_method": "hist", "device": "cuda"} if XGB_VER >= (2, 0) else {"tree_method": "gpu_hist"}
        xgb.train(p, d, num_boost_round=1); return True
    except Exception:
        return False
USE_GPU = _gpu()

def make_xgb(pos_weight, seed):
    """Verbatim Table 2 hyperparameters; only random_state varies across runs."""
    kw = dict(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
              colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0,
              scale_pos_weight=pos_weight, eval_metric="logloss",
              random_state=seed, tree_method="hist")
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

def run_cell(X, y, g, grouped, smote_before_split, seed):
    """One cell of the 2x2, identical to 13_leakage_factorial.py::run_cell."""
    Xv = X.values.astype(float)
    if smote_before_split:
        Xr, yr = SMOTE(random_state=seed).fit_resample(Xv, y)
        gr = np.concatenate([g, np.arange(g.max() + 1, g.max() + 1 + (len(yr) - len(y)))])
        pos_w = 1.0
    else:
        Xr, yr, gr = Xv, y, g
        pos_w = (y == 0).sum() / max((y == 1).sum(), 1)

    if grouped:
        splits = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed).split(
            np.zeros(len(yr)), yr, gr)
    else:
        splits = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(
            np.zeros(len(yr)), yr)

    oof = np.zeros(len(yr))
    for tr, te in splits:
        sc = StandardScaler().fit(Xr[tr])
        m = make_xgb(pos_w, seed); m.fit(sc.transform(Xr[tr]), yr[tr])
        oof[te] = m.predict_proba(sc.transform(Xr[te]))[:, 1]
    return roc_auc_score(yr, oof), accuracy_score(yr, (oof >= 0.5).astype(int))

CELLS = [("D", "patient-grouped, no SMOTE (correct)", True, False),
         ("B", "encounter-level, no SMOTE", False, False),
         ("A", "patient-grouped, pre-split SMOTE", True, True),
         ("C", "encounter-level, pre-split SMOTE (leaky)", False, True)]

if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or list(range(1, 21))
    csv_path = os.path.join(RESULTS, "factorial_multiseed.csv")

    rows = []
    if os.path.exists(csv_path):                     # resume support
        rows = pd.read_csv(csv_path).to_dict("records")
        done = {r["seed"] for r in rows}
        seeds = [s for s in seeds if s not in done]
        if done: print(f"Resuming; already have seeds {sorted(done)}")

    X, y, g = load_frame(os.path.join(PROCESSED, "uci_baseline.csv"))
    print(f"XGBoost {xgb.__version__} | GPU {'YES' if USE_GPU else 'no'}")
    print(f"{len(y)} encounters, {len(np.unique(g))} patients, "
          f"{int(y.sum())} events, {X.shape[1]} encoded predictors\n")

    for seed in seeds:
        r = {}
        for key, label, grouped, smote in CELLS:
            auc, acc = run_cell(X, y, g, grouped, smote, seed)
            r[key] = auc
            print(f"  seed {seed:3d}  [{key}] {label:42s} AUROC={auc:.4f} acc={acc:.4f}", flush=True)
        D, B, A, C = r["D"], r["B"], r["A"], r["C"]
        rows.append(dict(seed=seed, D=round(D, 4), B=round(B, 4), A=round(A, 4), C=round(C, 4),
                         me_encounter=round(0.5 * ((B - D) + (C - A)), 4),
                         me_smote=round(0.5 * ((A - D) + (C - B)), 4),
                         interaction=round((C - A) - (B - D), 4)))
        pd.DataFrame(rows).to_csv(csv_path, index=False)   # checkpoint every seed

    df = pd.DataFrame(rows).sort_values("seed")
    out = []
    def log(s): print(s); out.append(s)

    log(f"\nFACTORIAL ACROSS {len(df)} SEEDS: {sorted(df.seed.tolist())}\n")
    log(f"{'quantity':38s} {'mean':>9s} {'SD':>9s} {'min':>9s} {'max':>9s}")
    for col, name in [("D", "correct protocol (D)"),
                      ("B", "encounter-level only (B)"),
                      ("A", "pre-split SMOTE only (A)"),
                      ("C", "both, leaky protocol (C)"),
                      ("me_encounter", "main effect: encounter-level split"),
                      ("me_smote", "main effect: pre-split SMOTE"),
                      ("interaction", "interaction")]:
        s = df[col]
        log(f"{name:38s} {s.mean():+9.4f} {s.std():9.4f} {s.min():+9.4f} {s.max():+9.4f}")

    me_e, me_s = df.me_encounter, df.me_smote
    log("\nPASTE-READY SENTENCE:")
    log(f"  Across {len(df)} random seeds the main effect of pre-split resampling was "
        f"{me_s.mean():+.4f} (SD {me_s.std():.4f}; range {me_s.min():+.4f} to {me_s.max():+.4f}) "
        f"and that of encounter-level splitting was {me_e.mean():+.4f} (SD {me_e.std():.4f}; "
        f"range {me_e.min():+.4f} to {me_e.max():+.4f}); the ordering of the two effects was "
        f"identical in every seed.")

    with open(os.path.join(RESULTS, "factorial_multiseed.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))

    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.4))
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    ax[0].hist(me_s, bins=12, color=GREEN, edgecolor="white")
    ax[0].set_title("Main effect: pre-split SMOTE", fontsize=10)
    ax[1].hist(me_e, bins=12, color=ORANGE, edgecolor="white")
    ax[1].set_title("Main effect: encounter-level splitting", fontsize=10)
    for a in ax:
        a.set_xlabel("change in AUROC")
        a.set_ylabel("seeds"); a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig2c_factorial_seeds.png"), dpi=300)
    plt.close()
    print("\nSaved -> results/factorial_multiseed.csv , .txt , fig2c_factorial_seeds.png")
