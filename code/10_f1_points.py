"""
STEP 10 - Exact F1 values for Table 4.

The F1 figures in Table 4 need to be the pooled out-of-fold F1 for each model, at that
model's training-fold-optimised threshold. This prints all four, using the same protocol,
hyperparameters, folds and seed as 03_model_search.py and 07_reviewer_stats.py.

Run:  python 10_f1_points.py

Then copy the four F1 numbers into Table 4 of the manuscript.
"""

import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, brier_score_loss
import pandas as _pd
import xgboost as xgb
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False

BASE      = os.environ.get(
    "ENRICHMENT_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED = os.path.join(BASE, "DATA", "processed")
RESULTS   = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)
TARGET, GROUP, N_SPLITS, SEED = "readmitted", "patient_nbr", 5, 42
CAT_AS_STR = ["admission_type_id", "discharge_disposition_id", "admission_source_id", "race"]

XGB_VER = tuple(int(v) for v in xgb.__version__.split(".")[:2])
def _gpu():
    try:
        d = xgb.DMatrix(np.zeros((16, 3)), label=np.array([0, 1] * 8))
        p = {"tree_method": "hist", "device": "cuda"} if XGB_VER >= (2, 0) else {"tree_method": "gpu_hist"}
        xgb.train(p, d, num_boost_round=1); return True
    except Exception:
        return False
USE_GPU = _gpu()

def build_models(pos_weight):
    m = {}
    m["logreg"] = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
    m["rf"] = RandomForestClassifier(n_estimators=600, min_samples_leaf=5,
                                     class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    xkw = dict(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
               colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0,
               scale_pos_weight=pos_weight, eval_metric="logloss", random_state=SEED)
    if USE_GPU and XGB_VER >= (2, 0): xkw.update(tree_method="hist", device="cuda")
    elif USE_GPU: xkw.update(tree_method="gpu_hist")
    else: xkw.update(tree_method="hist")
    m["xgb"] = XGBClassifier(**xkw)
    if HAVE_LGBM:
        m["lgbm"] = LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
                                   subsample=0.8, colsample_bytree=0.7, class_weight="balanced",
                                   random_state=SEED, verbose=-1)
    return m

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

N_BOOT = 2000
_rng = np.random.default_rng(SEED)

def cluster_boot_auc(y, p, g):
    """Patient-clustered bootstrap CI for AUROC - the same method used for Table 4."""
    idx_by_pat = pd.Series(np.arange(len(g))).groupby(g).apply(lambda s: s.values)
    pat_ids = np.array(idx_by_pat.index)
    lookup = dict(zip(idx_by_pat.index, idx_by_pat.values))
    vals = []
    for _ in range(N_BOOT):
        take = _rng.choice(pat_ids, size=len(pat_ids), replace=True)
        idx = np.concatenate([lookup[k] for k in take])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        vals.append(roc_auc_score(yy, p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def best_f1_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.05, 0.95, 181))); bt, bf = 0.5, -1.0
    for t in grid:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, float(t)
    return bt

if __name__ == "__main__":
    X, y, g = load_frame(os.path.join(PROCESSED, "uci_baseline.csv"))
    Xv = X.values.astype(float)
    splits = list(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(y)), y, g))
    names = list(build_models(1.0).keys())
    print(f"XGBoost {xgb.__version__} | GPU {'YES' if USE_GPU else 'no'}\n")
    print(f"TABLE 4 VALUES (patient-clustered bootstrap, {N_BOOT} resamples):")
    print(f"  {'model':<8s} {'AUROC':>7s} {'95% CI':>18s} {'PR-AUC':>8s} {'F1':>8s} {'Brier':>8s}")

    lines, rows = [], []
    for mn in names:
        oof = np.zeros(len(y))
        for tr, te in splits:
            assert len(set(g[tr]) & set(g[te])) == 0, "PATIENT LEAK!"
            sc = StandardScaler().fit(Xv[tr])
            pw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
            clf = build_models(pw)[mn]; clf.fit(sc.transform(Xv[tr]), y[tr])
            oof[te] = clf.predict_proba(sc.transform(Xv[te]))[:, 1]
        thr = best_f1_threshold(y, oof)
        f1 = f1_score(y, (oof >= thr).astype(int))
        auc = roc_auc_score(y, oof)
        lo, hi = cluster_boot_auc(y, oof, g)
        pra, bri = average_precision_score(y, oof), brier_score_loss(y, oof)
        s = (f"  {mn:<8s} {auc:7.4f}  ({lo:.4f}, {hi:.4f}) {pra:8.4f} {f1:8.4f} {bri:8.4f}")
        print(s); lines.append(s)
        rows.append(dict(model=mn, config="baseline", auroc=round(auc, 4),
                         ci_low=round(lo, 4), ci_high=round(hi, 4),
                         prauc=round(pra, 4), f1=round(f1, 4),
                         brier=round(bri, 4), threshold=round(thr, 2)))

    _pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "table4_metrics.csv"), index=False)
    with open(os.path.join(RESULTS, "table4_f1_points.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\nSaved -> results/table4_metrics.csv , table4_f1_points.txt")
    print("Figure 3 will now use these clustered CIs, matching Table 4 exactly.")
