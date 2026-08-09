"""
STEP 3 - Model search (find the honest ceiling + settle "is the ensemble hurting us?").

Leak-free, patient-grouped 5-fold CV (same protocol as step 2). For EACH dataset config
(baseline / mean / mean_clinical) it trains several single models side by side and reports
AUC / PR-AUC / F1 / Brier with 95% CIs, plus a DeLong test of the best-enriched vs baseline.

Key differences from step 2 (this is what usually lifts AUC on this dataset):
  * SINGLE tuned models instead of averaging 5 heterogeneous classifiers (averaging a strong
    XGBoost with weak models drags it down - that is why step 2 sat at ~0.59).
  * CLASS WEIGHTING (scale_pos_weight / class_weight) instead of SMOTE. For gradient-boosted
    trees this preserves ranking (AUC) and calibration far better than synthetic oversampling.
  * XGBoost and LightGBM run on the GPU (RTX 3050) if available.

Reference ceiling on THIS dataset with correct patient-level evaluation: AUROC ~= 0.66-0.68
(state of the art ~0.683). Treat anything materially higher as a leak.

Run:  python 03_model_search.py
Needs: scikit-learn, xgboost (GPU build); lightgbm optional (auto-skipped if absent).
"""

import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             precision_score, recall_score, f1_score)
from scipy import stats
import xgboost as xgb
from xgboost import XGBClassifier
try:
    from sklearn.model_selection import StratifiedGroupKFold
    SGK = True
except Exception:
    SGK = False
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

CONFIGS = {"baseline": "uci_baseline.csv",
           "mean": "uci_mean_enrichment.csv",
           "mean_clinical": "uci_mean_clinical.csv"}
ENRICHED = ["mean", "mean_clinical"]
TARGET, GROUP, N_SPLITS, SEED = "readmitted", "patient_nbr", 5, 42
CAT_AS_STR = ["admission_type_id", "discharge_disposition_id", "admission_source_id", "race"]

# ---------------- XGBoost GPU ----------------
XGB_VER = tuple(int(x) for x in xgb.__version__.split(".")[:2])
def _detect_xgb_gpu():
    try:
        d = xgb.DMatrix(np.zeros((16, 3)), label=np.array([0, 1] * 8))
        p = {"tree_method": "hist", "device": "cuda"} if XGB_VER >= (2, 0) else {"tree_method": "gpu_hist"}
        xgb.train(p, d, num_boost_round=1); return True
    except Exception:
        return False
USE_GPU = _detect_xgb_gpu()

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

# ---------------- data ----------------
def load_frame(path):
    df = pd.read_csv(path)
    if df[TARGET].dtype == object:
        df[TARGET] = (df[TARGET].astype(str) == "<30").astype(int)
    y = df[TARGET].astype(int).values; g = df[GROUP].values
    X = df.drop(columns=[TARGET, GROUP])
    for c in CAT_AS_STR:
        if c in X.columns: X[c] = X[c].astype(str)
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    X = pd.get_dummies(X, columns=cat, dummy_na=False).astype(float)
    out = X.copy(); out[TARGET] = y; out[GROUP] = g; return out

def make_splits(y, groups):
    if SGK:
        return list(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
                    .split(np.zeros(len(y)), y, groups))
    from sklearn.model_selection import GroupKFold
    return list(GroupKFold(n_splits=N_SPLITS).split(np.zeros(len(y)), y, groups))

def best_f1_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.05, 0.95, 181))); bt, bf = 0.5, -1.0
    for t in grid:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, float(t)
    return bt

def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5*(i+j-1)+1; i = j
    T2 = np.empty(N); T2[J] = T; return T2
def delong_test(y, pa, pb):
    order = (-np.array(y)).argsort(); label = np.array(y)[order]
    preds = np.vstack((pa, pb))[:, order]; m = int(label.sum()); n = len(label)-m
    tx = np.empty([2, m]); ty = np.empty([2, n]); tz = np.empty([2, m+n])
    for r in range(2):
        tx[r] = _midrank(preds[r, :m]); ty[r] = _midrank(preds[r, m:]); tz[r] = _midrank(preds[r, :])
    aucs = (tz[:, :m].sum(1)/m - (m+1)/2)/n
    s = np.cov((tz[:, :m]-tx)/n)/m + np.cov(1.0-(tz[:, m:]-ty)/m)/n
    var = np.array([[1, -1]]).dot(s).dot(np.array([[1, -1]]).T)
    z = (aucs[0]-aucs[1])/np.sqrt(var[0, 0]+1e-12)
    return float(aucs[0]), float(aucs[1]), float(2*(1-stats.norm.cdf(abs(z))))

# ---------------- run one (config, model) ----------------
def run(df, feats, splits, model_name):
    X = df[feats].values.astype(float); y = df[TARGET].values.astype(int); g = df[GROUP].values
    oof_p = np.zeros(len(y)); per_fold = []
    for tr, te in splits:
        assert len(set(g[tr]) & set(g[te])) == 0, "PATIENT LEAK!"
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        pos_w = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        clf = build_models(pos_w)[model_name]
        clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]; oof_p[te] = p
        thr = best_f1_threshold(y[tr], clf.predict_proba(Xtr)[:, 1]); pred = (p >= thr).astype(int)
        per_fold.append(dict(auc=roc_auc_score(y[te], p), prauc=average_precision_score(y[te], p),
                             brier=brier_score_loss(y[te], p), f1=f1_score(y[te], pred),
                             rec=recall_score(y[te], pred), prec=precision_score(y[te], pred, zero_division=0)))
    return oof_p, per_fold

def ci(vals):
    v = np.array(vals); m, s = v.mean(), v.std(ddof=1); h = 1.96*s/np.sqrt(len(v))
    return m, m-h, m+h

# ======================================================================
if __name__ == "__main__":
    print(f"XGBoost {xgb.__version__} | GPU: {'YES' if USE_GPU else 'no'} | "
          f"LightGBM: {'yes' if HAVE_LGBM else 'not installed'}")
    frames = {n: load_frame(os.path.join(PROCESSED, f)) for n, f in CONFIGS.items()}
    base = frames["baseline"]
    splits = make_splits(base[TARGET].values.astype(int), base[GROUP].values)
    model_names = list(build_models(1.0).keys())
    print(f"Patient-grouped {N_SPLITS}-fold CV | models: {model_names}\n")

    rows = []; oof = {}
    for cfg, df in frames.items():
        feats = [c for c in df.columns if c not in (TARGET, GROUP)]
        for mname in model_names:
            op, pf = run(df, feats, splits, mname)
            oof[(cfg, mname)] = (df[TARGET].values.astype(int), op)
            am, alo, ahi = ci([f["auc"] for f in pf]); pm, *_ = ci([f["prauc"] for f in pf])
            fm, *_ = ci([f["f1"] for f in pf]); bm, *_ = ci([f["brier"] for f in pf])
            rows.append(dict(config=cfg, model=mname, auc=round(am, 4), auc_lo=round(alo, 4),
                             auc_hi=round(ahi, 4), prauc=round(pm, 4), f1=round(fm, 4), brier=round(bm, 4)))
            print(f"  {cfg:14s} {mname:7s}  AUC={am:.3f} [{alo:.3f},{ahi:.3f}]  PR-AUC={pm:.3f}  F1={fm:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS, "model_search.csv"), index=False)

    # best model overall (by baseline AUC), then does enrichment help with THAT model?
    best_model = res[res.config == "baseline"].sort_values("auc", ascending=False).iloc[0]["model"]
    print(f"\nBest model on baseline: {best_model.upper()}")
    yb, pb = oof[("baseline", best_model)]
    with open(os.path.join(RESULTS, "delong_modelsearch.txt"), "w") as fh:
        for cfg in ENRICHED:
            ye, pe = oof[(cfg, best_model)]
            ae, ab, p = delong_test(yb, pe, pb)
            line = f"[{best_model}] {cfg} vs baseline: AUC {ae:.3f} vs {ab:.3f} | DeLong p = {p:.4f}"
            print("  " + line); fh.write(line + "\n")
    print("\nSaved: results/model_search.csv , results/delong_modelsearch.txt")
