"""
STEP 8 - Survey-weight sensitivity analysis (the number for manuscript section 3.2).

Runs the paper's XGBoost, under the paper's leak-free protocol, on five configurations:

    baseline                 UCI features only
    mean                     + demographic proxy, UNWEIGHTED  (as reported)
    mean_clinical            + clinical-state proxy, UNWEIGHTED (as reported)
    mean_wtd                 + demographic proxy, SURVEY-WEIGHTED      <- new
    mean_clinical_wtd        + clinical-state proxy, SURVEY-WEIGHTED   <- new

Reports AUROC with patient-clustered bootstrap 95% CIs, PR-AUC, Brier, and DeLong tests
against the baseline (with Holm correction across the two weighted comparisons).

Nothing about the model, hyperparameters, folds or seed differs from 03_model_search.py.
The ONLY thing that changes is how the NHANES stratum means were computed.

PREREQUISITE:  python 01b_build_datasets_weighted.py   (creates the *_wtd.csv files)

Outputs:  results/weighted_sensitivity.txt , results/weighted_sensitivity.csv

Run:  python 08_weighted_sensitivity.py
"""

import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score)
from scipy import stats
import xgboost as xgb
from xgboost import XGBClassifier

BASE      = os.environ.get(
    "ENRICHMENT_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED = os.path.join(BASE, "DATA", "processed")
RESULTS   = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

TARGET, GROUP, N_SPLITS, SEED = "readmitted", "patient_nbr", 5, 42
CAT_AS_STR = ["admission_type_id", "discharge_disposition_id", "admission_source_id", "race"]
N_BOOT = 2000
rng = np.random.default_rng(SEED)

CONFIGS = {
    "baseline":          "uci_baseline.csv",
    "mean":              "uci_mean_enrichment.csv",
    "mean_clinical":     "uci_mean_clinical.csv",
    "mean_wtd":          "uci_mean_enrichment_wtd.csv",
    "mean_clinical_wtd": "uci_mean_clinical_wtd.csv",
}
WEIGHTED = ["mean_wtd", "mean_clinical_wtd"]

XGB_VER = tuple(int(x) for x in xgb.__version__.split(".")[:2])
def _gpu():
    try:
        d = xgb.DMatrix(np.zeros((16, 3)), label=np.array([0, 1] * 8))
        p = {"tree_method": "hist", "device": "cuda"} if XGB_VER >= (2, 0) else {"tree_method": "gpu_hist"}
        xgb.train(p, d, num_boost_round=1); return True
    except Exception:
        return False
USE_GPU = _gpu()

def make_xgb(pos_weight):
    """Verbatim Table 2 hyperparameters."""
    kw = dict(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
              colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0,
              scale_pos_weight=pos_weight, eval_metric="logloss", random_state=SEED)
    if USE_GPU and XGB_VER >= (2, 0): kw.update(tree_method="hist", device="cuda")
    elif USE_GPU: kw.update(tree_method="gpu_hist")
    else: kw.update(tree_method="hist")
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
    X = pd.get_dummies(X, columns=cat, dummy_na=False).astype(float)
    return X, y, g

def oof_predict(X, y, g, splits):
    Xv = X.values.astype(float); oof = np.zeros(len(y))
    for tr, te in splits:
        assert len(set(g[tr]) & set(g[te])) == 0, "PATIENT LEAK!"
        sc = StandardScaler().fit(Xv[tr])
        pos_w = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = make_xgb(pos_w); m.fit(sc.transform(Xv[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(Xv[te]))[:, 1]
    return oof

def best_f1_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.05, 0.95, 181))); bt, bf = 0.5, -1.0
    for t in grid:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, float(t)
    return bt

def cluster_boot_auc(y, p, g):
    """Patient-clustered bootstrap CI for AUROC (resamples patients, not encounters)."""
    idx_by_pat = pd.Series(np.arange(len(g))).groupby(g).apply(lambda s: s.values)
    pat_ids = np.array(idx_by_pat.index)
    lookup = {k: v for k, v in zip(idx_by_pat.index, idx_by_pat.values)}
    vals = []
    for _ in range(N_BOOT):
        take = rng.choice(pat_ids, size=len(pat_ids), replace=True)
        idx = np.concatenate([lookup[k] for k in take])
        yy, pp = y[idx], p[idx]
        if yy.min() == yy.max(): continue
        vals.append(roc_auc_score(yy, pp))
    return round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)

def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N); T2[J] = T; return T2

def delong_test(y, pa, pb):
    order = (-np.array(y)).argsort(); label = np.array(y)[order]
    preds = np.vstack((pa, pb))[:, order]; m = int(label.sum()); n = len(label) - m
    tx = np.empty([2, m]); ty = np.empty([2, n]); tz = np.empty([2, m + n])
    for r in range(2):
        tx[r] = _midrank(preds[r, :m]); ty[r] = _midrank(preds[r, m:]); tz[r] = _midrank(preds[r, :])
    aucs = (tz[:, :m].sum(1) / m - (m + 1) / 2) / n
    s = np.cov((tz[:, :m] - tx) / n) / m + np.cov(1.0 - (tz[:, m:] - ty) / m) / n
    var = np.array([[1, -1]]).dot(s).dot(np.array([[1, -1]]).T)
    z = (aucs[0] - aucs[1]) / np.sqrt(var[0, 0] + 1e-12)
    return float(aucs[0]), float(aucs[1]), float(2 * (1 - stats.norm.cdf(abs(z))))

def holm(pvals):
    order = np.argsort(pvals); adj = np.empty(len(pvals)); run = 0.0
    for rank, i in enumerate(order):
        val = (len(pvals) - rank) * pvals[i]; run = max(run, val); adj[i] = min(run, 1.0)
    return adj

# ======================================================================
if __name__ == "__main__":
    out = []
    def log(s): print(s); out.append(s)

    missing = [f for f in CONFIGS.values() if not os.path.exists(os.path.join(PROCESSED, f))]
    if missing:
        raise SystemExit(f"Missing input file(s): {missing}\n"
                         f"Run  python 01b_build_datasets_weighted.py  first.")

    log(f"XGBoost {xgb.__version__} | GPU {'YES' if USE_GPU else 'no'} | bootstrap n={N_BOOT}")
    frames = {n: load_frame(os.path.join(PROCESSED, f)) for n, f in CONFIGS.items()}
    Xb, yb, gb = frames["baseline"]
    splits = list(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(yb)), yb, gb))
    log(f"Patient-grouped {N_SPLITS}-fold CV | {len(yb)} encounters, {len(np.unique(gb))} patients\n")

    # ---- how different are the weighted proxy columns from the unweighted ones? ----
    log("PROXY COLUMN SHIFT (weighted vs unweighted, clinical-state):")
    Xu = frames["mean_clinical"][0]; Xw = frames["mean_clinical_wtd"][0]
    prox = [c for c in Xu.columns if c.endswith("_mean")]
    for c in prox:
        a, b = Xw[c].values, Xu[c].values
        r = np.corrcoef(a, b)[0, 1]
        log(f"   {c:<22s} mean|diff| = {np.abs(a-b).mean():8.4f}   corr = {r:.4f}")
    log("")

    log("RESULTS (XGBoost, identical hyperparameters throughout):")
    log(f"   {'config':<20s} {'AUROC':>7s} {'95% CI':>18s} {'PR-AUC':>8s} {'Brier':>8s}")
    oof, rows = {}, []
    for cfg in CONFIGS:
        X, y, g = frames[cfg]
        p = oof_predict(X, y, g, splits); oof[cfg] = p
        auc = roc_auc_score(y, p); lo, hi = cluster_boot_auc(y, p, g)
        pra = average_precision_score(y, p); bri = brier_score_loss(y, p)
        thr = best_f1_threshold(y, p); f1 = f1_score(y, (p >= thr).astype(int))
        log(f"   {cfg:<20s} {auc:7.4f}  ({lo:.4f}, {hi:.4f})  {pra:8.4f} {bri:8.4f}")
        rows.append(dict(config=cfg, auroc=round(auc, 4), ci_low=lo, ci_high=hi,
                         prauc=round(pra, 4), f1=round(f1, 4), brier=round(bri, 4)))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "weighted_sensitivity.csv"), index=False)

    log("\nDeLONG TESTS vs BASELINE:")
    pv = []
    for cfg in WEIGHTED:
        ae, ab, p = delong_test(yb, oof[cfg], oof["baseline"])
        pv.append(p)
        log(f"   {cfg:<20s} AUROC {ae:.4f} vs baseline {ab:.4f}   DeLong p = {p:.4f}")
    hp = holm(np.array(pv))
    log(f"   Holm-adjusted across the two weighted comparisons: "
        f"{hp[0]:.3f}, {hp[1]:.3f}")

    log("\nWEIGHTED vs UNWEIGHTED ENRICHMENT (does weighting change the conclusion?):")
    for wcfg, ucfg in [("mean_wtd", "mean"), ("mean_clinical_wtd", "mean_clinical")]:
        aw, au, p = delong_test(yb, oof[wcfg], oof[ucfg])
        log(f"   {wcfg:<20s} {aw:.4f}  vs  {ucfg:<16s} {au:.4f}   DeLong p = {p:.4f}")

    log("\nINTERPRETATION:")
    log("   If the weighted AUROC is within a few thousandths of the unweighted one and")
    log("   both remain non-significant against baseline, the negative result is robust")
    log("   to survey weighting - which is the sentence section 3.2 needs.")

    with open(os.path.join(RESULTS, "weighted_sensitivity.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print("\nWritten -> results/weighted_sensitivity.txt , weighted_sensitivity.csv")
