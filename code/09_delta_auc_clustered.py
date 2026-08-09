"""
STEP 9 - Patient-clustered comparison of AUROC differences (replaces bare DeLong).

WHY THIS IS NEEDED
    DeLong's test assumes independent observations. This dataset has 98,490 encounters
    from 69,311 patients, so encounters are CLUSTERED within patients. Ignoring that
    understates the variance and makes DeLong p-values anti-conservative (too small).
    The manuscript already uses a patient-clustered bootstrap for confidence intervals;
    this script applies the SAME clustering logic to the model COMPARISONS, so all
    inference in the paper is internally consistent.

WHAT IT DOES
    1. Computes out-of-fold predictions for every configuration (same protocol, same
       hyperparameters, same folds, seed 42) and caches them to results/oof_predictions.npz
       so you never have to recompute them again.
    2. For each enrichment configuration vs baseline, resamples PATIENTS with replacement
       and recomputes delta-AUROC = AUROC(enriched) - AUROC(baseline) on each resample
       (paired: both models scored on the identical resample).
    3. Reports delta-AUROC with a 95% CI and a two-sided bootstrap p-value, alongside the
       original DeLong p for transparency.

HOW TO READ IT
    If the delta-AUROC CI contains 0, the difference is not distinguishable from zero once
    patient clustering is respected. If it excludes 0 but the delta is ~0.001, the honest
    statement is "a statistically detectable but negligible decrement", not "no difference".
    Either way, report the delta and its CI - the effect size is the point, not the p-value.

Outputs: results/delta_auc_clustered.txt , results/delta_auc_clustered.csv

Run:  python 09_delta_auc_clustered.py
"""

import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
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
CACHE = os.path.join(RESULTS, "oof_predictions.npz")
rng = np.random.default_rng(SEED)

CONFIGS = {
    "baseline":          "uci_baseline.csv",
    "mean":              "uci_mean_enrichment.csv",
    "mean_clinical":     "uci_mean_clinical.csv",
    "mean_wtd":          "uci_mean_enrichment_wtd.csv",
    "mean_clinical_wtd": "uci_mean_clinical_wtd.csv",
}
PRIMARY   = ["mean", "mean_clinical"]              # the two comparisons in Table 5
SENSITIVITY = ["mean_wtd", "mean_clinical_wtd"]    # survey-weighted sensitivity

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

# ---------------- DeLong (kept for side-by-side reporting) ----------------
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
    return float(2 * (1 - stats.norm.cdf(abs(z))))

def holm(pvals):
    order = np.argsort(pvals); adj = np.empty(len(pvals)); run = 0.0
    for rank, i in enumerate(order):
        val = (len(pvals) - rank) * pvals[i]; run = max(run, val); adj[i] = min(run, 1.0)
    return adj

# ======================================================================
if __name__ == "__main__":
    out = []
    def log(s): print(s); out.append(s)

    log(f"XGBoost {xgb.__version__} | GPU {'YES' if USE_GPU else 'no'} | bootstrap n={N_BOOT}")

    frames = {n: load_frame(os.path.join(PROCESSED, f)) for n, f in CONFIGS.items()}
    Xb, yb, gb = frames["baseline"]
    splits = list(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(yb)), yb, gb))

    # ---- out-of-fold predictions (cached) ----
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        oof = {k: z[k] for k in z.files if k in CONFIGS}
        if set(oof) == set(CONFIGS):
            log(f"Loaded cached out-of-fold predictions from {os.path.basename(CACHE)}")
        else:
            oof = {}
    else:
        oof = {}
    if not oof:
        log("Computing out-of-fold predictions (this is the slow part) ...")
        for cfg in CONFIGS:
            X, y, g = frames[cfg]
            oof[cfg] = oof_predict(X, y, g, splits)
            log(f"   {cfg:<20s} AUROC = {roc_auc_score(y, oof[cfg]):.4f}")
        np.savez(CACHE, **oof)
        log(f"Cached -> {os.path.basename(CACHE)}")

    # ---- paired patient-clustered bootstrap ----
    log(f"\nPaired patient-clustered bootstrap ({N_BOOT} resamples of {len(np.unique(gb)):,} patients)")
    idx_by_pat = pd.Series(np.arange(len(gb))).groupby(gb).apply(lambda s: s.values)
    pat_ids = np.array(idx_by_pat.index)
    lookup = {k: v for k, v in zip(idx_by_pat.index, idx_by_pat.values)}

    resamples = []
    for _ in range(N_BOOT):
        take = rng.choice(pat_ids, size=len(pat_ids), replace=True)
        resamples.append(np.concatenate([lookup[k] for k in take]))

    pb = oof["baseline"]
    log(f"\n   {'config':<20s} {'dAUROC':>9s} {'95% CI':>20s} {'boot p':>9s} {'DeLong p':>10s}")
    rows, boot_p = [], {}
    for cfg in PRIMARY + SENSITIVITY:
        pe = oof[cfg]
        d_obs = roc_auc_score(yb, pe) - roc_auc_score(yb, pb)
        deltas = []
        for idx in resamples:
            yy = yb[idx]
            if yy.min() == yy.max(): continue
            deltas.append(roc_auc_score(yy, pe[idx]) - roc_auc_score(yy, pb[idx]))
        deltas = np.array(deltas)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        p_boot = min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))
        p_dl = delong_test(yb, pe, pb)
        boot_p[cfg] = p_boot
        log(f"   {cfg:<20s} {d_obs:+9.4f}  ({lo:+.4f}, {hi:+.4f}) {p_boot:9.4f} {p_dl:10.4f}")
        rows.append(dict(config=cfg, delta_auroc=round(d_obs, 4),
                         ci_low=round(float(lo), 4), ci_high=round(float(hi), 4),
                         bootstrap_p=round(p_boot, 4), delong_p=round(p_dl, 4)))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "delta_auc_clustered.csv"), index=False)

    for label, group in [("primary (Table 5)", PRIMARY), ("weighted sensitivity", SENSITIVITY)]:
        pv = np.array([boot_p[c] for c in group])
        hp = holm(pv)
        log(f"\n   Holm-adjusted bootstrap p, {label}: " +
            ", ".join(f"{c}={a:.3f}" for c, a in zip(group, hp)))

    log("\nREADING THIS:")
    log("   dAUROC is the effect size - report it with its CI as the primary result.")
    log("   Where the clustered bootstrap p exceeds the DeLong p, DeLong was")
    log("   anti-conservative because it ignored repeated encounters per patient.")
    log("   A dAUROC of ~0.001 is negligible regardless of any p-value: it is far")
    log("   below the precision that would matter for any clinical deployment.")

    with open(os.path.join(RESULTS, "delta_auc_clustered.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print("\nWritten -> results/delta_auc_clustered.txt , delta_auc_clustered.csv")
