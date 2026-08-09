"""
STEP 7 - Reviewer-response statistics (fills the highlighted placeholders in the manuscript).

Produces every remaining number the reviewers asked for, using the SAME leak-free protocol
and the SAME fixed hyperparameters as 03_model_search.py. Nothing here changes the model or
the headline results; it only adds rigorous uncertainty/robustness reporting.

Outputs (printed and written to results/reviewer_stats.txt):
  [Tables 4 & 5] Patient-CLUSTERED bootstrap 95% CIs for AUROC, PR-AUC, F1, Brier
                 (resamples PATIENTS, not encounters -> valid CIs; replaces the fold-based CIs).
  [Table 5]      DeLong p (baseline vs each enrichment) + Holm-adjusted p across the two.
  [4.5 / Fig 6]  Calibration of baseline XGBoost: Brier, ECE, MCE, Spiegelhalter z-test p.
  [3.2 / 4.4]    Strata fall-back count/% (encounters where the clinical-state proxy collapsed
                 to the demographic mean), read straight from the processed CSVs.
  [Table 3/Fig2] Leaky-protocol AUROC with a bootstrap 95% CI, and the correct-protocol AUROC.
  [Fig 5 / S1]   Permutation importance + SHAP (TreeExplainer) for the paper XGBoost.

Run:  python 07_reviewer_stats.py
Needs: scikit-learn, xgboost, scipy, matplotlib; shap optional (SHAP plot skipped if absent).
"""

import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss, f1_score)
from scipy import stats
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False

# ----- paths / constants (match your pipeline) -----
BASE      = os.environ.get(
    "ENRICHMENT_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED = os.path.join(BASE, "DATA", "processed")
RESULTS   = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)
CONFIGS = {"baseline": "uci_baseline.csv",
           "mean": "uci_mean_enrichment.csv",
           "mean_clinical": "uci_mean_clinical.csv"}
TARGET, GROUP, N_SPLITS, SEED = "readmitted", "patient_nbr", 5, 42
CAT_AS_STR = ["admission_type_id", "discharge_disposition_id", "admission_source_id", "race"]
N_BOOT = 2000
rng = np.random.default_rng(SEED)

# ---------------- XGBoost GPU (auto) ----------------
XGB_VER = tuple(int(x) for x in xgb.__version__.split(".")[:2])
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
    return X, y, g

def make_splits(y, g):
    return list(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
                .split(np.zeros(len(y)), y, g))

def oof_predict(X, y, g, splits, model_name):
    Xv = X.values.astype(float); oof = np.zeros(len(y))
    for tr, te in splits:
        assert len(set(g[tr]) & set(g[te])) == 0, "PATIENT LEAK!"
        sc = StandardScaler().fit(Xv[tr]); Xtr, Xte = sc.transform(Xv[tr]), sc.transform(Xv[te])
        pos_w = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        clf = build_models(pos_w)[model_name]; clf.fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(Xte)[:, 1]
    return oof

def best_f1_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.05, 0.95, 181))); bt, bf = 0.5, -1.0
    for t in grid:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, float(t)
    return bt

# ---------------- patient-clustered bootstrap ----------------
def cluster_bootstrap_cis(y, p, g, thr):
    """Resample unique PATIENTS with replacement; recompute metrics. Returns dict metric->(lo,hi)."""
    pat = pd.Series(range(len(g))).groupby(g).apply(lambda s: s.values)  # patient -> row indices
    pat_ids = np.array(pat.index)
    idx_by_pat = {k: v for k, v in zip(pat.index, pat.values)}
    auc, pra, f1v, bri = [], [], [], []
    for _ in range(N_BOOT):
        take = rng.choice(pat_ids, size=len(pat_ids), replace=True)
        idx = np.concatenate([idx_by_pat[k] for k in take])
        yy, pp = y[idx], p[idx]
        if yy.min() == yy.max():   # degenerate resample
            continue
        auc.append(roc_auc_score(yy, pp))
        pra.append(average_precision_score(yy, pp))
        f1v.append(f1_score(yy, (pp >= thr).astype(int), zero_division=0))
        bri.append(brier_score_loss(yy, pp))
    q = lambda a: (round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4))
    return {"AUROC": q(auc), "PR-AUC": q(pra), "F1": q(f1v), "Brier": q(bri)}

def boot_auc_ci(y, p):
    vals = []
    n = len(y)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], p[idx]
        if yy.min() == yy.max(): continue
        vals.append(roc_auc_score(yy, pp))
    return round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)

# ---------------- calibration ----------------
def calib_errors(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); n = len(y); ece = 0.0; mce = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0: continue
        conf, acc, w = p[m].mean(), y[m].mean(), m.sum() / n
        gap = abs(acc - conf); ece += w * gap; mce = max(mce, gap)
    return ece, mce

def spiegelhalter_z(y, p):
    num = np.sum((y - p) * (1 - 2 * p))
    den = np.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
    z = num / (den + 1e-12)
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))

# ---------------- DeLong ----------------
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

# ---------------- strata fall-back (from processed CSVs only) ----------------
def fallback_stats():
    demo = pd.read_csv(os.path.join(PROCESSED, "uci_mean_enrichment.csv"))
    clin = pd.read_csv(os.path.join(PROCESSED, "uci_mean_clinical.csv"))
    proxcols = [c for c in clin.columns if c.endswith("_mean")]
    same = np.ones(len(clin), dtype=bool)
    for c in proxcols:
        same &= np.isclose(clin[c].values, demo[c].values, rtol=0, atol=1e-9)
    return int(same.sum()), len(clin), proxcols

# ======================================================================
if __name__ == "__main__":
    out = []
    def log(s): print(s); out.append(s)
    log(f"XGBoost {xgb.__version__} | GPU {'YES' if USE_GPU else 'no'} | LightGBM {'yes' if HAVE_LGBM else 'no'} | boot={N_BOOT}")

    frames = {n: load_frame(os.path.join(PROCESSED, f)) for n, f in CONFIGS.items()}
    Xb, yb, gb = frames["baseline"]
    splits = make_splits(yb, gb)
    model_names = list(build_models(1.0).keys())

    # ---- Table 4: all models on baseline, cluster-bootstrap CIs ----
    log("\n[Table 4] Baseline models - patient-clustered bootstrap 95% CIs")
    oof_store = {}
    for mn in model_names:
        p = oof_predict(Xb, yb, gb, splits, mn); oof_store[("baseline", mn)] = p
        thr = best_f1_threshold(yb, p)
        cis = cluster_bootstrap_cis(yb, p, gb, thr)
        log(f"  {mn:7s} AUROC={roc_auc_score(yb,p):.3f} {cis['AUROC']}  "
            f"PR-AUC={average_precision_score(yb,p):.3f} {cis['PR-AUC']}  "
            f"F1(thr={thr:.2f}) {cis['F1']}  Brier {cis['Brier']}")

    # ---- Table 5: XGB enrichment configs + DeLong + Holm ----
    log("\n[Table 5] XGBoost enrichment - CIs, DeLong, Holm")
    pv = []
    for cfg in ["mean", "mean_clinical"]:
        Xc, yc, gc = frames[cfg]
        p = oof_predict(Xc, yc, gc, splits, "xgb"); oof_store[(cfg, "xgb")] = p
        thr = best_f1_threshold(yc, p)
        cis = cluster_bootstrap_cis(yc, p, gc, thr)
        _, _, dp = delong_test(yb, p, oof_store[("baseline", "xgb")])
        pv.append(dp)
        log(f"  {cfg:14s} AUROC={roc_auc_score(yc,p):.3f} {cis['AUROC']}  PR-AUC {cis['PR-AUC']}  "
            f"F1 {cis['F1']}  DeLong p={dp:.4f}")
    hp = holm(np.array(pv))
    log(f"  DeLong p (mean, clinical) = {pv[0]:.4f}, {pv[1]:.4f} | Holm-adjusted = {hp[0]:.3f}, {hp[1]:.3f}")

    # ---- 4.5 / Fig 6: calibration of baseline XGB ----
    pxgb = oof_store[("baseline", "xgb")]
    ece, mce = calib_errors(yb, pxgb, bins=10); z, zp = spiegelhalter_z(yb, pxgb)
    log("\n[4.5 / Figure 6] Baseline XGBoost calibration")
    log(f"  Brier={brier_score_loss(yb,pxgb):.4f}  ECE={ece:.4f}  MCE={mce:.4f}  "
        f"Spiegelhalter z={z:.3f}, p={zp:.4f}")

    # ---- 3.2 / 4.4: strata fall-back ----
    nfb, ntot, proxcols = fallback_stats()
    log("\n[3.2 / 4.4] Clinical-state -> demographic fall-back (n<30 strata)")
    log(f"  {nfb} of {ntot} encounters ({100*nfb/ntot:.1f}%) received the demographic-mean fall-back")

    # ---- Table 3 / Fig 2: leakage demo with CI ----
    log("\n[Table 3 / Figure 2] Leakage demonstration (baseline XGBoost)")
    Xv = Xb.values.astype(float)
    Xa, ya = SMOTE(random_state=SEED).fit_resample(Xv, yb)
    leaky_oof = np.zeros(len(ya)); accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xa, ya):
        sc = StandardScaler().fit(Xa[tr]); m = build_models(1.0)["xgb"]; m.fit(sc.transform(Xa[tr]), ya[tr])
        pp = m.predict_proba(sc.transform(Xa[te]))[:, 1]; leaky_oof[te] = pp
        accs.append(((pp >= 0.5).astype(int) == ya[te]).mean())
    leaky_auc = roc_auc_score(ya, leaky_oof); leaky_ci = boot_auc_ci(ya, leaky_oof)
    log(f"  LEAKY   AUROC={leaky_auc:.3f} 95% CI {leaky_ci}  accuracy={np.mean(accs):.3f}")
    log(f"  CORRECT AUROC={roc_auc_score(yb,pxgb):.3f} (baseline XGB, patient-grouped)")

    # ---- Fig 5 / S1: permutation + SHAP for the paper XGBoost (clinical-state) ----
    log("\n[Figure 5 / S1] Feature importance (clinical-state XGBoost)")
    Xc, yc, gc = frames["mean_clinical"]; feats = list(Xc.columns)
    tr, te = splits[0]
    sc = StandardScaler().fit(Xc.values[tr]); Xtr, Xte = sc.transform(Xc.values[tr]), sc.transform(Xc.values[te])
    pos_w = (yc[tr] == 0).sum() / max((yc[tr] == 1).sum(), 1)
    xgbm = build_models(pos_w)["xgb"]; xgbm.fit(Xtr, yc[tr])
    perm = permutation_importance(xgbm, Xte, yc[te], scoring="roc_auc", n_repeats=10, random_state=SEED, n_jobs=-1)
    pi = pd.Series(perm.importances_mean, index=feats).sort_values(ascending=False)
    proxy_rank = [(c, int(np.where(pi.index == c)[0][0]) + 1) for c in feats if c.endswith("_mean")]
    log("  Permutation-importance rank of proxies (1 = most important):")
    for c, r in sorted(proxy_rank, key=lambda t: t[1]):
        log(f"     {c:22s} rank {r}/{len(feats)}")
    try:
        import shap
        expl = shap.TreeExplainer(xgbm)
        sv = expl.shap_values(Xte[:2000])
        shap.summary_plot(sv, Xte[:2000], feature_names=feats, show=False, plot_type="bar", max_display=20)
        plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "figS1_shap_bar.png"), dpi=200, bbox_inches="tight"); plt.close()
        log("  SHAP bar plot saved -> results/figS1_shap_bar.png")
    except Exception as e:
        log(f"  SHAP skipped ({e}); use the permutation ranking above for Supplementary Figure S1.")

    with open(os.path.join(RESULTS, "reviewer_stats.txt"), "w") as fh:
        fh.write("\n".join(out))
    print("\nAll reviewer statistics written to results/reviewer_stats.txt")
