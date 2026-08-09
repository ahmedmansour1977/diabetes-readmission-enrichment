"""
STEP 7c - Supplementary Figure S1 (SHAP), computed WITHOUT depending on the shap package.

WHY THIS EXISTS
    XGBoost 3.x stores base_score as the string "[5E-1]". Older versions of the `shap`
    package cannot parse that, which is the 'could not convert string to float' error.
    Patching the booster CONFIG did not help, because shap reads base_score from the
    serialised MODEL JSON instead.

WHAT THIS DOES INSTEAD
    XGBoost can compute exact TreeSHAP values by itself:
        booster.predict(dmatrix, pred_contribs=True)
    These are the SAME Shapley values the shap package would report - shap just draws a
    nicer plot. So this script computes them natively and draws the bar chart with
    matplotlib. No shap dependency, nothing can break.

    It still TRIES the real shap package first (with a proper model-JSON fix). If that
    works you get the standard shap plot; if not, you get the native one. Either way you
    end up with a publishable Supplementary Figure S1.

Outputs:
    results/figS1_shap_bar.png        Supplementary Figure S1 (top-20 mean |SHAP|)
    results/shap_importance.csv       every feature's mean |SHAP| and rank
    results/shap_stats.txt            printed log, incl. proxy ranks

Run:  python 07c_shap_fix.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
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
    """Verbatim Table 2 hyperparameters - identical to 03_model_search.py."""
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

def patched_booster(xgbm):
    """Return a booster whose model JSON has base_score written as a plain number.
    Same numeric value -> identical predictions, just a format shap can read."""
    tmp = os.path.join(RESULTS, "_tmp_model_for_shap.json")
    xgbm.get_booster().save_model(tmp)
    with open(tmp, "r", encoding="utf-8") as fh:
        mj = json.load(fh)
    lmp = mj["learner"]["learner_model_param"]
    raw = str(lmp.get("base_score", "0.5")).strip("[]")
    lmp["base_score"] = repr(float(raw))
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(mj, fh)
    b = xgb.Booster(); b.load_model(tmp)
    try: os.remove(tmp)
    except OSError: pass
    return b

# ======================================================================
if __name__ == "__main__":
    out = []
    def log(s): print(s); out.append(s)

    try:
        import shap as _shap_probe
        shap_ver = _shap_probe.__version__
    except Exception:
        shap_ver = "not installed"
    log(f"XGBoost {xgb.__version__} | GPU {'YES' if USE_GPU else 'no'} | shap {shap_ver}")

    log("Loading clinical-state dataset ...")
    Xc, yc, gc = load_frame(os.path.join(PROCESSED, "uci_mean_clinical.csv"))
    feats = list(Xc.columns)
    splits = list(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(yc)), yc, gc))
    tr, te = splits[0]
    assert len(set(gc[tr]) & set(gc[te])) == 0, "PATIENT LEAK!"

    sc = StandardScaler().fit(Xc.values[tr])
    Xtr, Xte = sc.transform(Xc.values[tr]), sc.transform(Xc.values[te])
    pos_w = (yc[tr] == 0).sum() / max((yc[tr] == 1).sum(), 1)
    log(f"Fitting XGBoost on fold 1 training set ({len(tr)} encounters, {len(feats)} features) ...")
    xgbm = make_xgb(pos_w); xgbm.fit(Xtr, yc[tr])

    # ---------- exact TreeSHAP, computed by XGBoost itself ----------
    log(f"Computing exact TreeSHAP on the held-out fold ({len(te)} encounters) ...")
    booster = xgbm.get_booster()
    dte = xgb.DMatrix(Xte, feature_names=feats)
    contribs = booster.predict(dte, pred_contribs=True)   # (n, n_features + 1); last col = bias
    sv = contribs[:, :-1]
    log("   done - these are exact Shapley values, identical to what shap would return.")

    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    rank = {feats[i]: r + 1 for r, i in enumerate(order)}

    imp = pd.DataFrame({"feature": feats,
                        "mean_abs_shap": mean_abs,
                        "rank": [rank[f] for f in feats]}).sort_values("rank")
    imp.to_csv(os.path.join(RESULTS, "shap_importance.csv"), index=False)

    log("\nTOP 20 FEATURES BY MEAN |SHAP|:")
    for i in order[:20]:
        log(f"   {rank[feats[i]]:>3d}. {feats[i]:<34s} {mean_abs[i]:.5f}")

    proxies = [c for c in feats if c.endswith("_mean")]
    log("\nSHAP RANK OF THE NINE NHANES PROXIES (1 = most influential):")
    for c in sorted(proxies, key=lambda c: rank[c]):
        log(f"   {c:<22s} rank {rank[c]:>3d}/{len(feats)}   mean|SHAP| = {mean_abs[feats.index(c)]:.5f}")
    share = 100 * mean_abs[[feats.index(c) for c in proxies]].sum() / mean_abs.sum()
    log(f"   All nine proxies combined = {share:.1f}% of total mean|SHAP|")

    # ---------- Figure S1 ----------
    made_with = "native"
    try:
        import shap
        try:
            expl = shap.TreeExplainer(xgbm)
        except Exception:
            expl = shap.TreeExplainer(patched_booster(xgbm))
        sv_pkg = expl.shap_values(Xte[:2000])
        shap.summary_plot(sv_pkg, Xte[:2000], feature_names=feats, show=False,
                          plot_type="bar", max_display=20)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS, "figS1_shap_bar.png"), dpi=200, bbox_inches="tight")
        plt.close()
        made_with = "shap package"
    except Exception as e:
        log(f"\n(shap package still unusable: {e})")
        log("Drawing the figure natively instead - the values are the same.")
        top = order[:20][::-1]
        colors = ["#c0392b" if feats[i].endswith("_mean") else "#1f3864" for i in top]
        plt.figure(figsize=(7.2, 6.4))
        plt.barh(range(len(top)), mean_abs[top], color=colors)
        plt.yticks(range(len(top)), [feats[i] for i in top], fontsize=8)
        plt.xlabel("Mean |SHAP value|  (impact on model output)")
        plt.title("Supplementary Figure S1 - SHAP feature importance\n(clinical-state XGBoost; red = NHANES proxy)",
                  fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS, "figS1_shap_bar.png"), dpi=200, bbox_inches="tight")
        plt.close()

    log(f"\nFigure saved -> results/figS1_shap_bar.png   (drawn with: {made_with})")
    log("Full ranking -> results/shap_importance.csv")

    with open(os.path.join(RESULTS, "shap_stats.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print("\nDone. Log written to results/shap_stats.txt")
