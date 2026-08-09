"""
STEP 2 (v2) - Leakage-free training and evaluation, GPU-accelerated.

Same rigorous protocol as v1 (patient-grouped CV, all preprocessing inside the fold,
CIs / DeLong / PR-AUC / calibration), with the two requested upgrades:

  OPTION 2  - stronger, honest features:
      * ONE-HOT encoding of categoricals (A1Cresult, max_glu_serum, admission/discharge
        ids, medications, change, diabetesMed, race) instead of arbitrary label-encoding.
      * A better-tuned gradient booster inside the ensemble.
  GPU (RTX 3050):
      * XGBoost runs on the GPU automatically (device="cuda" / gpu_hist) if a CUDA build
        is present; otherwise it falls back to fast CPU hist.
      * The 3 Keras MLP feature-extractors use the GPU automatically IF TensorFlow sees it
        (see the GPU note at the bottom of this file). They are tiny, so CPU is fine too.

  CONFIGS compared:  baseline  vs  mean (demographic proxy)  vs  mean_clinical (OPTION 1).

Run:  python 02_train_evaluate_v2.py
Needs: scikit-learn>=1.3, imbalanced-learn, xgboost>=2.0 (GPU build), tensorflow (optional), shap, scipy, matplotlib
"""

import os, random, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, precision_recall_curve, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import xgboost as xgb
from xgboost import XGBClassifier
from scipy import stats

try:
    from sklearn.model_selection import StratifiedGroupKFold
    SGK = True
except Exception:
    from sklearn.model_selection import GroupKFold
    SGK = False

# ----------------------------------------------------------------------
BASE      = os.environ.get(
    "ENRICHMENT_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED = os.path.join(BASE, "DATA", "processed")
RESULTS   = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

CONFIGS = {
    "baseline":      "uci_baseline.csv",
    "mean":          "uci_mean_enrichment.csv",
    "mean_clinical": "uci_mean_clinical.csv",
}
ENRICHED = ["mean", "mean_clinical"]                 # compared against baseline in DeLong
TARGET, GROUP, N_SPLITS, SEED = "readmitted", "patient_nbr", 5, 42
# numeric-coded NOMINAL columns that must be one-hot (not treated as ordered numbers)
CAT_AS_STR = ["admission_type_id", "discharge_disposition_id", "admission_source_id", "race"]

def reset_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    try:
        import tensorflow as tf; tf.random.set_seed(seed)
    except Exception:
        pass
reset_seeds()

# ----------------------------------------------------------------------
# XGBoost GPU (auto-detect; RTX 3050)
# ----------------------------------------------------------------------
XGB_VER = tuple(int(x) for x in xgb.__version__.split(".")[:2])
def _detect_xgb_gpu():
    try:
        d = xgb.DMatrix(np.zeros((16, 3)), label=np.array([0, 1] * 8))
        p = {"tree_method": "hist", "device": "cuda"} if XGB_VER >= (2, 0) else {"tree_method": "gpu_hist"}
        xgb.train(p, d, num_boost_round=1)
        return True
    except Exception:
        return False
USE_GPU = _detect_xgb_gpu()

def make_xgb():
    kw = dict(learning_rate=0.05, max_depth=5, n_estimators=600, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
              eval_metric="logloss", random_state=SEED)
    if USE_GPU and XGB_VER >= (2, 0):
        kw.update(tree_method="hist", device="cuda")
    elif USE_GPU:
        kw.update(tree_method="gpu_hist")
    else:
        kw.update(tree_method="hist")
    return XGBClassifier(**kw)

# ----------------------------------------------------------------------
# Deep MLP feature extractors (trained inside each fold). TF-GPU if available.
# ----------------------------------------------------------------------
MLP_ARCHS = [
    {"hidden": [128, 64], "feat": 32, "dropout": 0.3, "bn": True},
    {"hidden": [256, 128], "feat": 64, "dropout": 0.4, "bn": False},
    {"hidden": [64, 64],  "feat": 16, "dropout": 0.2, "bn": False},
]
try:
    import tensorflow as _tf          # noqa: F401
    HAVE_TF = True
except Exception:
    HAVE_TF = False
BACKEND = "tensorflow" if HAVE_TF else "sklearn"

from sklearn.neural_network import MLPClassifier
def _hidden(clf, X):
    a = np.asarray(X, dtype=float)
    for i in range(len(clf.coefs_) - 1):
        a = np.maximum(a @ clf.coefs_[i] + clf.intercepts_[i], 0.0)
    return a

def fit_feature_extractors(Xtr, ytr):
    if HAVE_TF:
        from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        exts = []
        for arch in MLP_ARCHS:
            reset_seeds()
            inp = Input(shape=(Xtr.shape[1],)); x = inp
            for i, u in enumerate(arch["hidden"]):
                x = Dense(u, activation="relu")(x)
                if arch.get("bn") and i == 0: x = BatchNormalization()(x)
                if arch.get("dropout"): x = Dropout(arch["dropout"])(x)
            feats = Dense(arch["feat"], activation="relu", name="features")(x)
            out = Dense(1, activation="sigmoid")(feats)
            m = Model(inp, out); m.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
            m.fit(Xtr, ytr, validation_split=0.1, epochs=40, batch_size=256, verbose=0,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)])
            exts.append(Model(m.input, m.get_layer("features").output))
        return lambda X: np.concatenate([e.predict(X, verbose=0) for e in exts], 1)
    else:
        mlps = []
        for arch in MLP_ARCHS:
            sizes = tuple(arch["hidden"] + [arch["feat"]])
            c = MLPClassifier(hidden_layer_sizes=sizes, activation="relu", alpha=1e-3,
                              batch_size=256, early_stopping=True, n_iter_no_change=6,
                              max_iter=200, random_state=SEED)
            c.fit(Xtr, ytr); mlps.append(c)
        return lambda X: np.concatenate([_hidden(c, X) for c in mlps], 1)

def best_f1_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.05, 0.95, 181)))
    best_t, best_f = 0.5, -1.0
    for t in grid:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f: best_f, best_t = f, float(t)
    return best_t

def base_classifiers():
    return {"rf": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
            "et": ExtraTreesClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
            "gb": GradientBoostingClassifier(random_state=SEED),
            "xgb": make_xgb(),
            "lr": LogisticRegression(max_iter=1000)}

def max_confidence_predict(models, Xte):
    probs = np.stack([m.predict_proba(Xte) for m in models.values()], 0)
    best = probs.max(2).argmax(0)
    return probs[best, np.arange(probs.shape[1]), 1]

def soft_vote_predict(models, Xte):
    # Average P(readmit) across the base classifiers. Smooth, well-calibrated
    # probabilities -> sane operating threshold (max-confidence collapsed to thr=1.0, F1=0).
    return np.mean([m.predict_proba(Xte)[:, 1] for m in models.values()], axis=0)

# ----------------------------------------------------------------------
def make_splits(y, groups):
    if SGK:
        return list(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
                    .split(np.zeros(len(y)), y, groups))
    from sklearn.model_selection import GroupKFold
    return list(GroupKFold(n_splits=N_SPLITS).split(np.zeros(len(y)), y, groups))

def load_frame(path):
    """Read a config CSV and ONE-HOT encode categoricals (leak-free: deterministic mapping)."""
    df = pd.read_csv(path)
    if df[TARGET].dtype == object:
        df[TARGET] = (df[TARGET].astype(str) == "<30").astype(int)
    y = df[TARGET].astype(int).values
    g = df[GROUP].values
    X = df.drop(columns=[TARGET, GROUP])
    for c in CAT_AS_STR:
        if c in X.columns:
            X[c] = X[c].astype(str)                     # numeric-coded nominal -> category
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    X = pd.get_dummies(X, columns=cat, dummy_na=False).astype(float)
    out = X.copy()
    out[TARGET] = y
    out[GROUP] = g
    return out

def run_config(df, feats, splits, name):
    X = df[feats].values.astype(float); y = df[TARGET].values.astype(int); g = df[GROUP].values
    oof_true, oof_prob, oof_idx, per_fold = [], [], [], []
    for k, (tr, te) in enumerate(splits):
        assert len(set(g[tr]) & set(g[te])) == 0, "PATIENT LEAK!"
        sc = StandardScaler().fit(X[tr]); Xtr0, Xte = sc.transform(X[tr]), sc.transform(X[te])
        ytr0, yte = y[tr], y[te]
        Xr, yr = TomekLinks().fit_resample(Xtr0, ytr0)
        Xr, yr = SMOTE(sampling_strategy=0.5, random_state=SEED).fit_resample(Xr, yr)
        extract = fit_feature_extractors(Xr, yr)
        feat = lambda Z: np.concatenate([Z, extract(Z)], axis=1)   # KEEP raw features + learned MLP features
        clfs = base_classifiers()
        for c in clfs.values(): c.fit(feat(Xr), yr)
        thr = best_f1_threshold(ytr0, soft_vote_predict(clfs, feat(Xtr0)))
        prob = soft_vote_predict(clfs, feat(Xte)); pred = (prob >= thr).astype(int)
        per_fold.append(dict(fold=k+1, thr=round(thr, 3), acc=accuracy_score(yte, pred),
                             auc=roc_auc_score(yte, prob), prauc=average_precision_score(yte, prob),
                             brier=brier_score_loss(yte, prob), prec=precision_score(yte, pred, zero_division=0),
                             rec=recall_score(yte, pred), f1=f1_score(yte, pred)))
        oof_true.append(yte); oof_prob.append(prob); oof_idx.append(te)
        print(f"[{name}] fold {k+1}: AUC={per_fold[-1]['auc']:.3f} "
              f"PR-AUC={per_fold[-1]['prauc']:.3f} F1={per_fold[-1]['f1']:.3f} (thr={thr:.2f})")
    return dict(per_fold=per_fold, oof_true=np.concatenate(oof_true),
                oof_prob=np.concatenate(oof_prob), oof_idx=np.concatenate(oof_idx))

def summarize(per_fold):
    out = {}
    for key in per_fold[0]:
        if key == "fold": continue
        v = np.array([f[key] for f in per_fold]); m, s = v.mean(), v.std(ddof=1)
        ci = 1.96 * s / np.sqrt(len(v)); out[key] = (m, s, m - ci, m + ci)
    return out

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

def save_plots(name, res):
    yt, yp = res["oof_true"], res["oof_prob"]
    fpr, tpr, _ = roc_curve(yt, yp)
    plt.figure(figsize=(5,5)); plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(yt,yp):.3f}")
    plt.plot([0,1],[0,1],"--",color="gray"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC - {name}"); plt.legend(); plt.savefig(os.path.join(RESULTS, f"roc_{name}.png"), dpi=200, bbox_inches="tight"); plt.close()
    pr, rc, _ = precision_recall_curve(yt, yp)
    plt.figure(figsize=(5,5)); plt.plot(rc, pr, label=f"PR-AUC={average_precision_score(yt,yp):.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR curve - {name}"); plt.legend()
    plt.savefig(os.path.join(RESULTS, f"pr_{name}.png"), dpi=200, bbox_inches="tight"); plt.close()
    frac, mean_pred = calibration_curve(yt, yp, n_bins=10)
    plt.figure(figsize=(5,5)); plt.plot(mean_pred, frac, "o-"); plt.plot([0,1],[0,1],"--",color="gray")
    plt.xlabel("Mean predicted"); plt.ylabel("Observed"); plt.title(f"Calibration - {name} (Brier={brier_score_loss(yt,yp):.3f})")
    plt.savefig(os.path.join(RESULTS, f"calibration_{name}.png"), dpi=200, bbox_inches="tight"); plt.close()
    cm = confusion_matrix(yt, (yp>=0.5).astype(int), normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=["No", "<30"]).plot(cmap="Blues", values_format=".2f")
    plt.title(f"Confusion - {name}"); plt.savefig(os.path.join(RESULTS, f"confusion_{name}.png"), dpi=200, bbox_inches="tight"); plt.close()

# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sklearn, imblearn
    print(f"XGBoost {xgb.__version__} | GPU: {'YES (device=cuda)' if USE_GPU else 'no - CPU hist'}")
    print(f"Feature-extractor backend: {BACKEND}"
          f"{'  (TensorFlow not found - using scikit-learn MLPs on CPU)' if not HAVE_TF else ''}")
    with open(os.path.join(RESULTS, "versions.txt"), "w") as fh:
        fh.write(f"numpy {np.__version__}\nsklearn {sklearn.__version__}\n"
                 f"imblearn {imblearn.__version__}\nxgboost {xgb.__version__}\n"
                 f"xgb_gpu {USE_GPU}\nfeature_extractor_backend {BACKEND}\n")
        try:
            import tensorflow as tf
            fh.write(f"tensorflow {tf.__version__}\n")
            fh.write(f"tf_gpus {[d.name for d in tf.config.list_physical_devices('GPU')]}\n")
        except Exception:
            fh.write("tensorflow NOT installed\n")

    frames = {n: load_frame(os.path.join(PROCESSED, f)) for n, f in CONFIGS.items()}
    base = frames["baseline"]
    splits = make_splits(base[TARGET].values.astype(int), base[GROUP].values)
    print(f"{'StratifiedGroupKFold' if SGK else 'GroupKFold'} x{N_SPLITS}, grouped by {GROUP}\n")

    results, rows = {}, []
    for name, df in frames.items():
        feats = [c for c in df.columns if c not in (TARGET, GROUP)]
        results[name] = run_config(df, feats, splits, name)
        save_plots(name, results[name])
        for k, (m, sd, lo, hi) in summarize(results[name]["per_fold"]).items():
            rows.append(dict(config=name, metric=k, mean=round(m,4), sd=round(sd,4),
                             ci_low=round(lo,4), ci_high=round(hi,4)))
        pd.DataFrame(results[name]["per_fold"]).to_csv(
            os.path.join(RESULTS, f"per_fold_{name}.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "metrics_summary.csv"), index=False)

    b = results["baseline"]; ob = np.argsort(b["oof_idx"])
    yb, pb = b["oof_true"][ob], b["oof_prob"][ob]
    with open(os.path.join(RESULTS, "delong.txt"), "w") as fh:
        for name in ENRICHED:
            r = results[name]; orr = np.argsort(r["oof_idx"]); pr = r["oof_prob"][orr]
            ae, ab, p = delong_test(yb, pr, pb)
            line = f"{name} vs baseline: AUC {ae:.3f} vs {ab:.3f} | DeLong p = {p:.4f}"
            print(line); fh.write(line + "\n")

    # SHAP on the clinical-state model
    try:
        import shap
        df = frames["mean_clinical"]; feats = [c for c in df.columns if c not in (TARGET, GROUP)]
        tr, te = splits[0]
        sc = StandardScaler().fit(df[feats].values[tr])
        Xr = sc.transform(df[feats].values[tr]); yr = df[TARGET].values[tr]
        Xr, yr = SMOTE(sampling_strategy=0.5, random_state=SEED).fit_resample(Xr, yr)
        extract = fit_feature_extractors(Xr, yr)
        feat = lambda Z: np.concatenate([Z, extract(Z)], axis=1)
        clfs = base_classifiers()
        for c in clfs.values(): c.fit(feat(Xr), yr)
        predict_fn = lambda Z: soft_vote_predict(clfs, feat(sc.transform(Z)))
        expl = shap.KernelExplainer(predict_fn, shap.sample(df[feats].values[tr], 50))
        sv = expl.shap_values(df[feats].values[te][:100])
        shap.summary_plot(sv, df[feats].values[te][:100], feature_names=feats, show=False, plot_type="bar")
        plt.savefig(os.path.join(RESULTS, "shap_summary_clinical.png"), dpi=200, bbox_inches="tight"); plt.close()
    except Exception as e:
        print("SHAP step skipped:", e)

    print("\nAll results written to", RESULTS)

# ======================================================================
# GPU NOTES (RTX 3050)
# ----------------------------------------------------------------------
# XGBoost (main speed-up, easiest):
#     pip install --upgrade xgboost           # 2.x wheels ship with CUDA support
#     -> this script auto-uses device="cuda".  Verify: it prints "GPU: YES" at start,
#        or check nvidia-smi during the run.
#
# TensorFlow MLPs on GPU (optional; the MLPs are tiny, CPU is fine):
#     Native Windows: the LAST TF version with GPU is 2.10.
#         pip install "tensorflow==2.10" and install CUDA 11.2 + cuDNN 8.1.
#     Newer TF on Windows: run inside WSL2 (Ubuntu) with the NVIDIA CUDA-on-WSL driver,
#         then `pip install tensorflow[and-cuda]`.
#     If TensorFlow isn't set up, the script silently uses scikit-learn MLPs on CPU
#     (same 3-network -> 112-D fusion), so it always runs.
# ======================================================================
