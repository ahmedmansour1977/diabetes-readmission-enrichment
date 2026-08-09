"""
STEP 5 - Publication figures for the baseline XGBoost model (Supplementary Figure S1).
Regenerates the out-of-fold predictions on uci_baseline.csv (same leak-free protocol as step 3)
and saves ROC, precision-recall, and calibration curves to results/.

Run:  python 05_make_figures.py
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, brier_score_loss
from xgboost import XGBClassifier
import xgboost as xgb

BASE = os.environ.get(
    "ENRICHMENT_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED = os.path.join(BASE, "DATA", "processed"); RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)
TARGET, GROUP, SEED = "readmitted", "patient_nbr", 42
CAT_AS_STR = ["admission_type_id", "discharge_disposition_id", "admission_source_id", "race"]
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

df = pd.read_csv(os.path.join(PROCESSED, "uci_baseline.csv"))
if df[TARGET].dtype == object: df[TARGET] = (df[TARGET].astype(str) == "<30").astype(int)
y = df[TARGET].astype(int).values; g = df[GROUP].values
X = df.drop(columns=[TARGET, GROUP])
for c in CAT_AS_STR:
    if c in X.columns: X[c] = X[c].astype(str)
X = pd.get_dummies(X, columns=[c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])], dummy_na=False).astype(float).values

def gpu():
    try:
        xgb.train({"tree_method": "hist", "device": "cuda"}, xgb.DMatrix(np.zeros((8, 2)), label=[0, 1]*4), num_boost_round=1); return True
    except Exception: return False
GPU = gpu()
oof = np.zeros(len(y))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g):
    sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    kw = dict(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
              min_child_weight=5, reg_lambda=2.0, scale_pos_weight=(y[tr] == 0).sum()/max((y[tr] == 1).sum(), 1),
              eval_metric="logloss", random_state=SEED)
    kw.update(dict(tree_method="hist", device="cuda") if GPU else dict(tree_method="hist"))
    m = XGBClassifier(**kw); m.fit(Xtr, y[tr]); oof[te] = m.predict_proba(Xte)[:, 1]

# ROC
fpr, tpr, _ = roc_curve(y, oof)
plt.figure(figsize=(4.6, 4.6)); plt.plot(fpr, tpr, color="#1f3864", label=f"XGBoost (AUC = {roc_auc_score(y, oof):.3f})")
plt.plot([0, 1], [0, 1], "--", color="#999"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curve"); plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "figS1a_roc.png"), dpi=200); plt.close()
# PR
pr, rc, _ = precision_recall_curve(y, oof)
plt.figure(figsize=(4.6, 4.6)); plt.plot(rc, pr, color="#1f3864", label=f"PR-AUC = {average_precision_score(y, oof):.3f}")
plt.axhline(y.mean(), ls="--", color="#999", label=f"prevalence = {y.mean():.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall curve"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "figS1b_pr.png"), dpi=200); plt.close()
# Calibration
frac, mean_pred = calibration_curve(y, oof, n_bins=10)
plt.figure(figsize=(4.6, 4.6)); plt.plot(mean_pred, frac, "o-", color="#1f3864"); plt.plot([0, 1], [0, 1], "--", color="#999")
plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
plt.title(f"Calibration (Brier = {brier_score_loss(y, oof):.3f})"); plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "figS1c_calibration.png"), dpi=200); plt.close()
print("Saved figS1a_roc.png, figS1b_pr.png, figS1c_calibration.png to", RESULTS)
