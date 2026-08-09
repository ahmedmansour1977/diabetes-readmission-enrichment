"""
STEP 6 - Controlled demonstration of the evaluation-leakage effect.

Runs the SAME model (tuned XGBoost) on the SAME data under two protocols and reports the gap:
  A) LEAKY   : SMOTE applied to the whole dataset BEFORE splitting, then encounter-level
               StratifiedKFold (same patient can appear in train and test). = the common practice.
  B) CORRECT : patient-grouped StratifiedGroupKFold, class weighting, no oversampling leakage.

The difference is a first-hand, model-held-constant measurement of how much data leakage
inflates AUROC on this benchmark. Writes results/leakage_demo.txt.

Run:  python 06_leakage_demo.py
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import xgboost as xgb

BASE = os.environ.get(
    "ENRICHMENT_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED = os.path.join(BASE, "DATA", "processed"); RESULTS = os.path.join(BASE, "results")
TARGET, GROUP, SEED = "readmitted", "patient_nbr", 42
CAT_AS_STR = ["admission_type_id", "discharge_disposition_id", "admission_source_id", "race"]

def gpu():
    try:
        xgb.train({"tree_method": "hist", "device": "cuda"}, xgb.DMatrix(np.zeros((8, 2)), label=[0, 1]*4), num_boost_round=1); return True
    except Exception: return False
GPU = gpu()
def make_xgb(spw):
    kw = dict(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
              min_child_weight=5, reg_lambda=2.0, scale_pos_weight=spw, eval_metric="logloss", random_state=SEED)
    kw.update(dict(tree_method="hist", device="cuda") if GPU else dict(tree_method="hist"))
    return XGBClassifier(**kw)

df = pd.read_csv(os.path.join(PROCESSED, "uci_baseline.csv"))
if df[TARGET].dtype == object: df[TARGET] = (df[TARGET].astype(str) == "<30").astype(int)
y = df[TARGET].astype(int).values; g = df[GROUP].values
X = df.drop(columns=[TARGET, GROUP])
for c in CAT_AS_STR:
    if c in X.columns: X[c] = X[c].astype(str)
X = pd.get_dummies(X, columns=[c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])], dummy_na=False).astype(float).values

# ---------- A) LEAKY: SMOTE whole dataset, then encounter-level KFold ----------
Xa, ya = SMOTE(random_state=SEED).fit_resample(X, y)          # <-- leakage: before the split
aucs_a, accs_a = [], []
for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xa, ya):
    sc = StandardScaler().fit(Xa[tr])
    m = make_xgb(1.0); m.fit(sc.transform(Xa[tr]), ya[tr])
    p = m.predict_proba(sc.transform(Xa[te]))[:, 1]
    aucs_a.append(roc_auc_score(ya[te], p)); accs_a.append(accuracy_score(ya[te], (p >= 0.5).astype(int)))

# ---------- B) CORRECT: patient-grouped CV, class weighting ----------
aucs_b = []
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g):
    sc = StandardScaler().fit(X[tr])
    spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
    m = make_xgb(spw); m.fit(sc.transform(X[tr]), y[tr])
    p = m.predict_proba(sc.transform(X[te]))[:, 1]
    aucs_b.append(roc_auc_score(y[te], p))

A, B = np.mean(aucs_a), np.mean(aucs_b)
lines = [
    f"SAME MODEL (tuned XGBoost), two evaluation protocols:",
    f"  A) LEAKY   (SMOTE-before-split + encounter-level KFold): AUROC = {A:.3f}, accuracy = {np.mean(accs_a):.3f}",
    f"  B) CORRECT (patient-grouped CV + class weighting)      : AUROC = {B:.3f}",
    f"  => Leakage inflates AUROC by {A-B:+.3f} on this benchmark, with no change to model or data.",
]
os.makedirs(RESULTS, exist_ok=True)
open(os.path.join(RESULTS, "leakage_demo.txt"), "w").write("\n".join(lines))
print("\n".join(lines))
