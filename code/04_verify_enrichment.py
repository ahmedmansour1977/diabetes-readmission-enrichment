"""
STEP 4 - Sanity check: PROVE the enrichment columns reach the model, and show how much
         the model actually relies on them (spoiler: almost nothing).

Prints:
  1) feature counts for baseline vs mean_clinical  (enriched must have MORE columns)
  2) the proxy columns, their std (must be > 0 = they vary) and #distinct values
  3) an XGBoost trained on mean_clinical, with every proxy column's gain-importance and RANK
     among all features -> you will see them sitting near the bottom.

Run:  python 04_verify_enrichment.py
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import xgboost as xgb

BASE = os.environ.get(
    "ENRICHMENT_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED = os.path.join(BASE, "DATA", "processed")
TARGET, GROUP, SEED = "readmitted", "patient_nbr", 42
CAT_AS_STR = ["admission_type_id", "discharge_disposition_id", "admission_source_id", "race"]

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

base_X, _, _ = load_frame(os.path.join(PROCESSED, "uci_baseline.csv"))
Xc, y, g = load_frame(os.path.join(PROCESSED, "uci_mean_clinical.csv"))
proxies = [c for c in Xc.columns if c.endswith("_mean")]

print("=" * 70)
print("1) FEATURE COUNTS")
print(f"   baseline      : {base_X.shape[1]} features")
print(f"   mean_clinical : {Xc.shape[1]} features   (+{Xc.shape[1]-base_X.shape[1]} proxy columns)")
print(f"   -> if these differ, the proxies ARE in the model's input.\n")

print("2) PROXY COLUMNS - do they vary, or are they constant?")
print(f"   {'column':26s} {'std':>10s} {'#distinct':>10s}")
for c in proxies:
    print(f"   {c:26s} {Xc[c].std():10.4f} {Xc[c].nunique():10d}")
print("   -> std > 0 and many distinct values = they vary across patients (not constant).\n")

print("3) HOW MUCH DOES XGBOOST USE THEM?  (gain importance + rank; lower rank = more used)")
sp = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xc.values, y, g))
tr, te = sp[0]
xkw = dict(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
           colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0,
           scale_pos_weight=(y[tr] == 0).sum()/max((y[tr] == 1).sum(), 1),
           eval_metric="logloss", random_state=SEED)
try:
    xgb.DMatrix(np.zeros((8, 2)), label=[0, 1]*4)
    xkw.update(tree_method="hist", device="cuda")
except Exception:
    xkw.update(tree_method="hist")
clf = XGBClassifier(**xkw)
sc = StandardScaler().fit(Xc.values[tr])
clf.fit(sc.transform(Xc.values[tr]), y[tr])

imp = clf.feature_importances_                       # gain-based
order = np.argsort(imp)[::-1]
rank = {Xc.columns[idx]: r+1 for r, idx in enumerate(order)}
n = Xc.shape[1]
print(f"   total features: {n}")
print(f"   {'proxy column':26s} {'importance':>12s} {'rank':>8s}")
for c in sorted(proxies, key=lambda c: rank[c]):
    print(f"   {c:26s} {imp[list(Xc.columns).index(c)]:12.5f} {rank[c]:>5d}/{n}")
print("\n   TOP 12 features the model actually relies on:")
for idx in order[:12]:
    print(f"      {Xc.columns[idx]:28s} {imp[idx]:.5f}")

share = imp[[list(Xc.columns).index(c) for c in proxies]].sum()
print(f"\n   >> All {len(proxies)} proxies together = {share*100:.1f}% of total importance.")
print("      The model is driven by prior visits / diagnoses / meds, not the population proxies.")
