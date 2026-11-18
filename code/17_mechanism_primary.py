"""
STEP 17 - Where does the inflation live? (Diabetes 130-US Hospitals)

Extends leakage_mechanism (rescoring on real encounters) by tagging every out-of-fold
prediction as real or created-by-resampling and reporting, for each pre-split resampler
and each learner:

    AUROC_pooled, AUROC_real, AUROC_new, pi_new, and the identity check.

Resamplers (all applied to the FULL dataset before the split, patient-grouped folds):
    smote        SMOTE, k=5 (the error studied in the paper)
    smote_round  the same synthetic rows with integer-valued features rounded back to
                 integers, so that no single feature value is impossible for a real record
    ros          random oversampling (exact copies of real minority records)
Learners:
    xgb          the fixed XGBoost configuration of the paper
    lr           L2 logistic regression (C=1)

Usage:  python 17_mechanism_primary.py <uci_baseline.csv> <out_dir> [seed ...]
"""
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mech

TARGET, GROUP = "readmitted", "patient_nbr"
CAT_AS_STR = ["admission_type_id", "discharge_disposition_id", "admission_source_id", "race"]


def load_frame(path):
    """Verbatim from 13_leakage_factorial.py."""
    df = pd.read_csv(path, low_memory=False)
    if df[TARGET].dtype == object:
        df[TARGET] = (df[TARGET].astype(str) == "<30").astype(int)
    y = df[TARGET].astype(int).values; g = df[GROUP].values
    X = df.drop(columns=[TARGET, GROUP])
    for c in CAT_AS_STR:
        if c in X.columns: X[c] = X[c].astype(str)
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    return pd.get_dummies(X, columns=cat, dummy_na=False).astype(float), y, g


if __name__ == "__main__":
    path, out_dir = sys.argv[1], sys.argv[2]
    seeds = [int(s) for s in sys.argv[3:]] or [42, 1, 2]
    os.makedirs(out_dir, exist_ok=True)
    Xdf, y, g = load_frame(path)
    X = Xdf.values.astype(float)
    assert X.shape == (98490, 155) and y.sum() == 11271 and len(np.unique(g)) == 69311
    out_csv = os.path.join(out_dir, "mechanism_primary.csv")
    rows = pd.read_csv(out_csv).to_dict("records") if os.path.exists(out_csv) else []
    done = {(r["method"], r["learner"], r["seed"]) for r in rows}

    with open(os.path.join(out_dir, "off_support_primary.json"), "w") as fh:
        json.dump(mech.off_support(X, y, seeds[0]), fh, indent=1)

    for seed in seeds:
        for learner in ("xgb", "lr"):
            for method in ("none", "smote", "smote_round", "ros"):
                if (method, learner, seed) in done: continue
                t = time.time()
                r = mech.run(X, y, g, method, learner, seed, grouped=True)
                r["seconds"] = round(time.time() - t, 1)
                rows.append(r)
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
