"""
STEP 18 - Does the decomposition hold on other public clinical datasets?

Five additional public datasets with no missing values (so that no imputation step can
itself leak), spanning sample size, imbalance and feature type. Each is run through the
same pipelines as the primary dataset with the same two learners and no tuning.

    cdc_dhi        CDC Diabetes Health Indicators (UCI 891)    outcome: Diabetes_binary
    actg175        AIDS Clinical Trials Group Study 175 (UCI 890)  outcome: cid
                   baseline covariates only: time, offtrt, cd420 and cd820 are measured
                   after baseline and are removed; zprior is constant and is removed
    thoracic       Thoracic Surgery Data (UCI 277)              outcome: Risk1Yr (death within 1 year)
    ctg            Cardiotocography (UCI 193)                   outcome: NSP = 3 (pathologic)
                   the CLASS pattern label is a second target and is not used as a feature
    heart_failure  Heart Failure Clinical Records (UCI 519)     outcome: death_event
                   follow-up time is removed (it is not known at baseline)
    mammography    Mammography (OpenML 310)                     outcome: class = 1

Usage:  python 18_replication_datasets.py <data_dir> <out_dir>
"""
import sys, os, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mech


def load(name, d):
    df = pd.read_csv(os.path.join(d, name + ".csv"))
    if name == "cdc_dhi":
        y = df.pop("Diabetes_binary").astype(int).values
    elif name == "actg175":
        y = df.pop("cid").astype(int).values
        df = df.drop(columns=["time", "offtrt", "cd420", "cd820", "zprior"])
    elif name == "thoracic":
        y = (df.pop("Risk1Yr").astype(str) == "T").astype(int).values
        for c in df.columns:
            if set(df[c].astype(str).unique()) <= {"T", "F"}:
                df[c] = (df[c].astype(str) == "T").astype(int)
        cat = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        df = pd.get_dummies(df, columns=cat, dummy_na=False)
    elif name == "ctg":
        y = (df.pop("NSP").astype(int) == 3).astype(int).values
        df = df.drop(columns=["CLASS"])
    elif name == "heart_failure":
        y = df.pop("death_event").astype(int).values
        df = df.drop(columns=["time"])
    elif name == "mammography":
        y = (df.pop("class").astype(str).str.strip("'") == "1").astype(int).values
    X = df.astype(float).values
    assert not np.isnan(X).any()
    return X, y


SEEDS = {"cdc_dhi": [42], "default": [42, 1, 2, 3, 4]}
METHODS = ["none", "smote", "smote_round", "ros", "postsplit_both"]

if __name__ == "__main__":
    d, out_dir = sys.argv[1], sys.argv[2]
    names = sys.argv[3:] or ["thoracic", "heart_failure", "actg175", "ctg", "mammography", "cdc_dhi"]
    out_csv = os.path.join(out_dir, "replication_datasets.csv")
    rows = pd.read_csv(out_csv).to_dict("records") if os.path.exists(out_csv) else []
    done = {(r["dataset"], r["method"], r["learner"], r["seed"]) for r in rows}
    desc = {}
    for name in names:
        X, y = load(name, d)
        ic, bc = mech.integer_columns(X), mech.binary_columns(X)
        os_ = mech.off_support(X, y, 42)
        desc[name] = dict(n=len(y), events=int(y.sum()), prevalence=float(y.mean()), p=X.shape[1],
                          integer_cols=int(ic.sum()), binary_cols=int(bc.sum()),
                          share_new_rows_off_support=os_["share_rows_off_support"])
        print(name, desc[name], flush=True)
        for seed in SEEDS.get(name, SEEDS["default"]):
            for learner in ("xgb", "lr"):
                for method in METHODS:
                    stored = "postsplit_both_separately" if method == "postsplit_both" else method
                    if (name, stored, learner, seed) in done: continue
                    t = time.time()
                    if method == "postsplit_both":
                        r = mech.run_postsplit(X, y, None, "both_separately", learner, seed)
                    else:
                        r = mech.run(X, y, None, method, learner, seed, grouped=True)
                    r["dataset"] = name; r["seconds"] = round(time.time() - t, 1)
                    rows.append(r); pd.DataFrame(rows).to_csv(out_csv, index=False)
        sub = pd.DataFrame([r for r in rows if r["dataset"] == name])
        print(sub.groupby(["learner", "method"])[["auroc_pooled", "auroc_real", "auroc_new", "pi_new"]]
              .mean().round(4).to_string(), flush=True)
        old = json.load(open(os.path.join(out_dir, "replication_descriptives.json"))) \
            if os.path.exists(os.path.join(out_dir, "replication_descriptives.json")) else {}
        old.update(desc)
        json.dump(old, open(os.path.join(out_dir, "replication_descriptives.json"), "w"), indent=1)
