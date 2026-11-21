"""
STEP 19 - Simulation: when does resampling before the split inflate AUROC, and why?

Data with a KNOWN discrimination are generated from a logistic model with 20 independent
predictors and an event rate near 10%. The same latent data are presented to the learner
either as continuous predictors or as binary predictors (each dichotomised at zero), so
that the only thing that changes is whether an interpolated record can take a value that
no real record takes.

Pipelines:  correct (no resampling) | SMOTE before the split | SMOTE separately in the
            training folds and in the test folds (nothing crosses the boundary)
Learners:   XGBoost (the fixed configuration of the paper) | logistic regression
Sizes:      n = 1,000 and n = 10,000
Replicates: 10 per scenario

Usage:  python 19_simulation.py <out_dir> [n_reps]
"""
import sys, os, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mech
from sklearn.metrics import roc_auc_score

P = 20


def generate(n, kind, rng):
    z = rng.standard_normal((n, P))
    X = z if kind == "continuous" else (z > 0).astype(float)
    signal = z if kind == "continuous" else 2.0 * X - 1.0
    beta = np.full(P, 0.17)
    eta = -2.35 + signal @ beta
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y, roc_auc_score(y, eta)


if __name__ == "__main__":
    out_dir = sys.argv[1]; reps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    out_csv = os.path.join(out_dir, "simulation.csv")
    rows = pd.read_csv(out_csv).to_dict("records") if os.path.exists(out_csv) else []
    done = {(r["n"], r["kind"], r["rep"], r["learner"], r["method"]) for r in rows}
    for n in (1000, 10000):
        for kind in ("continuous", "binary"):
            for rep in range(reps):
                rng = np.random.default_rng(100000 * (kind == "binary") + 1000 * rep + n)
                X, y, true_auc = generate(n, kind, rng)
                for learner in ("xgb", "lr"):
                    for method in ("none", "smote", "postsplit_both_separately"):
                        if (n, kind, rep, learner, method) in done: continue
                        if method == "postsplit_both_separately":
                            r = mech.run_postsplit(X, y, None, "both_separately", learner, rep)
                        else:
                            r = mech.run(X, y, None, method, learner, rep, grouped=True)
                        r.update(n=n, kind=kind, rep=rep, true_auroc=true_auc)
                        rows.append(r); pd.DataFrame(rows).to_csv(out_csv, index=False)
            df = pd.DataFrame(rows); s = df[(df.n == n) & (df.kind == kind)]
            print(n, kind, "true AUROC %.3f prevalence %.3f" % (s.true_auroc.mean(), s.prevalence.mean()))
            print(s.groupby(["learner", "method"])[["auroc_pooled", "auroc_real", "auroc_new"]]
                  .mean().round(3).to_string(), flush=True)
