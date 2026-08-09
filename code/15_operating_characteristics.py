"""
STEP 15 - Operating characteristics by intervention capacity.

WHY
    A single confusion matrix describes the model at one arbitrary threshold. What a
    discharge-planning service actually needs to know is: if we can follow up k% of
    discharges, how many of the 30-day readmissions do we catch, and how many people do we
    contact for each one caught? This script answers that directly, and includes the
    confusion matrix at the F1-selected threshold reported in the manuscript as one row.

WHAT IT REPORTS
    For each capacity level (the top k% of encounters by predicted risk):
        threshold, flagged n, TP, FP, FN, TN
        recall (sensitivity)   - share of readmissions captured
        precision (PPV)        - share of flagged encounters that are readmitted
        specificity
        NNF                    - number needed to flag per readmission captured (1/PPV)
        lift                   - PPV divided by the 11.4% prevalence
    Repeated for the whole cohort and for each age band, since Section 4.6 shows
    discrimination is weakest in the oldest and largest stratum.

Outputs: results/operating_characteristics.csv , .txt , fig8_operating_characteristics.png

Run:  python 15_operating_characteristics.py
Prerequisite: 09_delta_auc_clustered.py (writes results/oof_predictions.npz)
"""

import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score

BASE = os.environ.get(
    "ENRICHMENT_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED = os.path.join(BASE, "DATA", "processed")
RESULTS   = os.path.join(BASE, "results")
TARGET = "readmitted"
CACHE = os.path.join(RESULTS, "oof_predictions.npz")
CAPACITIES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
NAVY, RED = "#1f3864", "#b3261e"

plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

def counts(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    spec = tn / max(tn + fp, 1)
    return tp, fp, fn, tn, rec, prec, spec

def best_f1_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.05, 0.95, 181)))
    bt, bf = 0.5, -1.0
    for t in grid:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, float(t)
    return bt

if __name__ == "__main__":
    out = []
    def log(s): print(s); out.append(s)

    if not os.path.exists(CACHE):
        raise SystemExit("results/oof_predictions.npz not found - run "
                         "09_delta_auc_clustered.py first.")
    p_all = np.load(CACHE)["baseline"]

    df = pd.read_csv(os.path.join(PROCESSED, "uci_baseline.csv"))
    if df[TARGET].dtype == object:
        df[TARGET] = (df[TARGET].astype(str) == "<30").astype(int)
    y_all = df[TARGET].astype(int).values
    age = df["age"].astype(str).values
    prev = y_all.mean()

    log(f"Baseline XGBoost, out-of-fold. n={len(y_all):,}  events={int(y_all.sum()):,}  "
        f"prevalence={prev:.4f}  AUROC={roc_auc_score(y_all, p_all):.4f}")
    log("NNF = number needed to flag per readmission captured; lift = PPV / prevalence.\n")

    rows = []
    def block(label, y, p):
        log(f"--- {label}  (n={len(y):,}, events={int(y.sum()):,}, "
            f"prevalence={y.mean():.3f}) ---")
        log(f"  {'capacity':<22s} {'thr':>6s} {'flagged':>8s} {'TP':>6s} {'FP':>7s} "
            f"{'FN':>6s} {'recall':>7s} {'PPV':>7s} {'spec':>7s} {'NNF':>6s} {'lift':>5s}")
        pts = [(f"top {int(c*100)}%", float(np.quantile(p, 1 - c))) for c in CAPACITIES]
        pts.append(("F1-selected (reported)", best_f1_threshold(y, p)))
        for name, thr in pts:
            tp, fp, fn, tn, rec, prec, spec = counts(y, p, thr)
            nnf = 1 / prec if prec > 0 else float("nan")
            log(f"  {name:<22s} {thr:6.3f} {tp+fp:8,d} {tp:6,d} {fp:7,d} {fn:6,d} "
                f"{rec:7.3f} {prec:7.3f} {spec:7.3f} {nnf:6.1f} {prec/y.mean():5.2f}")
            rows.append(dict(stratum=label, operating_point=name, threshold=round(thr, 3),
                             flagged=tp + fp, tp=tp, fp=fp, fn=fn, tn=tn,
                             recall=round(rec, 4), ppv=round(prec, 4),
                             specificity=round(spec, 4), nnf=round(nnf, 2),
                             lift=round(prec / y.mean(), 2)))
        log("")

    block("Whole cohort", y_all, p_all)
    for band in ["20-39", "40-59", ">=60"]:
        m = age == band
        if m.sum() >= 500:
            block(f"Age {band}", y_all[m], p_all[m])

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "operating_characteristics.csv"),
                              index=False)

    # ---- figure: recall and PPV against capacity ----
    caps = np.linspace(0.02, 0.40, 39)
    rec_c, ppv_c = [], []
    for c in caps:
        thr = float(np.quantile(p_all, 1 - c))
        _, _, _, _, rec, prec, _ = counts(y_all, p_all, thr)
        rec_c.append(rec); ppv_c.append(prec)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(caps * 100, np.array(rec_c) * 100, "-", color=NAVY,
            label="Readmissions captured (recall)")
    ax.plot(caps * 100, np.array(ppv_c) * 100, "-", color=RED,
            label="Flagged encounters readmitted (PPV)")
    ax.axhline(prev * 100, ls=":", color="#888", lw=1,
               label=f"prevalence ({prev*100:.1f}%)")
    ax.plot(caps * 100, caps * 100, "--", color="#bbb", lw=1, label="no-skill recall")
    ax.set_xlabel("Share of discharges flagged for follow-up (%)")
    ax.set_ylabel("Percent")
    ax.set_title("Operating characteristics by intervention capacity", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig8_operating_characteristics.png"), dpi=300)
    plt.close()
    log("Saved -> results/fig8_operating_characteristics.png , operating_characteristics.csv")

    with open(os.path.join(RESULTS, "operating_characteristics.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print("\nLog -> results/operating_characteristics.txt")
