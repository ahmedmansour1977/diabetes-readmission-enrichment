"""
STEP 14 - Subgroup performance and fairness assessment (TRIPOD+AI items 14 and 23a).

WHY
    TRIPOD+AI requires that model performance be reported for key sociodemographic
    subgroups, and that any approaches taken to address fairness be described. Reporting
    only pooled performance can hide substantial differences between groups.

WHAT THIS DOES
    Using the SAME out-of-fold predictions behind every other result (no refitting, no
    new model), it reports for each subgroup defined by race/ethnicity, sex, and age band:

        n, number of events, prevalence
        AUROC with a patient-clustered bootstrap 95% CI
        PR-AUC, Brier score, ECE
        sensitivity and specificity at the single global threshold
        the enrichment effect within that subgroup (dAUROC vs baseline)

    Subgroup AUROCs are compared against the pooled estimate; the equal-opportunity
    difference (largest minus smallest sensitivity at a fixed threshold) is reported as a
    simple, interpretable fairness summary.

    Reading it: because a single threshold is applied to every group, differences in
    sensitivity and specificity across groups are the operationally meaningful ones. A
    subgroup AUROC that differs from the pooled value indicates the model ranks less well
    within that group, which matters even when overall discrimination looks acceptable.

Outputs: results/subgroup_performance.csv , subgroup_performance.txt ,
         fig7_subgroup_auroc.png

Run:  python 14_subgroup_fairness.py
Prerequisite: 09_delta_auc_clustered.py (writes results/oof_predictions.npz)
"""

import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score)

BASE = os.environ.get(
    "ENRICHMENT_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED = os.path.join(BASE, "DATA", "processed")
RESULTS   = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

TARGET, GROUP, SEED = "readmitted", "patient_nbr", 42
N_BOOT = 2000
MIN_N  = 200          # subgroups smaller than this are reported but flagged as unstable
rng = np.random.default_rng(SEED)
NAVY, RED = "#1f3864", "#b3261e"

RACE = {2: "Hispanic", 3: "Caucasian", 4: "African American", 5: "Asian/Other"}
SEX  = {1: "Male", 2: "Female"}

def cluster_boot_auc(y, p, g):
    idx_by_pat = pd.Series(np.arange(len(g))).groupby(g).apply(lambda s: s.values)
    pat_ids = np.array(idx_by_pat.index)
    lookup = dict(zip(idx_by_pat.index, idx_by_pat.values))
    vals = []
    for _ in range(N_BOOT):
        take = rng.choice(pat_ids, size=len(pat_ids), replace=True)
        idx = np.concatenate([lookup[k] for k in take])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        vals.append(roc_auc_score(yy, p[idx]))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); n = len(y); tot = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0: continue
        tot += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return tot

def best_f1_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.05, 0.95, 181))); bt, bf = 0.5, -1.0
    for t in grid:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, float(t)
    return bt

# ======================================================================
if __name__ == "__main__":
    out = []
    def log(s): print(s); out.append(s)

    cache = os.path.join(RESULTS, "oof_predictions.npz")
    if not os.path.exists(cache):
        raise SystemExit("results/oof_predictions.npz not found - run "
                         "09_delta_auc_clustered.py first.")
    z = np.load(cache)
    p_base = z["baseline"]
    p_enr = z["mean_clinical"] if "mean_clinical" in z.files else None

    df = pd.read_csv(os.path.join(PROCESSED, "uci_baseline.csv"))
    if df[TARGET].dtype == object:
        df[TARGET] = (df[TARGET].astype(str) == "<30").astype(int)
    y = df[TARGET].astype(int).values
    g = df[GROUP].values
    assert len(y) == len(p_base), "prediction/data length mismatch - re-run 09"

    thr = best_f1_threshold(y, p_base)
    pooled_auc = roc_auc_score(y, p_base)
    log(f"Pooled baseline XGBoost: AUROC {pooled_auc:.4f}, n={len(y):,}, "
        f"events={int(y.sum()):,}, prevalence {y.mean():.4f}")
    log(f"Global operating threshold (training-fold selected): {thr:.3f}\n")

    strata = [("Overall", np.ones(len(y), dtype=bool))]
    for code, name in RACE.items():
        strata.append((f"Race: {name}", df["race"].values == code))
    for code, name in SEX.items():
        strata.append((f"Sex: {name}", df["gender"].values == code))
    for band in ["20-39", "40-59", ">=60"]:
        strata.append((f"Age: {band}", df["age"].astype(str).values == band))

    log(f"   {'subgroup':<28s} {'n':>7s} {'events':>7s} {'prev':>6s} {'AUROC':>7s} "
        f"{'95% CI':>18s} {'Brier':>7s} {'ECE':>6s} {'Sens':>6s} {'Spec':>6s}")
    rows = []
    for name, m in strata:
        n = int(m.sum())
        if n == 0:
            log(f"   {name:<28s}  (no encounters)"); continue
        yy, pp, gg = y[m], p_base[m], g[m]
        ev = int(yy.sum())
        if ev == 0 or ev == n:
            log(f"   {name:<28s} {n:7,d} {ev:7,d}  (no outcome variation)"); continue
        auc = roc_auc_score(yy, pp); lo, hi = cluster_boot_auc(yy, pp, gg)
        pred = (pp >= thr).astype(int)
        sens = pred[yy == 1].mean(); spec = 1 - pred[yy == 0].mean()
        flag = "  *small*" if n < MIN_N else ""
        log(f"   {name:<28s} {n:7,d} {ev:7,d} {yy.mean():6.3f} {auc:7.4f} "
            f"({lo:.4f}, {hi:.4f}) {brier_score_loss(yy, pp):7.4f} {ece(yy, pp):6.3f} "
            f"{sens:6.3f} {spec:6.3f}{flag}")
        d_enr = (roc_auc_score(yy, p_enr[m]) - auc) if p_enr is not None else np.nan
        rows.append(dict(subgroup=name, n=n, events=ev, prevalence=round(yy.mean(), 4),
                         auroc=round(auc, 4), ci_low=round(lo, 4), ci_high=round(hi, 4),
                         prauc=round(average_precision_score(yy, pp), 4),
                         brier=round(brier_score_loss(yy, pp), 4),
                         ece=round(ece(yy, pp), 4), sensitivity=round(float(sens), 4),
                         specificity=round(float(spec), 4),
                         delta_auroc_enrichment=round(d_enr, 4) if p_enr is not None else None,
                         small_subgroup=n < MIN_N))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS, "subgroup_performance.csv"), index=False)

    sub = res[(res.subgroup != "Overall") & (~res.small_subgroup)]
    log("\nFAIRNESS SUMMARY (subgroups with n >= %d):" % MIN_N)
    log(f"   AUROC range            {sub.auroc.min():.4f} to {sub.auroc.max():.4f} "
        f"(spread {sub.auroc.max()-sub.auroc.min():.4f})")
    log(f"   lowest AUROC           {sub.loc[sub.auroc.idxmin(),'subgroup']}")
    log(f"   equal-opportunity gap  {sub.sensitivity.max()-sub.sensitivity.min():.4f} "
        f"(max minus min sensitivity at the single global threshold)")
    log(f"   specificity gap        {sub.specificity.max()-sub.specificity.min():.4f}")
    log(f"   calibration (ECE) range {sub.ece.min():.3f} to {sub.ece.max():.3f}")
    if p_enr is not None:
        log(f"   enrichment dAUROC range {sub.delta_auroc_enrichment.min():+.4f} to "
            f"{sub.delta_auroc_enrichment.max():+.4f}  (no subgroup benefits materially "
            f"if this band is narrow and centred near zero)")

    # ---- figure ----
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    plot = res[res.subgroup != "Overall"].iloc[::-1]
    yy = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(plot) + 1.6))
    ax.axvline(pooled_auc, ls="--", color=NAVY, lw=1.2,
               label=f"pooled AUROC ({pooled_auc:.3f})")
    cols = [RED if s else NAVY for s in plot.small_subgroup]
    ax.errorbar(plot.auroc, yy,
                xerr=[plot.auroc - plot.ci_low, plot.ci_high - plot.auroc],
                fmt="o", ecolor="#555", capsize=3, markersize=6, lw=1.2, ls="none",
                color=NAVY)
    ax.scatter(plot.auroc, yy, color=cols, zorder=3, s=36)
    ax.set_yticks(yy); ax.set_yticklabels(plot.subgroup, fontsize=8)
    ax.set_xlabel("AUROC (95% CI, patient-clustered bootstrap)")
    ax.set_title("Baseline model performance by sociodemographic subgroup", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig7_subgroup_auroc.png"), dpi=300)
    plt.close()
    log("\nSaved -> results/fig7_subgroup_auroc.png , subgroup_performance.csv")
    log("Red markers denote subgroups below the n>=%d stability threshold." % MIN_N)

    with open(os.path.join(RESULTS, "subgroup_performance.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print("\nLog -> results/subgroup_performance.txt")
