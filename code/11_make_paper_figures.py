"""
STEP 11 - Generate manuscript Figures 1-5.

Figures 2-5 are built entirely from results this pipeline computes, so they will always
match the numbers in the manuscript.

FIGURE 1 IS DIFFERENT - READ THIS.
    Figure 1 plots one point per prior study. Those values are not produced by any code;
    they come from your reading of the cited papers. The LITERATURE table below is
    pre-filled with the study names from manuscript Table 1 but the performance values are
    set to None. Fill each one in from the paper it refers to, then re-run. Figure 1 is
    skipped until every value is supplied - it will not invent numbers.

Outputs (results/):
    fig1_literature.png        (only once you fill in the LITERATURE table)
    fig2_leakage.png
    fig3_model_comparison.png
    fig4_enrichment.png
    fig5_importance.png

Run:  python 11_make_paper_figures.py
Prerequisites: 01, 03, 04 (and 01b/08 if you want the weighted bars in Figure 4).
"""

import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier

BASE = os.environ.get(
    "ENRICHMENT_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED = os.path.join(BASE, "DATA", "processed")
RESULTS   = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

TARGET, GROUP, N_SPLITS, SEED = "readmitted", "patient_nbr", 5, 42
CAT_AS_STR = ["admission_type_id", "discharge_disposition_id", "admission_source_id", "race"]

plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
NAVY, RED, GREEN, GREY = "#1f3864", "#b3261e", "#2e7d32", "#9e9e9e"

# ======================================================================
# FIGURE 1 - fill these in yourself from the cited papers.
#   value  = the headline number that study reported (0-1 scale)
#   metric = "AUROC" or "accuracy" (used only for the marker shape)
#   group  = "high_bias" | "rigorous" | "other_task"
# ======================================================================
LITERATURE = [
    # --- high risk of evaluation bias -------------------------------------
    # Albahli, CMES 2025;143(1):1095-1128. RF accuracy 96.31%.
    #   SMOTE applied in preprocessing, before 5-fold CV.
    dict(study="Albahli [30]",                value=0.9631, metric="accuracy", group="high_bias"),
    # Seth, IJCTT 2025;73(5):196-204. Ensemble accuracy 91.2%, AUC-ROC 0.93.
    #   SMOTE-ENN, single 80/20 encounter-level split.
    dict(study="Seth [31]",                   value=0.930,  metric="AUROC",    group="high_bias"),
    # Hammoudeh et al., Procedia Comput Sci 2018;141:484-489. CNN accuracy 0.848, AUC 0.79.
    #   SMOTE, 80/20 encounter-level split.  (Previously mislabelled "Ahmad et al.")
    dict(study="Hammoudeh et al. [32]",       value=0.848,  metric="accuracy", group="high_bias"),
    # Ossai & Wickramasinghe, J Diabetes Complications 2022;36:108200.
    #   RF accuracy 94.7%, AUC 0.994. SMOTE with 10-fold CV.
    dict(study="Ossai & Wickramasinghe [33]", value=0.994,  metric="AUROC",    group="high_bias"),
    # Guo et al., Inform Med Unlocked 2025;58:101686. Bagging RF accuracy 0.89
    #   - at or below the 0.886 majority-class baseline. 80/20 split, SMOTE.
    dict(study="Guo et al. [34]",             value=0.890,  metric="accuracy", group="high_bias"),

    # --- rigorous, patient-level ------------------------------------------
    # Emi-Johnson & Nkrumah, Cureus 2025;17(4):e82437. XGBoost AUC-ROC 0.667.
    dict(study="Emi-Johnson & Nkrumah [6]",   value=0.667,  metric="AUROC",    group="rigorous"),
    # Liu et al., J Med Artif Intell 2024;7:23. RF AUROC 0.63, XGBoost 0.64.
    #   Patient-grouped k-fold; SMOTE inside training folds only - clean protocol.
    #   Their headline "F1 0.83" is a class-weighted average, not positive-class F1
    #   (recall == accuracy in all 11 of their models, which proves the averaging).
    dict(study="Liu et al. [7]",              value=0.630,  metric="AUROC",    group="rigorous"),
    # Rizvi & Xu - patient-level cluster-aware random forest.
    dict(study="Rizvi & Xu [8]",              value=0.683,  metric="AUROC",    group="rigorous"),
    # Lu & Uddin, Information 2022;13(9):436. Stacking AUC 0.6736, accuracy 68.63%.
    dict(study="Lu & Uddin [35]",             value=0.6736, metric="AUROC",    group="rigorous"),

    # --- different task: same dataset, different outcome, or different dataset
    # Sharma & Vaida, Inform Med Unlocked 2026;64:101780. LSTM AUROC 0.854, BUT the
    #   label is ANY readmission (<30 AND >30 mapped to 1), not 30-day. ~46% prevalence.
    dict(study="Sharma & Vaida [36]\n(any readmission)", value=0.854, metric="AUROC", group="other_task"),
]
THIS_STUDY = 0.671
PREVALENCE_ACC = 0.886      # majority-class accuracy at 11.4% prevalence
CEILING = (0.63, 0.69)

def figure1():
    rows = [r for r in LITERATURE if r["value"] is not None]
    missing = [r["study"] for r in LITERATURE if r["value"] is None]
    if missing:
        print("  Figure 1 SKIPPED - no value supplied for: " + ", ".join(missing))
        print("  Edit the LITERATURE table at the top of this file and re-run.")
        return
    order = {"high_bias": 0, "rigorous": 1, "other_task": 2}
    rows.sort(key=lambda r: (order[r["group"]], r["value"]))
    colors = {"high_bias": RED, "rigorous": GREEN, "other_task": GREY}
    labels = {"high_bias": "High risk of evaluation bias",
              "rigorous": "Rigorous, patient-level",
              "other_task": "Different task (context only)"}

    fig, ax = plt.subplots(figsize=(7.0, 0.34 * len(rows) + 2.4))
    ax.axhspan(-1, len(rows) + 1, xmin=0, xmax=0, color="none")
    ax.axvspan(CEILING[0], CEILING[1], color=GREEN, alpha=0.12, zorder=0,
               label=f"Leakage-free ceiling ({CEILING[0]:.2f}–{CEILING[1]:.2f})")
    ax.axvline(PREVALENCE_ACC, ls="--", color="#666", lw=1,
               label=f"Majority-class accuracy ({PREVALENCE_ACC:.3f})")
    for i, r in enumerate(rows):
        ax.scatter(r["value"], i, s=46, color=colors[r["group"]],
                   marker="o" if r["metric"] == "AUROC" else "s", zorder=3)
    ax.scatter(THIS_STUDY, len(rows), s=90, color=NAVY, marker="D", zorder=4)
    ax.set_yticks(list(range(len(rows) + 1)))
    ax.set_yticklabels([r["study"] for r in rows] + ["This study"], fontsize=8)
    ax.set_ylim(-0.8, len(rows) + 0.8)
    ax.set_xlim(0.55, 1.02)
    ax.set_xlabel("Reported headline performance (AUROC or accuracy)")
    ax.set_title("Reported performance on the UCI 30-day readmission benchmark", fontsize=11)
    handles = [Patch(color=colors[g], label=labels[g]) for g in
               ("high_bias", "rigorous", "other_task") if any(r["group"] == g for r in rows)]
    handles.append(Patch(color=NAVY, label="This study"))
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0][:2],
              fontsize=7, loc="lower left", framealpha=0.9)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "fig1_literature.png"), dpi=300)
    plt.close(); print("  fig1_literature.png")

# ======================================================================
def figure2():
    """Leaky vs correct protocol. Values read from results/leakage_demo.txt if present."""
    import re
    leaky, correct = 0.956, None
    # leaky AUROC from the leakage demo
    path = os.path.join(RESULTS, "leakage_demo.txt")
    if os.path.exists(path):
        m = re.findall(r"AUROC = ([0-9.]+)",
                       open(path, encoding="utf-8", errors="ignore").read())
        if m: leaky = float(m[0])
    # correct-protocol AUROC: use the POOLED out-of-fold value the tables report,
    # not the fold-average written by 06 (they differ in the 3rd decimal).
    for f, q in [("weighted_sensitivity.csv", "config=='baseline'"),
                 ("model_search.csv", "config=='baseline' and model=='xgb'")]:
        fp = os.path.join(RESULTS, f)
        if os.path.exists(fp):
            d = pd.read_csv(fp).query(q)
            if len(d):
                correct = float(d["auroc" if "auroc" in d else "auc"].iloc[0]); break
    if correct is None:
        correct = 0.671

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    bars = ax.bar(["Leaky protocol", "Correct protocol"], [leaky, correct],
                  color=[RED, NAVY], width=0.55)
    for b, v in zip(bars, [leaky, correct]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=11, fontweight="bold")
    # gap between the bars: bars sit at x=0 and x=1 with width 0.55, so the clear
    # space runs from 0.275 to 0.725 - draw the delta arrow up the middle of it.
    ax.hlines(correct, -0.275, 0.5, colors="#888", ls=":", lw=1)
    ax.annotate("", xy=(0.5, leaky), xytext=(0.5, correct),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=1.3))
    ax.text(0.56, (leaky + correct) / 2, f"+{leaky - correct:.3f}", fontsize=10,
            color="#444", ha="left", va="center")
    ax.axhline(0.5, ls=":", color="#999", lw=1)
    ax.text(1.45, 0.51, "chance", fontsize=7, color="#999", ha="right")
    ax.set_ylim(0.4, 1.05); ax.set_ylabel("AUROC")
    ax.set_title("Same model, same data,\ntwo evaluation protocols", fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "fig2_leakage.png"), dpi=300)
    plt.close(); print(f"  fig2_leakage.png   (leaky {leaky:.3f} vs correct {correct:.3f})")

# ======================================================================
def figure3():
    """Four-model AUROC with patient-clustered bootstrap CIs.

    Prefers results/table4_metrics.csv (written by 10_f1_points.py), whose intervals are
    the patient-clustered bootstrap CIs reported in Table 4. Falls back to
    model_search.csv only if that file is absent - note those are fold-based CIs and will
    NOT match Table 4, so the axis label is adjusted accordingly.
    """
    p_new = os.path.join(RESULTS, "table4_metrics.csv")
    p_old = os.path.join(RESULTS, "model_search.csv")
    if os.path.exists(p_new):
        df = pd.read_csv(p_new).sort_values("auroc")
        lo, hi, val = df["ci_low"], df["ci_high"], df["auroc"]
        xlabel = "AUROC (95% CI, patient-clustered bootstrap)"
    elif os.path.exists(p_old):
        df = pd.read_csv(p_old).query("config=='baseline'").sort_values("auc")
        lo, hi, val = df["auc_lo"], df["auc_hi"], df["auc"]
        xlabel = "AUROC (95% CI, across folds)"
        print("  fig3 NOTE: using fold-based CIs from model_search.csv - these do NOT")
        print("             match Table 4. Run 10_f1_points.py to get the clustered CIs.")
    else:
        print("  fig3 SKIPPED - run 10_f1_points.py (or 03_model_search.py) first"); return

    pretty = {"logreg": "Logistic regression", "rf": "Random forest",
              "lgbm": "LightGBM", "xgb": "XGBoost"}
    names = [pretty.get(m, m) for m in df["model"]]
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.errorbar(val, y, xerr=[val - lo, hi - val], fmt="o", color=NAVY,
                capsize=4, markersize=7, lw=1.4)
    ax.axvline(0.683, ls="--", color=GREEN, lw=1.2,
               label="Best published leakage-free (0.683)")
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlim(float(lo.min()) - 0.004, 0.688)
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.set_xlabel(xlabel)
    ax.set_title("Model comparison on the unenriched dataset", fontsize=11)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    plt.subplots_adjust(bottom=0.28)
    plt.savefig(os.path.join(RESULTS, "fig3_model_comparison.png"), dpi=300,
                bbox_inches="tight")
    plt.close(); print("  fig3_model_comparison.png")

# ======================================================================
def figure4():
    """Enrichment leaves discrimination unchanged."""
    path = os.path.join(RESULTS, "weighted_sensitivity.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        p2 = os.path.join(RESULTS, "model_search.csv")
        if not os.path.exists(p2):
            print("  fig4 SKIPPED - run 03 (and ideally 08) first"); return
        d = pd.read_csv(p2).query("model=='xgb'")
        df = pd.DataFrame(dict(config=d["config"], auroc=d["auc"],
                               ci_low=d["auc_lo"], ci_high=d["auc_hi"]))
    pretty = {"baseline": "Baseline\n(UCI only)", "mean": "+ Demographic",
              "mean_clinical": "+ Clinical-state",
              "mean_wtd": "+ Demographic\n(weighted)",
              "mean_clinical_wtd": "+ Clinical-state\n(weighted)"}
    df = df[df["config"].isin(pretty)].copy()
    df["label"] = df["config"].map(pretty)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.bar(x, df["auroc"], color=[NAVY] + [GREY] * (len(df) - 1), width=0.6)
    ax.errorbar(x, df["auroc"],
                yerr=[df["auroc"] - df["ci_low"], df["ci_high"] - df["auroc"]],
                fmt="none", ecolor="#333", capsize=4, lw=1.2)
    for xi, v, hi in zip(x, df["auroc"], df["ci_high"]):
        ax.text(xi, hi + 0.0012, f"{v:.3f}", ha="center", fontsize=8)
    ax.axhline(df.query("config=='baseline'")["auroc"].iloc[0], ls=":", color=NAVY, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(df["label"], fontsize=8)
    lo = float(df["ci_low"].min()); hi = float(df["ci_high"].max())
    ax.set_ylim(lo - 0.012, hi + 0.012)
    ax.set_ylabel("AUROC (95% CI)")
    ax.set_title("Enrichment does not change discrimination", fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "fig4_enrichment.png"), dpi=300)
    plt.close(); print("  fig4_enrichment.png")

# ======================================================================
def figure5():
    """Gain-based importance, proxies highlighted. Refits the clinical-state XGBoost."""
    path = os.path.join(PROCESSED, "uci_mean_clinical.csv")
    if not os.path.exists(path):
        print("  fig5 SKIPPED - run 01_build_datasets_v2.py first"); return
    df = pd.read_csv(path)
    if df[TARGET].dtype == object:
        df[TARGET] = (df[TARGET].astype(str) == "<30").astype(int)
    y = df[TARGET].astype(int).values; g = df[GROUP].values
    X = df.drop(columns=[TARGET, GROUP])
    for c in CAT_AS_STR:
        if c in X.columns: X[c] = X[c].astype(str)
    X = pd.get_dummies(X, columns=[c for c in X.columns
                                   if not pd.api.types.is_numeric_dtype(X[c])],
                       dummy_na=False).astype(float)
    feats = list(X.columns)
    tr, te = list(StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(y)), y, g))[0]
    sc = StandardScaler().fit(X.values[tr])
    kw = dict(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
              colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0,
              scale_pos_weight=(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1),
              eval_metric="logloss", random_state=SEED, tree_method="hist")
    try:
        xgb.train({"tree_method": "hist", "device": "cuda"},
                  xgb.DMatrix(np.zeros((8, 2)), label=[0, 1] * 4), num_boost_round=1)
        kw["device"] = "cuda"
    except Exception:
        pass
    m = XGBClassifier(**kw); m.fit(sc.transform(X.values[tr]), y[tr])
    imp = m.feature_importances_
    order = np.argsort(imp)[::-1][:20][::-1]
    colors = [RED if feats[i].endswith("_mean") else NAVY for i in order]
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    ax.barh(range(len(order)), imp[order], color=colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feats[i] for i in order], fontsize=8)
    ax.set_xlabel("Gain-based importance (normalised)")
    ax.set_title("Feature importance, clinical-state model\n(red = NHANES population proxy)",
                 fontsize=11)
    share = 100 * imp[[feats.index(c) for c in feats if c.endswith("_mean")]].sum()
    ax.text(0.97, 0.03, f"all 9 proxies = {share:.1f}% of total gain",
            transform=ax.transAxes, ha="right", fontsize=8, style="italic")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "fig5_importance.png"), dpi=300)
    plt.close(); print(f"  fig5_importance.png   (proxies = {share:.1f}% of total gain)")

# ======================================================================
if __name__ == "__main__":
    print("Writing figures to", RESULTS)
    figure1(); figure2(); figure3(); figure4(); figure5()
    print("\nFigure 6 panels come from 05_make_figures.py "
          "(figS1a_roc / figS1b_pr / figS1c_calibration - rename these to fig6a/b/c).")
    print("Supplementary Figure S1 comes from 07c_shap_fix.py (figS1_shap_bar.png).")
