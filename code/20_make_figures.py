"""
STEP 20 - Figures 1 to 3 of the manuscript, drawn from the result files written by
scripts 16 to 19. Nothing is typed in by hand.

Usage:  python 20_make_figures.py <results_dir> <figs_dir>
"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#dcdbd6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
                     "figure.dpi": 100, "savefig.dpi": 300, "savefig.facecolor": "white"})


def save(fig, figs, name):
    for ext in ("png", "pdf", "tif"):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
        fig.savefig(os.path.join(figs, f"{name}.{ext}"), bbox_inches="tight", **kw)
    from PIL import Image
    im = Image.open(os.path.join(figs, f"{name}.tif")).convert("RGBA"); bg = Image.new("RGB", im.size, "white")
    bg.paste(im, mask=im.split()[3]); bg.save(os.path.join(figs, f"{name}.tif"), dpi=(300, 300), compression="tiff_lzw")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1 (schematic)
def fig1(figs):
    fig, ax = plt.subplots(figsize=(7.2, 4.4)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(-3, 64)

    def box(x, y, w, h, label, fc="white", ec=MUTED, fs=7.2, bold=False, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.9",
                                    fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                     lw=0.8, color=MUTED, shrinkA=0, shrinkB=0))

    def split_bar(x, y, w, h, n_real_w, title):
        """A fold drawn as real negatives | real positives | created positives."""
        ax.text(x, y + h + 1.0, title, fontsize=6.6, color=MUTED, ha="left", va="bottom")
        neg, pos, new = n_real_w
        ax.add_patch(plt.Rectangle((x, y), w * neg, h, fc="#e9e8e3", ec="white", lw=0.8))
        ax.add_patch(plt.Rectangle((x + w * neg, y), w * pos, h, fc=BLUE, ec="white", lw=0.8))
        if new > 0:
            ax.add_patch(plt.Rectangle((x + w * (neg + pos), y), w * new, h, fc=ORANGE, ec="white", lw=0.8))

    rows = [
        (45.5, "a", "Correct protocol", "split patients", None,
         (0.885, 0.115, 0.0), (0.885, 0.115, 0.0)),
        (28.5, "b", "Resampling before the split", "resample all records", "split",
         (0.50, 0.065, 0.435), (0.50, 0.065, 0.435)),
        (10.5, "c", "Resampling after the split,\nin both parts", "split patients", "resample each part",
         (0.50, 0.065, 0.435), (0.50, 0.065, 0.435)),
    ]
    for y, tag, name, step1, step2, tr, te in rows:
        ax.text(0.5, y + 8.2, tag, fontsize=10, fontweight="bold", color=INK, va="center")
        ax.text(4.0, y + 8.2, name, fontsize=7.6, fontweight="bold", color=INK, va="center")
        box(4, y, 12, 5.2, "all records")
        arrow(16.6, y + 2.6, 19.4, y + 2.6)
        box(20, y, 21, 5.2, step1)
        x = 41.6
        if step2:
            arrow(x, y + 2.6, x + 2.8, y + 2.6)
            box(x + 3.4, y, 20, 5.2, step2); x = x + 24.0
        arrow(x, y + 2.6, 69.4, y + 2.6)
        split_bar(70, y + 3.6, 21, 2.4, tr, "training folds")
        split_bar(70, y - 3.0, 9, 2.4, te, "test fold (scored)")
    # legend, one row above the panels
    lx = 4.0
    for c, lab, dx in [("#e9e8e3", "real negatives", 0), (BLUE, "real positives", 20),
                       (ORANGE, "positives created by resampling", 40)]:
        ax.add_patch(plt.Rectangle((lx + dx, 61.0), 2.2, 1.5, fc=c, ec=MUTED, lw=0.4))
        ax.text(lx + dx + 3.0, 61.75, lab, fontsize=6.8, va="center", color=INK)
    ax.text(4, 2.2, r"Pooled AUROC $=(1-\pi_{\mathrm{new}})\times$AUROC$_{\mathrm{real}}$"
                    r"$\;+\;\pi_{\mathrm{new}}\times$AUROC$_{\mathrm{new}}$",
            fontsize=8.6, color=INK, va="center")
    ax.text(4, -1.6, r"$\pi_{\mathrm{new}}$: share of scored positives that were created by resampling.   "
                     r"AUROC$_{\mathrm{real}}$: real positives against real negatives.   "
                     r"AUROC$_{\mathrm{new}}$: created positives against real negatives.",
            fontsize=6.4, color=MUTED, va="center")
    save(fig, figs, "Fig1")


# ---------------------------------------------------------------- Figure 2 (datasets)
DS_ORDER = ["diabetes130", "cdc_dhi", "actg175", "thoracic", "heart_failure", "ctg", "mammography"]
DS_LABEL = {"diabetes130": "Diabetes 130-US", "cdc_dhi": "CDC Diabetes\nHealth Indicators",
            "actg175": "ACTG 175", "thoracic": "Thoracic\nSurgery", "heart_failure": "Heart Failure",
            "ctg": "Cardio-\ntocography", "mammography": "Mammography"}


def all_datasets(res):
    a = pd.read_csv(os.path.join(res, "mechanism_primary.csv")); a["dataset"] = "diabetes130"
    b = pd.read_csv(os.path.join(res, "mechanism_postsplit.csv")); b["dataset"] = "diabetes130"
    c = pd.read_csv(os.path.join(res, "replication_datasets.csv"))
    d = pd.concat([a, b, c], ignore_index=True)
    return d[d.method != "postsplit_train_only"]


def fig2(res, figs):
    d = all_datasets(res)
    m = d.groupby(["dataset", "learner", "method"])[["auroc_pooled", "auroc_real", "auroc_new"]].mean()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
    for ax, learner, title in zip(axes, ("xgb", "lr"), ("Gradient-boosted trees (XGBoost)", "Logistic regression")):
        x = np.arange(len(DS_ORDER)); w = 0.36
        real = [m.loc[(ds, learner, "smote"), "auroc_real"] for ds in DS_ORDER]
        new = [m.loc[(ds, learner, "smote"), "auroc_new"] for ds in DS_ORDER]
        ref = [m.loc[(ds, learner, "none"), "auroc_pooled"] for ds in DS_ORDER]
        pooled = [m.loc[(ds, learner, "smote"), "auroc_pooled"] for ds in DS_ORDER]
        ax.bar(x - w / 2 - 0.015, real, w, color=BLUE, label="real positives (AUROC$_{real}$)", zorder=3)
        ax.bar(x + w / 2 + 0.015, new, w, color=ORANGE, label="created positives (AUROC$_{new}$)", zorder=3)
        for i in range(len(x)):
            ax.plot([x[i] - w - 0.06, x[i] + w + 0.06], [ref[i], ref[i]], color=INK, lw=1.2, zorder=4,
                    label="correct protocol" if i == 0 else None)
            ax.plot(x[i], pooled[i], marker="D", ms=4.5, mfc="white", mec=INK, mew=1.0, ls="none", zorder=5,
                    label="reported (pooled) AUROC" if i == 0 else None)
            ax.text(x[i] + w / 2 + 0.015, new[i] + 0.012, f"{new[i]:.2f}", ha="center", fontsize=6.4, color=INK)
        ax.set_ylim(0.4, 1.06); ax.set_yticks(np.arange(0.4, 1.01, 0.1))
        ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        ax.set_ylabel("AUROC"); ax.set_title(title, fontsize=8.5, loc="left", color=INK, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([DS_LABEL[k] for k in DS_ORDER], fontsize=7)
    axes[0].legend(ncol=4, fontsize=6.6, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.26),
                   handlelength=1.6, columnspacing=1.2)
    fig.tight_layout()
    save(fig, figs, "Fig2")


# ---------------------------------------------------------------- Figure 3 (simulation)
def fig3(res, figs):
    s = pd.read_csv(os.path.join(res, "simulation.csv"))
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.6), sharey=True)
    meth = [("none", "correct"), ("smote", "before\nsplit"), ("postsplit_both_separately", "after split,\nboth parts")]
    for r, learner in enumerate(("xgb", "lr")):
        for c, (kind, n) in enumerate([("continuous", 1000), ("continuous", 10000), ("binary", 1000), ("binary", 10000)]):
            ax = axes[r, c]; q = s[(s.learner == learner) & (s.kind == kind) & (s.n == n)]
            x = np.arange(len(meth)); w = 0.36
            for i, (mth, _) in enumerate(meth):
                g = q[q.method == mth]
                ax.bar(i - (w / 2 + 0.015 if mth != "none" else 0), g.auroc_real.mean(), w, color=BLUE, zorder=3,
                       yerr=g.auroc_real.std(), error_kw=dict(lw=0.6, ecolor=INK, capsize=1.5))
                if mth != "none":
                    ax.bar(i + w / 2 + 0.015, g.auroc_new.mean(), w, color=ORANGE, zorder=3,
                           yerr=g.auroc_new.std(), error_kw=dict(lw=0.6, ecolor=INK, capsize=1.5))
                    ax.plot(i, g.auroc_pooled.mean(), marker="D", ms=4, mfc="white", mec=INK, mew=0.9, zorder=5)
            ax.axhline(q.true_auroc.mean(), color=INK, lw=0.9, ls=(0, (4, 2)), zorder=2)
            ax.set_ylim(0.4, 1.04); ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
            ax.set_xticks(x); ax.set_xticklabels([m[1] for m in meth], fontsize=6)
            if r == 0:
                ax.set_title(f"{kind} predictors\nn = {n:,}", fontsize=7.5, color=INK)
            if c == 0:
                ax.set_ylabel(("XGBoost" if learner == "xgb" else "Logistic regression") + "\nAUROC", fontsize=7.5)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Patch(fc=BLUE, label="real positives (AUROC$_{real}$)"), Patch(fc=ORANGE, label="created positives (AUROC$_{new}$)"),
         Line2D([], [], marker="D", mfc="white", mec=INK, ls="none", ms=4, label="reported (pooled) AUROC"),
         Line2D([], [], color=INK, lw=0.9, ls=(0, (4, 2)), label="AUROC of the data-generating model")]
    fig.legend(handles=h, ncol=4, fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.035))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, figs, "Fig3")


if __name__ == "__main__":
    res, figs = sys.argv[1], sys.argv[2]; os.makedirs(figs, exist_ok=True)
    fig1(figs)
    if os.path.exists(os.path.join(res, "replication_datasets.csv")): fig2(res, figs)
    if os.path.exists(os.path.join(res, "simulation.csv")): fig3(res, figs)
    print("figures written to", figs)
