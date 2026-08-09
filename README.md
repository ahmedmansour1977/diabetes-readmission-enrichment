# Population-Level Clinical Enrichment and 30-Day Readmission Prediction

Analysis code for the study *"Population-Level Enrichment Does Not Improve 30-Day
Readmission Prediction in Diabetes: A Leakage-Free Benchmark and Critical Appraisal"*
(submitted to *Diagnostics*, MDPI).

The code reproduces a leakage-free benchmark on the UCI Diabetes 130-US Hospitals
dataset, tests whether enriching it with NHANES population-level physiological proxies
(computed as deterministic stratum means) improves individual-level readmission
prediction, and quantifies the effect of common evaluation-leakage practices.

## Repository layout

```
code/              all analysis scripts (01_* through 15_*)
DATA/              raw and processed data (NOT tracked - see "Data" below)
results/           generated tables and figures (NOT tracked)
README.md
requirements.txt
.gitignore
```

`DATA/` and `results/` are created at run time and are excluded by `.gitignore`.

## Data

The datasets are public and are **not** redistributed in this repository:

- UCI Diabetes 130-US Hospitals (1999-2008): https://archive.ics.uci.edu/dataset/296
- NHANES (1999-2010), CDC/NCHS: https://www.cdc.gov/nchs/nhanes/

Download them and place the raw files under `DATA/` (see the paths at the top of
`code/01_build_datasets_v2.py`).

## Environment

Python 3.11. Install dependencies with:

```
pip install -r requirements.txt
```

Scripts locate the project root through the `ENRICHMENT_BASE` environment variable, and
otherwise fall back to the parent of the `code/` folder. Set it to the project root (the
folder that contains `code/` and `DATA/`). On Windows:

```
setx ENRICHMENT_BASE "path\to\project\root"
```

## Run order

| # | Script | Produces |
|---|--------|----------|
| 1 | `code/01_build_datasets_v2.py` | baseline + enrichment datasets |
| 2 | `code/03_model_search.py` | model comparison (Tables 4-5) |
| 3 | `code/04_verify_enrichment.py` | Table 6 |
| 4 | `code/06_leakage_demo.py` | Table 3 (leaky vs correct) |
| 5 | `code/07_reviewer_stats.py` | clustered CIs, calibration, fall-back count |
| 6 | `code/07b_calibration_and_shap.py` | calibration sensitivity |
| 7 | `code/07c_shap_fix.py` | Supplementary Figure S1 (SHAP) |
| 8 | `code/01b_build_datasets_weighted.py` | survey-weighted datasets |
| 9 | `code/08_weighted_sensitivity.py` | weighted sensitivity analysis |
| 10 | `code/09_delta_auc_clustered.py` | delta-AUROC + caches out-of-fold predictions |
| 11 | `code/10_f1_points.py` | Table 4 metrics |
| 12 | `code/11_make_paper_figures.py` | Figures 1-5 |
| 13 | `code/12_make_fig6.py` | Figure 6 |
| 14 | `code/13_leakage_factorial.py` | Figure 2b (leakage decomposition) |
| 15 | `code/14_subgroup_fairness.py` | Figure 7, subgroup table |
| 16 | `code/15_operating_characteristics.py` | Table 7, Figure 8 |

`code/02_train_evaluate_v2.py` and `code/05_make_figures.py` are retained for completeness
but are superseded by later scripts and are not part of the reported results. All
stochastic steps use a fixed random seed (42).

## License

MIT (see `LICENSE`).
