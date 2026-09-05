# Population-Level Clinical Enrichment and 30-Day Readmission Prediction

Analysis code for the study *"Pre-Split Resampling, Not Encounter-Level Splitting, Drove
Performance Inflation in a Low-Encounter-Ratio Readmission Benchmark: Factorial
Decomposition, Leakage-Controlled Evaluation, and Critical Appraisal of the Diabetes
130-US Hospitals Dataset"*.

The code reproduces a leakage-controlled benchmark on the UCI Diabetes 130-US Hospitals
dataset, decomposes the contributions of encounter-level splitting and pre-split
resampling to reported performance inflation, and tests whether enriching the dataset with
NHANES population-level physiological proxies (computed as deterministic stratum means)
improves individual-level readmission prediction.

## Repository layout

```
code/                        analysis scripts (01_* through 16_*)
revision_analyses/           notebooks and outputs added during revision
DATA/                        raw and processed data (NOT tracked - see "Data" below)
results/                     generated tables and figures (NOT tracked)
README.md
README_revision_analyses.md  what the revision analyses show
requirements.txt
versions.txt
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

Exact versions used for the reported results are in `versions.txt` (xgboost 3.2.0,
scikit-learn 1.8.0, imbalanced-learn 0.14.1, GPU enabled).

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
| 2 | `code/03_model_search.py` | model comparison (Tables 5-6) |
| 3 | `code/04_verify_enrichment.py` | Table 7 |
| 4 | `code/06_leakage_demo.py` | Table 4 (leaky vs correct) |
| 5 | `code/07_reviewer_stats.py` | clustered CIs, calibration, fall-back count |
| 6 | `code/07b_calibration_and_shap.py` | calibration sensitivity |
| 7 | `code/07c_shap_fix.py` | Additional file 1 (SHAP) |
| 8 | `code/01b_build_datasets_weighted.py` | survey-weighted datasets |
| 9 | `code/08_weighted_sensitivity.py` | weighted sensitivity analysis |
| 10 | `code/09_delta_auc_clustered.py` | delta-AUROC + caches out-of-fold predictions |
| 11 | `code/10_f1_points.py` | Table 5 metrics |
| 12 | `code/11_make_paper_figures.py` | Figures 1-5 |
| 13 | `code/12_make_fig6.py` | Figure 6 |
| 14 | `code/13_leakage_factorial.py` | Figure 2b (leakage decomposition) |
| 15 | `code/14_subgroup_fairness.py` | Figure 7, subgroup table |
| 16 | `code/15_operating_characteristics.py` | Table 8, Figure 8 |
| 17 | `code/16_factorial_multiseed.py` | multi-seed factorial (Additional file 4) |

`code/02_train_evaluate_v2.py` and `code/05_make_figures.py` are retained for completeness
but are superseded by later scripts and are not part of the reported results. All
stochastic steps use a fixed random seed (42).

## Revision analyses

Analyses added in response to peer review are in `revision_analyses/`, with a summary of
what each shows in `README_revision_analyses.md`. They cover the multi-seed factorial
replication, calibration slope and calibration-in-the-large, a within-fold SMOTE
comparison, subgroup contrast tests, decision curve analysis, a first-encounter sensitivity
analysis, the decomposition separating training contamination from evaluation on synthetic
observations, and distributional (mean + SD) enrichment.

The notebooks are self-contained: each rebuilds the cohort from the public UCI dataset and
asserts that it matches the manuscript (98,490 encounters, 69,311 patients, 11,271 events,
155 encoded predictors) before computing anything. They are intended to be run in Google
Colab with a GPU runtime.

## License

MIT (see `LICENSE`).
