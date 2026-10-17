# Leakage-Controlled Benchmark and Evaluation-Bias Decomposition, Diabetes 130-US Hospitals

Analysis code for the study *"Pre-split resampling inflates reported performance by replacing the
evaluation target rather than by contaminating training: a factorial decomposition of evaluation
bias on the Diabetes 130-US Hospitals readmission benchmark"* (under review).

The code reproduces a leakage-controlled benchmark for 30-day readmission on the UCI Diabetes
130-US Hospitals dataset, decomposes the contributions of encounter-level splitting and pre-split
resampling to reported performance inflation, separates training contamination from evaluation
against synthetic observations, and tests whether enriching the dataset with NHANES
population-level physiological proxies (computed as deterministic stratum means) improves
individual-level prediction.

Headline result: holding model, hyperparameters, feature matrix and random seed fixed, pre-split
resampling raised AUROC from 0.671 to 0.956 while encounter-level splitting alone changed it by
-0.0015. Rescoring the identical contaminated model on the real encounters alone recovered 0.668,
so essentially the whole effect came from grading the model on a half-synthetic test fold.

## Repository layout

```
code/                         analysis scripts (01_* through 15_*)
16_factorial_multiseed.py     multi-seed factorial, run alongside the code/ scripts
*_colab.ipynb                 self-contained notebooks for the revision analyses
*.txt, *.csv, *.png           outputs of those analyses
DATA/                         raw and processed data (NOT tracked - see "Data" below)
results/                      generated tables and figures (NOT tracked)
README.md
README_revision_analyses.md   what the revision analyses show
requirements.txt
versions.txt
.gitignore
```

`DATA/` and `results/` are created at run time and are excluded by `.gitignore`.

## Data

The datasets are public and are **not** redistributed in this repository:

- UCI Diabetes 130-US Hospitals (1999-2008): https://archive.ics.uci.edu/dataset/296
  (doi:10.24432/C5230J)
- NHANES (1999-2010), CDC/NCHS: https://www.cdc.gov/nchs/nhanes/

Download them and place the raw files under `DATA/` (see the paths at the top of
`code/01_build_datasets_v2.py`).

## Environment

Python 3.11. Install dependencies with:

```
pip install -r requirements.txt
```

Two environments were used, and they differ slightly. This is deliberate and is disclosed in the
manuscript wherever the affected quantities appear.

| | Primary analysis | Revision analyses |
|---|---|---|
| Where | Local workstation | Google Colab, T4 GPU |
| What | Everything under `code/` | The `*_colab.ipynb` notebooks |
| xgboost | 3.2.0 | 3.2.0 |
| scikit-learn | 1.7.2 | 1.8.0 |
| imbalanced-learn | 0.14.2 | 0.14.1 |

`versions.txt` is a full freeze of the **primary** environment. Three of the four factorial cells
reproduced exactly across the two environments and the fourth differed by 0.0011 AUROC; for the
calibration analyses the Brier score and calibration error reproduced exactly while the
corresponding AUROC differed by 0.0015. Expect differences of that order between GPU models and
package builds.

Scripts locate the project root through the `ENRICHMENT_BASE` environment variable, and otherwise
fall back to the parent of the `code/` folder. Set it to the project root (the folder that contains
`code/` and `DATA/`). On Windows:

```
setx ENRICHMENT_BASE "path\to\project\root"
```

## Run order

Outputs are described by what they contain rather than by table or figure number, so that this
table does not go stale as the manuscript is renumbered between submissions.

| # | Script | Produces |
|---|--------|----------|
| 1 | `code/01_build_datasets_v2.py` | baseline and enrichment datasets |
| 2 | `code/03_model_search.py` | four-model comparison; enrichment configurations |
| 3 | `code/04_verify_enrichment.py` | enrichment verification diagnostics |
| 4 | `code/06_leakage_demo.py` | leaky versus correct protocol |
| 5 | `code/07_reviewer_stats.py` | patient-clustered CIs, calibration, fall-back count |
| 6 | `code/07b_calibration_and_shap.py` | calibration sensitivity across four variants |
| 7 | `code/07c_shap_fix.py` | SHAP summary for the supporting information |
| 8 | `code/01b_build_datasets_weighted.py` | survey-weighted datasets |
| 9 | `code/08_weighted_sensitivity.py` | survey-weighted sensitivity analysis |
| 10 | `code/09_delta_auc_clustered.py` | delta-AUROC; caches out-of-fold predictions |
| 11 | `code/10_f1_points.py` | threshold-based metrics for the model comparison |
| 12 | `code/11_make_paper_figures.py` | literature appraisal, leakage, model comparison, enrichment, importance |
| 13 | `code/12_make_fig6.py` | ROC, precision-recall and calibration panels |
| 14 | `code/13_leakage_factorial.py` | 2x2 factorial decomposition |
| 15 | `code/14_subgroup_fairness.py` | subgroup discrimination and table |
| 16 | `code/15_operating_characteristics.py` | capacity-based operating characteristics |
| 17 | `16_factorial_multiseed.py` | multi-seed factorial |

`code/02_train_evaluate_v2.py` and `code/05_make_figures.py` are retained for completeness but are
superseded by later scripts and are not part of the reported results. All stochastic steps use a
fixed random seed (42).

## Revision analyses

Analyses added in response to peer review are in the repository root alongside the `code/` folder,
with a summary of what each shows in `README_revision_analyses.md`.

| Notebook | What it computes | Output |
|---|---|---|
| `factorial_multiseed_colab.ipynb` | 2x2 factorial across 11 random seeds | `factorial_multiseed.txt`, `.csv` |
| `leakage_mechanism_colab.ipynb` | Contamination versus evaluation on synthetic rows | `leakage_mechanism.txt` |
| `calibration_slope_colab.ipynb` | Calibration slope and calibration-in-the-large | `calibration_slope.txt`, `.csv` |
| `subgroup_and_fifthcell_colab.ipynb` | Within-fold SMOTE cell; subgroup contrasts | `subgroup_and_fifthcell.txt`, `subgroup_contrasts.csv` |
| `optional_analyses_colab.ipynb` | Decision curve; first-encounter set; SMOTE neighbourhood | `optional_analyses.txt`, `fig9_decision_curve.png` |
| `distributional_enrichment_fixed_colab.ipynb` | Mean versus mean + SD enrichment, cohort fixed | `distributional_enrichment_fixed.txt` |

The notebooks are self-contained: each rebuilds the cohort from the public UCI dataset and asserts
that it matches the manuscript (98,490 encounters, 69,311 patients, 11,271 events, 155 encoded
predictors) before computing anything. Run them in Google Colab with a GPU runtime.

## License

MIT (see `LICENSE`).
