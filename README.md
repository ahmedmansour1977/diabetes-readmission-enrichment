# Oversampling before the split: two routes to inflated performance

Analysis code for the study *"Oversampling before data splitting inflates clinical prediction performance through two separable routes: information leakage and recognisable synthetic test records"* (submitted).

When SMOTE is applied to a whole dataset before it is split into training and test folds, reported
performance is inflated. This repository separates the two routes by which that happens:
**information leakage** into training, and **synthetic records in the test set** that the model learns
to recognise. The reported AUROC is split exactly into the part earned on real records and the part
earned on created ones:

    AUROC_pooled = (1 - pi_new) * AUROC_real + pi_new * AUROC_new

**Headline result** (Diabetes 130-US Hospitals, 98,490 encounters, gradient-boosted trees): SMOTE
before the split raised the AUROC from 0.670 to 0.957. On real held-out patients the
contaminated model scored 0.668; on synthetic records it scored 1.000. Applying SMOTE
separately to the training and test folds, so that nothing could leak, still gave 0.957.
Splitting by encounter instead of by patient changed the AUROC by less than 0.001. Six further public
clinical datasets and a simulation show when each route dominates.

## Repository layout

```
code/
  01_* to 15_*                  original pipeline (cohort, benchmark, enrichment, factorial)
  16_factorial_multiseed.py     2x2 factorial across 11 seeds
  mech.py, 17_* to 20_*         decomposition, probes, replication, simulation, figures
  README_scripts_17-20.md       what scripts 17 to 20 do and how to run them
notebooks/
  verify_scirep_colab.ipynb     one-click Colab run of scripts 17-19 with a check of every value
  *_colab.ipynb                 self-contained notebooks for the earlier revision analyses
outputs/
  scirep/                       results reported in the manuscript (scripts 17-20)
  scirep_reproduction/          independent second run, and the comparison of the two
  revision_analyses/            outputs of the earlier notebooks, with their own README
DATA/                           raw and processed data (not tracked)
results/                        run-time output (not tracked)
```

## Data

All data are public and are **not** redistributed here.

- Diabetes 130-US Hospitals (1999-2008), UCI Machine Learning Repository: https://doi.org/10.24432/C5230J
- Six replication datasets (CDC Diabetes Health Indicators, AIDS Clinical Trials Group Study 175,
  Thoracic Surgery, Heart Failure Clinical Records, Cardiotocography, Mammography), downloaded
  automatically by `code/18a_fetch_datasets.py` from UCI and OpenML
- NHANES (1999-2010), CDC/NCHS, used only by the earlier enrichment analyses: https://www.cdc.gov/nchs/nhanes/

Place the raw UCI files under `DATA/` (paths at the top of `code/01_build_datasets_v2.py`).

## Reproducing the manuscript

```
pip install -r requirements.txt
python code/01_build_datasets_v2.py          # writes DATA/processed/uci_baseline.csv
python code/16_factorial_multiseed.py 1 2 3 4 5 6 7 8 9 10 42
```

then scripts 17 to 20 as described in `code/README_scripts_17-20.md`. Or open
`notebooks/verify_scirep_colab.ipynb` in Google Colab, upload `uci_baseline.csv` and choose Run all.

| Results | Environment |
|---|---|
| Factorial (scripts 13, 16) | Google Colab GPU; xgboost 3.2.0, scikit-learn 1.8.0, imbalanced-learn 0.14.1 |
| Scripts 17-19 (`outputs/scirep/`) | Google Colab CPU; Python 3.13, xgboost 3.4.1, scikit-learn 1.6.1, imbalanced-learn 0.14.2 |
| Reproduction (`outputs/scirep_reproduction/`) | as above but scikit-learn 1.9.1 |

The two runs of scripts 17-19 agreed on 236 of 270 values to within 0.005 and on all values to
within 0.03; the larger differences are confined to ACTG 175, Heart Failure and Thoracic Surgery and to simulated
samples of 1,000 records.

## Earlier analyses

`code/01_*` to `code/15_*` and the notebooks other than `verify_scirep_colab.ipynb` belong to earlier
versions of the study, which also tested NHANES-based enrichment. They are kept unchanged as the
record of that work. `versions.txt` is a freeze of the local environment used for `code/`, and
`outputs/revision_analyses/README.md` summarises the revision analyses. All stochastic steps use
fixed random seeds.

### Run order (earlier analyses)

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
| 17 | `code/16_factorial_multiseed.py` | multi-seed factorial |

`code/02_train_evaluate_v2.py` and `code/05_make_figures.py` are retained for completeness but are
superseded by later scripts and are not part of the reported results. All stochastic steps use a
fixed random seed (42).

## License

MIT (see `LICENSE`).
