# Revision analyses

Analyses added to the manuscript in response to peer review. These supplement the original
pipeline in `code/` (scripts 01 to 15) and replace none of it.

Everything here runs from the **public UCI Diabetes 130-US Hospitals dataset alone**, except
the enrichment comparisons, which additionally need the NHANES-derived files produced by
script 01.

## Files

| File | What it is |
|---|---|
| `16_factorial_multiseed.py` | Multi-seed factorial, for the local pipeline |
| `factorial_multiseed_colab.ipynb` | Same, self-contained for Colab |
| `calibration_slope_colab.ipynb` | Calibration slope and calibration-in-the-large |
| `subgroup_and_fifthcell_colab.ipynb` | Within-fold SMOTE cell; subgroup contrast tests |
| `optional_analyses_colab.ipynb` | Decision curve; first-encounter set; SMOTE neighbourhood |
| `leakage_mechanism_colab.ipynb` | Contamination vs synthetic-test-set decomposition |
| `distributional_enrichment_fixed_colab.ipynb` | Mean vs mean+SD enrichment, cohort held fixed |
| `factorial_multiseed.txt` / `.csv` | 11-seed factorial output |
| `subgroup_and_fifthcell.txt`, `subgroup_contrasts.csv` | Within-fold SMOTE result; subgroup contrasts |
| `optional_analyses.txt` | Decision curve, first-encounter, SMOTE k |
| `leakage_mechanism.txt` | Contamination decomposition |
| `distributional_enrichment_fixed.txt` | Distributional enrichment |
| `fig9_decision_curve.png` | Figure 9 |

## Headline results

**Multi-seed factorial (11 seeds).** Main effect of pre-split resampling +0.2860 (SD 0.0006);
encounter-level splitting -0.0001 (SD 0.0002). Ordering identical in every seed.

**Mechanism.** Rescoring the contaminated model on the real encounters alone gives AUROC
0.668 against 0.670 for the correct protocol. Training contamination conferred no advantage
on real patients; essentially the entire +0.286 comes from scoring on synthetic rows.

**Resampling configurations.** Class weighting 0.670; SMOTE within training folds 0.655;
SMOTE before the split 0.956. Inflation is +0.2862 to +0.2869 for 1, 5 and 15 neighbours.

**Calibration.** Slope 1.026 (95% CI 0.993-1.062); calibration-in-the-large -2.004
(-2.025 to -1.982), approximately the log of the positive-class weight. Platt scaling
restores slope 0.993 and CITL 0.000.

**Subgroups.** All three age contrasts significant (Holm-adjusted P=.003). Hispanic
encounters differ from Caucasian (P=.02) and African American (P=.01). Sex not significant.

**Decision curve.** Net benefit exceeds flag-all and flag-none from threshold 0.010 to 0.505;
16.4 contacts avoided per 100 discharges at a threshold of 0.10.

**First encounters only.** 69,311 encounters, 6257 events, prevalence 9.0%, AUROC 0.652.

**Distributional enrichment.** On the identical cohort: SD only -0.0007, means only -0.0009,
means and SD together -0.0014 (P=.004). Dispersion adds nothing.

## Running the notebooks

Upload to Google Colab, set Runtime > Change runtime type > T4 GPU, then Run all. Each
notebook rebuilds the cohort from the public dataset and asserts it matches the manuscript
(98,490 encounters, 69,311 patients, 11,271 events, 155 encoded predictors) before computing
anything.

## Environment

Produced with xgboost 3.2.0, scikit-learn 1.8.0 and imbalanced-learn 0.14.1 on a GPU,
matching `versions.txt`. The Brier score and calibration error reproduced the primary values
exactly; the corresponding AUROC differed by about 0.0011, which is noted in the manuscript
wherever these results appear. Expect differences of that order between GPU models and
package builds.
