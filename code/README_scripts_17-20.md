# Scripts 17 to 20: where the inflation comes from

These scripts add the decomposition, the mechanism probes, the replication on six further
datasets, the simulation and the three figures of the manuscript. They do not change
scripts 01 to 16.

| Script | What it does | Output (in the output folder you name) |
|---|---|---|
| `mech.py` | Shared functions: resampling, fold construction, tagging of real and created records, exact split of the pooled AUROC | none |
| `17_mechanism_primary.py` | Primary dataset: correct protocol, SMOTE before the split, rounded SMOTE, random oversampling; two learners | `mechanism_primary.csv`, `off_support_primary.json` |
| `17b_mechanism_postsplit.py` | Primary dataset: SMOTE in the training folds only, and SMOTE separately in training and test folds | `mechanism_postsplit.csv` |
| `17c_round_binary.py` | Primary dataset: SMOTE before the split with only the binary indicators rounded | `mechanism_round_binary.csv` |
| `18a_fetch_datasets.py` | Downloads the six replication datasets (UCI, OpenML) | CSV files in the data folder |
| `18_replication_datasets.py` | Same pipelines on the six datasets | `replication_datasets.csv`, `replication_descriptives.json` |
| `19_simulation.py` | Simulation with known discrimination | `simulation.csv` |
| `20_make_figures.py` | Figures 1 to 3 | `Fig1` to `Fig3` (.png, .pdf, .tif) |

## How to run

```
pip install xgboost scikit-learn imbalanced-learn pandas numpy scipy matplotlib ucimlrepo

python 17_mechanism_primary.py   <path>/uci_baseline.csv  results  42 1 2
python 17b_mechanism_postsplit.py <path>/uci_baseline.csv  results  42 1 2
python 17c_round_binary.py        <path>/uci_baseline.csv  results  42 1 2
python 18a_fetch_datasets.py      data_ext
python 18_replication_datasets.py data_ext results
python 19_simulation.py           results 10
python 20_make_figures.py         results figs
```

`uci_baseline.csv` is the processed file written by `code/01_build_datasets_v2.py` to
`DATA/processed/` (98,490 encounters, 155 predictors after encoding). Script 17 stops with an assertion error if the cohort differs.

The quickest route is `notebooks/verify_scirep_colab.ipynb`: it runs scripts 17 to 19 in Google Colab
and compares every value with the manuscript.

Run time on two CPU cores is about 25 minutes for 17, 15 for 17b, 5 for 17c, 20 for 18 and
15 for 19. Every script writes its results after each run and resumes where it stopped.

## Environment of the committed results

`outputs/scirep/` holds the results reported in the manuscript. They were produced in Google Colab (CPU, `tree_method="hist"`) with Python 3.13, xgboost 3.4.1,
scikit-learn 1.6.1, imbalanced-learn 0.14.2, numpy 2.4.6 and pandas 2.2.3. A second run with
scikit-learn 1.9.1 (`outputs/scirep_reproduction/`) agreed on 236 of 270 values to within 0.005 and on all
values to within 0.03; the larger differences are confined to ACTG 175, Heart Failure and Thoracic
Surgery and to the 1,000-record simulations. Scripts 13 and 16 were run earlier on a GPU with xgboost 3.2.0; shared cells
agree to within 0.001 AUROC.

## The identity that everything rests on

After oversampling, every scored positive record is either real or created, and every
negative record is real. Therefore

    AUROC_pooled = (1 - pi_new) * AUROC_real + pi_new * AUROC_new

exactly, where `pi_new` is the share of scored positives that were created. The column
`identity` in each result file recomputes the right-hand side so that it can be compared
with `auroc_pooled`.
