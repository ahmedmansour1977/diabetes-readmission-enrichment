"""
mech.py - shared functions for the mechanism analyses (scripts 17-19).

The model, encoding, scaling and fold construction follow 13_leakage_factorial.py and
16_factorial_multiseed.py. What is added here is book-keeping: every out-of-fold
prediction is tagged as belonging to a REAL record or to a record CREATED BY RESAMPLING,
so that the pooled AUROC of a pre-split-resampling pipeline can be split exactly into

    AUROC_pooled = pi_real * AUROC_real + pi_new * AUROC_new

where  AUROC_real = real positives      vs real negatives
       AUROC_new  = resampled positives vs real negatives
       pi_new     = share of positives in the evaluated set that were created by resampling

The identity is exact because AUROC is the probability that a randomly chosen positive is
ranked above a randomly chosen negative, and every positive is either real or created.
"""
import warnings
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
N_SPLITS = 5


def make_xgb(pos_weight, seed, n_jobs=2):
    """Hyperparameters copied verbatim from 13_leakage_factorial.py (CPU hist)."""
    return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
                         colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0,
                         scale_pos_weight=pos_weight, eval_metric="logloss",
                         random_state=seed, tree_method="hist", n_jobs=n_jobs)


def make_lr(balanced, seed):
    return LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                              class_weight="balanced" if balanced else None)


def integer_columns(X):
    """Columns in which every real record takes an integer value (binary, one-hot, counts)."""
    return np.all(np.mod(X, 1) == 0, axis=0)


def binary_columns(X):
    """Columns in which every real record is 0 or 1 (binary and one-hot indicators)."""
    return np.all((X == 0) | (X == 1), axis=0)


def resample(X, y, method, seed, k=5):
    """Return (Xr, yr, is_new). Real rows always come first, in their original order."""
    n = len(y)
    if method == "none":
        return X, y, np.zeros(n, bool)
    if method in ("smote", "smote_round", "smote_round_bin"):
        Xr, yr = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y)
        assert np.array_equal(Xr[:n], X) and np.array_equal(yr[:n], y)
        if method in ("smote_round", "smote_round_bin"):
            ic = integer_columns(X) if method == "smote_round" else binary_columns(X)
            Xr = Xr.copy()
            Xr[n:, ic] = np.rint(Xr[n:, ic])
    elif method == "ros":
        rng = np.random.RandomState(seed)
        pos = np.flatnonzero(y == 1)
        need = int((y == 0).sum() - (y == 1).sum())
        pick = rng.choice(pos, size=need, replace=True)
        Xr = np.vstack([X, X[pick]])
        yr = np.concatenate([y, np.ones(need, int)])
    else:
        raise ValueError(method)
    is_new = np.zeros(len(yr), bool)
    is_new[n:] = True
    return Xr, yr, is_new


def oof_predict(Xr, yr, gr, grouped, learner, balanced_by_weight, seed, n_jobs=2):
    if grouped:
        splits = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed).split(
            np.zeros(len(yr)), yr, gr)
    else:
        splits = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(
            np.zeros(len(yr)), yr)
    oof = np.zeros(len(yr))
    for tr, te in splits:
        sc = StandardScaler().fit(Xr[tr])
        if learner == "xgb":
            pw = (yr[tr] == 0).sum() / max((yr[tr] == 1).sum(), 1) if balanced_by_weight else 1.0
            # 13_leakage_factorial.py uses the full-sample ratio; the fold ratio is identical
            # to three decimals under stratification
            m = make_xgb(pw, seed, n_jobs)
        else:
            m = make_lr(balanced_by_weight, seed)
        m.fit(sc.transform(Xr[tr]), yr[tr])
        oof[te] = m.predict_proba(sc.transform(Xr[te]))[:, 1]
    return oof


def decompose(yr, oof, is_new):
    """Pooled AUROC and its exact split into real and resampled positives."""
    neg = yr == 0
    real_pos = (yr == 1) & ~is_new
    new_pos = (yr == 1) & is_new
    out = dict(auroc_pooled=roc_auc_score(yr, oof))
    m = neg | real_pos
    out["auroc_real"] = roc_auc_score(yr[m], oof[m])
    if new_pos.any():
        m2 = neg | new_pos
        out["auroc_new"] = roc_auc_score(yr[m2], oof[m2])
        out["pi_new"] = new_pos.sum() / (new_pos.sum() + real_pos.sum())
        out["identity"] = (1 - out["pi_new"]) * out["auroc_real"] + out["pi_new"] * out["auroc_new"]
    else:
        out["auroc_new"] = np.nan; out["pi_new"] = 0.0; out["identity"] = out["auroc_real"]
    return out


def run(X, y, g, method, learner, seed, grouped=True, k=5, n_jobs=2):
    """One pipeline. g may be None (no repeated records); new rows get unique group ids,
    exactly as in 13_leakage_factorial.py."""
    Xr, yr, is_new = resample(X, y, method, seed, k)
    if g is None:
        g = np.arange(len(y))
    g = np.asarray(g)
    _, g_int = np.unique(g, return_inverse=True)
    gr = np.concatenate([g_int, np.arange(g_int.max() + 1, g_int.max() + 1 + is_new.sum())])
    oof = oof_predict(Xr, yr, gr, grouped, learner, method == "none", seed, n_jobs)
    res = decompose(yr, oof, is_new)
    res.update(method=method, learner=learner, seed=seed, grouped=grouped,
               n_real=len(y), n_new=int(is_new.sum()), prevalence=float(y.mean()))
    return res


def off_support(X, y, seed, k=5):
    """How recognisable are SMOTE rows? Share with >=1 value no real record can take."""
    n = len(y)
    Xr, yr, is_new = resample(X, y, "smote", seed, k)
    ic = integer_columns(X)
    frac = np.mod(Xr[n:][:, ic], 1) != 0
    per_row = frac.sum(1)
    return dict(n_new=int(is_new.sum()), n_integer_cols=int(ic.sum()), n_cols=int(X.shape[1]),
                share_rows_off_support=float((per_row > 0).mean()),
                median_off_support_features=float(np.median(per_row)),
                share_real_rows_off_support=0.0)


def run_postsplit(X, y, g, mode, learner, seed, k=5, n_jobs=2):
    """Resampling AFTER the patient-grouped split, so that nothing crosses the boundary.

    mode = "train_only"       SMOTE inside each training fold; test fold left real
                              (the correct way to use SMOTE)
    mode = "both_separately"  SMOTE inside each training fold AND, independently, inside
                              each test fold. No test record, real or synthetic, shares a
                              parent with any training record, so there is no leakage in
                              the usual sense - but the evaluated set again contains
                              synthetic positives.
    """
    if g is None:
        g = np.arange(len(y))
    splits = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y, g)
    ys, ps, news = [], [], []
    for f, (tr, te) in enumerate(splits):
        Xtr, ytr = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X[tr], y[tr])
        if mode == "both_separately":
            Xte, yte = SMOTE(random_state=seed + 1000 + f, k_neighbors=k).fit_resample(X[te], y[te])
        else:
            Xte, yte = X[te], y[te]
        is_new = np.zeros(len(yte), bool); is_new[len(te):] = True
        sc = StandardScaler().fit(Xtr)
        m = make_xgb(1.0, seed, n_jobs) if learner == "xgb" else make_lr(False, seed)
        m.fit(sc.transform(Xtr), ytr)
        ys.append(yte); ps.append(m.predict_proba(sc.transform(Xte))[:, 1]); news.append(is_new)
    yr, oof, is_new = np.concatenate(ys), np.concatenate(ps), np.concatenate(news)
    res = decompose(yr, oof, is_new)
    res.update(method="postsplit_" + mode, learner=learner, seed=seed, grouped=True,
               n_real=len(y), n_new=int(is_new.sum()), prevalence=float(y.mean()))
    return res
