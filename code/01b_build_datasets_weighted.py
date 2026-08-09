"""
STEP 1b - Build SURVEY-WEIGHTED enrichment datasets (Reviewer 1's methodological point).

WHAT THIS FIXES
    01_build_datasets_v2.py computes stratum means as UNWEIGHTED sample averages of NHANES
    respondents. NHANES is a complex, stratified, oversampled probability survey - raw
    averages are NOT nationally representative estimates. Reviewer 1 called this "a
    significant methodological flaw, not merely a limitation."

WHAT THIS DOES
    Recomputes exactly the same stratum means using NHANES MEC examination weights,
    combined correctly across cycles per the CDC NHANES Analytic Guidelines:

        combined weight = original weight x (years covered by that weight / total years)

        1999-2000 and 2001-2002 : WTMEC4YR x (4 / total_years)
                                  (CDC publishes 4-year weights for this period and
                                   advises against using the 2-year weights there)
        2003-2004 onward        : WTMEC2YR x (2 / total_years)

    All nine proxies (BMI, SBP, HbA1c, creatinine, albumin, total cholesterol, HDL,
    haemoglobin, CRP) come from the MEC examination or its associated lab components,
    so MEC weights are the correct choice. (Fasting-subsample labs would need WTSAF2YR
    instead - none of the nine are fasting-subsample measures.)

    Everything else - cohort definition, strata, MIN_STRATUM=30 fall-back, UCI
    preprocessing, seeds - is IDENTICAL to 01_build_datasets_v2.py.

OUTPUTS (new filenames - your existing files are NOT overwritten)
    DATA/processed/uci_mean_enrichment_wtd.csv
    DATA/processed/uci_mean_clinical_wtd.csv
    results/weighting_report.txt      <- weighted vs unweighted stratum-mean comparison

Run:  python 01b_build_datasets_weighted.py
"""

import os, glob, random
import numpy as np
import pandas as pd

BASE      = os.environ.get(
    "ENRICHMENT_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA      = os.path.join(BASE, "DATA")
NHANES    = os.path.join(DATA, "NHANES")
UCI_CSV   = os.path.join(DATA, "diabetes+130-us+hospitals+for+years+1999-2008", "diabetic_data.csv")
PROCESSED = os.path.join(DATA, "processed")
RESULTS   = os.path.join(BASE, "results")
os.makedirs(PROCESSED, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED)
MIN_STRATUM = 30
EXPIRED_CODES = {11, 13, 14, 19, 20, 21}

CANDIDATE_VARS = {
    "BMXBMI":        "BMI",
    "Average_BPXSY": "SBP",
    "LBXGH":         "HbA1c",
    "LBXSCR":        "Creatinine",
    "LBXSAL":        "Albumin",
    "LBXTC":         "TotalChol",
    "LBXHGB":        "Hemoglobin",
    "LBXCRP":        "CRP",
    "LBDHDD":        "HDL",
    "LBXGLU":        "FastGlucose",
    "LBXTR":         "Triglyceride",
    "LBDLDL":        "LDL",
}

REPORT = []
def log(s):
    print(s); REPORT.append(s)

# ----------------------------------------------------------------------
# NHANES loading, now carrying the survey weight
# ----------------------------------------------------------------------
def _cycle_weight_plan(cycle_name, cols):
    """Decide which weight column to use for this cycle and how many years it covers."""
    is_early = ("1999" in cycle_name) or ("2001" in cycle_name)
    if is_early and "WTMEC4YR" in cols:
        return "WTMEC4YR", 4
    if is_early and "WTMEC4YR" not in cols:
        log(f"    ! {cycle_name}: WTMEC4YR not found; falling back to WTMEC2YR "
            f"(CDC advises 4-year weights for 1999-2002 - check your DEMO file)")
    return ("WTMEC2YR", 2) if "WTMEC2YR" in cols else (None, 0)

def load_nhanes_weighted():
    frames, plans = [], []
    for cycle_dir in sorted(glob.glob(os.path.join(NHANES, "*"))):
        if not os.path.isdir(cycle_dir):
            continue
        cycle_name = os.path.basename(cycle_dir)
        merged = None
        for xpt in sorted(glob.glob(os.path.join(cycle_dir, "*.xpt"))):
            df = pd.read_sas(xpt, format="xport")
            if "SEQN" not in df.columns:
                continue
            merged = df if merged is None else merged.merge(df, on="SEQN", how="outer")
        if merged is None:
            continue
        wcol, span = _cycle_weight_plan(cycle_name, merged.columns)
        if wcol is None:
            log(f"    ! {cycle_name}: NO MEC weight column found - cycle EXCLUDED from weighted analysis")
            continue
        merged["_raw_w"] = merged[wcol].astype(float)
        merged["_span"] = span
        merged["_cycle"] = cycle_name
        log(f"    {cycle_name}: weight = {wcol} ({span}-year), n = {len(merged)}")
        frames.append(merged); plans.append((cycle_name, wcol, span))

    nh = pd.concat(frames, ignore_index=True)

    # total years covered: the 4-year block counts ONCE, each 2-year cycle counts 2
    spans = {c: s for c, _, s in plans}
    total_years = (4 if 4 in spans.values() else 0) + 2 * sum(1 for s in spans.values() if s == 2)
    log(f"    Combined period = {total_years} years -> multiplier = span / {total_years}")
    nh["MEC_W"] = nh["_raw_w"] * (nh["_span"] / total_years)

    bp_cols = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"] if c in nh.columns]
    if bp_cols:
        nh["Average_BPXSY"] = nh[bp_cols].mean(axis=1)

    if "DIQ050" not in nh.columns:
        nh["DIQ050"] = np.nan
    nh["on_insulin"] = (nh["DIQ050"] == 1).astype(int)

    present = [v for v in CANDIDATE_VARS if v in nh.columns]
    keep = ["RIAGENDR", "RIDAGEYR", "RIDRETH1", "DIQ010", "on_insulin", "MEC_W"] + present
    nh = nh[[c for c in keep if c in nh.columns]].copy()

    nh = nh[(nh["DIQ010"] == 1) | (nh.get("LBXGH", pd.Series(np.nan, index=nh.index)) >= 6.5)]
    nh = nh.dropna(subset=["RIAGENDR", "RIDAGEYR", "RIDRETH1"])

    nh = nh.rename(columns={"RIAGENDR": "gender", "RIDAGEYR": "age", "RIDRETH1": "race"})
    for c in ["gender", "age", "race"]:
        nh[c] = nh[c].astype(int)
    nh["age"]  = pd.cut(nh["age"], bins=[20, 40, 60, np.inf],
                        labels=["20-39", "40-59", ">=60"], right=False)
    nh["race"] = nh["race"].map({1: 2, 2: 2, 3: 3, 4: 4, 5: 5})
    nh = nh.dropna(subset=["age", "race"])
    nh["race"] = nh["race"].astype(int)
    nh["age"] = nh["age"].astype(str)

    # respondents with no usable weight cannot contribute to weighted estimates
    bad_w = nh["MEC_W"].isna() | (nh["MEC_W"] <= 0)
    if bad_w.any():
        log(f"    {int(bad_w.sum())} respondents had zero/missing MEC weight (excluded from weighted means)")
    return nh, present

# ----------------------------------------------------------------------
def wmean(sub, var, wcol="MEC_W"):
    m = sub[var].notna() & sub[wcol].notna() & (sub[wcol] > 0)
    if m.sum() == 0:
        return np.nan
    return float(np.average(sub.loc[m, var], weights=sub.loc[m, wcol]))

def kish_neff(w):
    w = np.asarray(w, dtype=float); w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0: return 0.0
    return float(w.sum() ** 2 / np.sum(w ** 2))

def summaries(nh, VARS, weighted):
    """Return (demographic, clinical-state) stratum-mean tables."""
    def build(keys):
        rows = []
        for kv, sub in nh.groupby(keys, observed=True):
            kv = kv if isinstance(kv, tuple) else (kv,)
            d = dict(zip(keys, kv))
            for v in VARS:
                d[f"{v}_mean"] = wmean(sub, v) if weighted else sub[v].mean()
            d["n"] = len(sub)
            d["neff"] = kish_neff(sub["MEC_W"].values)
            rows.append(d)
        return pd.DataFrame(rows)
    return build(["age", "gender", "race"]), build(["age", "gender", "race", "on_insulin"])

# ----------------------------------------------------------------------
DISEASE_RANGES = {
    "has_diabetes_dx":    [(250, 250.99)], "has_circulatory_dx": [(390, 459)],
    "has_respiratory_dx": [(460, 519)],    "has_renal_dx":       [(580, 629)],
    "has_digestive_dx":   [(520, 579)],    "has_infectious_dx":  [(1, 139)],
    "has_injury_dx":      [(800, 999)],    "has_neoplasm_dx":    [(140, 239)],
    "has_symptoms_dx":    [(780, 799)],
}
DROP_COLS = ["encounter_id", "weight", "payer_code", "medical_specialty"]

def load_uci():
    """VERBATIM copy of load_uci() from 01_build_datasets_v2.py - nothing changed."""
    u = pd.read_csv(UCI_CSV).replace("?", np.nan)
    if "discharge_disposition_id" in u.columns:
        u = u[~u["discharge_disposition_id"].isin(EXPIRED_CODES)].copy()
    for c in ["diag_1", "diag_2", "diag_3"]:
        u[c] = u[c].astype(str).str.replace("V|E", "10", regex=True)
        u[c] = pd.to_numeric(u[c], errors="coerce")
    for dis, ranges in DISEASE_RANGES.items():
        mask = False
        for lo, hi in ranges:
            m = u[["diag_1", "diag_2", "diag_3"]].apply(lambda col: col.between(lo, hi)).any(axis=1)
            mask = mask | m
        u[dis] = mask.astype(int)
    MED_COLS = ["metformin","repaglinide","nateglinide","chlorpropamide","glimepiride",
                "acetohexamide","glipizide","glyburide","tolbutamide","pioglitazone",
                "rosiglitazone","acarbose","miglitol","troglitazone","tolazamide","examide",
                "citoglipton","insulin","glyburide-metformin","glipizide-metformin",
                "glimepiride-pioglitazone","metformin-rosiglitazone","metformin-pioglitazone"]
    med_present = [c for c in MED_COLS if c in u.columns]
    u["med_change_count"]  = u[med_present].isin(["Up", "Down"]).sum(axis=1)
    u["comorbidity_count"] = u[list(DISEASE_RANGES.keys())].sum(axis=1)
    u["total_prior_visits"] = (u["number_inpatient"].fillna(0) +
                               u["number_emergency"].fillna(0) +
                               u["number_outpatient"].fillna(0))
    u["on_insulin"] = (u["insulin"].astype(str) != "No").astype(int)
    u = u.drop(columns=["diag_1", "diag_2", "diag_3"])
    amap = {"[0-10)": None, "[10-20)": None, "[20-30)": "20-39", "[30-40)": "20-39",
            "[40-50)": "40-59", "[50-60)": "40-59", "[60-70)": ">=60", "[70-80)": ">=60",
            "[80-90)": ">=60", "[90-100)": ">=60"}
    u["age"] = u["age"].map(amap); u = u.dropna(subset=["age"])
    u = u[u["gender"].isin(["Male", "Female"])]
    u["gender"] = u["gender"].map({"Male": 1, "Female": 2}).astype(int)
    u["race"] = u["race"].map({"Caucasian": 3, "AfricanAmerican": 4, "Hispanic": 2,
                               "Asian": 5, "Other": 5}).fillna(5).astype(int)
    u["readmitted"] = (u["readmitted"].astype(str) == "<30").astype(int)
    u = u.drop(columns=[c for c in DROP_COLS if c in u.columns])
    u["patient_nbr"] = u["patient_nbr"].astype(int)
    return u

# ----------------------------------------------------------------------
def add_demographic(uci, demo, VARS):
    key = ["age", "gender", "race"]; means = [f"{v}_mean" for v in VARS]
    m = uci.merge(demo[key + means], on=key, how="left")
    for c in means: m[c] = m[c].fillna(m[c].mean())
    return m

def add_clinical(uci, demo, clin, VARS):
    key4, key3 = ["age", "gender", "race", "on_insulin"], ["age", "gender", "race"]
    means = [f"{v}_mean" for v in VARS]
    m = uci.merge(clin[key4 + means + ["n"]], on=key4, how="left")
    small = m["n"].isna() | (m["n"] < MIN_STRATUM)
    for c in means: m.loc[small, c] = np.nan
    demo_r = demo[key3 + means].rename(columns={c: c + "_demo" for c in means})
    m = m.merge(demo_r, on=key3, how="left")
    for c in means: m[c] = m[c].fillna(m[c + "_demo"]).fillna(m[c].mean())
    return m.drop(columns=["n"] + [c + "_demo" for c in means])

# ======================================================================
def main():
    log("=" * 72)
    log("SURVEY-WEIGHTED ENRICHMENT BUILD (NHANES MEC weights)")
    log("=" * 72)
    log("\nLoading NHANES cycles ...")
    nh, VARS = load_nhanes_weighted()
    log(f"\n  Proxy variables detected ({len(VARS)}): {[CANDIDATE_VARS[v] for v in VARS]}")
    log(f"  NHANES diabetic cohort: {len(nh)} respondents")
    log(f"  Sum of combined MEC weights = {nh['MEC_W'].sum():,.0f} "
        f"(approx. US diabetic adults represented)")
    log(f"  Kish effective sample size  = {kish_neff(nh['MEC_W'].values):,.0f} "
        f"of {len(nh)} actual respondents")

    demo_w, clin_w = summaries(nh, VARS, weighted=True)
    demo_u, clin_u = summaries(nh, VARS, weighted=False)
    log(f"  Strata: {len(demo_w)} demographic | {len(clin_w)} clinical-state")

    # ---- how much did weighting actually move the means? ----
    log("\nWEIGHTED vs UNWEIGHTED stratum means (clinical-state strata):")
    log(f"  {'variable':<16s} {'mean|diff|':>12s} {'max|diff|':>12s} {'rel.%':>8s}")
    diffs = []
    for v in VARS:
        c = f"{v}_mean"
        a = clin_w[c].values.astype(float); b = clin_u[c].values.astype(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() == 0: continue
        d = np.abs(a[ok] - b[ok]); scale = np.nanmean(b[ok])
        log(f"  {CANDIDATE_VARS[v]:<16s} {d.mean():12.4f} {d.max():12.4f} "
            f"{100*d.mean()/abs(scale) if scale else float('nan'):8.2f}")
        diffs.append(100 * d.mean() / abs(scale) if scale else np.nan)
    log(f"  -> average relative shift across all proxies: {np.nanmean(diffs):.2f}%")

    log("\nLoading UCI ...")
    uci = load_uci()
    log(f"  {len(uci)} encounters, {uci['patient_nbr'].nunique()} patients, "
        f"pos rate {uci['readmitted'].mean():.4f}")

    add_demographic(uci, demo_w, VARS).drop(columns=["on_insulin"]).to_csv(
        os.path.join(PROCESSED, "uci_mean_enrichment_wtd.csv"), index=False)
    add_clinical(uci, demo_w, clin_w, VARS).drop(columns=["on_insulin"]).to_csv(
        os.path.join(PROCESSED, "uci_mean_clinical_wtd.csv"), index=False)

    log("\nWrote:")
    log("  DATA/processed/uci_mean_enrichment_wtd.csv")
    log("  DATA/processed/uci_mean_clinical_wtd.csv")
    log("  (your original unweighted CSVs are untouched)")

    with open(os.path.join(RESULTS, "weighting_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT))
    print("\nReport -> results/weighting_report.txt")
    print("NEXT: python 08_weighted_sensitivity.py")

if __name__ == "__main__":
    main()
