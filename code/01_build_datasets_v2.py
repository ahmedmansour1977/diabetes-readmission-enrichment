"""
STEP 1 (v2) - Build datasets with CLINICAL-STATE enrichment + a stronger feature set.
              VARIABLE-DRIVEN: add more NHANES proxies just by dropping the lab .xpt files
              into the cycle folders - this script auto-detects and uses whatever is present.

OPTION 1 (make enrichment able to add signal):
    The population proxy is conditioned on CLINICAL STATE, not demographics alone.
    NHANES strata = (age, sex, race, on_insulin).  on_insulin exists in BOTH datasets
    (NHANES DIQ050; UCI `insulin` != 'No'), so proxies vary with treatment state instead
    of being a pure function of age/sex/race (which the model already has).
    Small strata (n < MIN_STRATUM) back off to the demographic mean.

OPTION 2 (stronger, honest baseline):
    - KEEP A1Cresult and max_glu_serum (real glycaemic signal).
    - REMOVE expired / hospice encounters (cannot be readmitted -> label noise).

ENRICHMENT VARIABLES  (CANDIDATE_VARS below):  the script uses every candidate NHANES
variable that is PRESENT in your merged cycle files.  You already have BMI / BP / HbA1c.
To add clinical dimensions UCI lacks, download these NHANES lab components for EACH cycle
(1999-2010) and drop the .xpt into the matching cycle folder:
    * "Cholesterol - Total & HDL"        -> gives LBXTC (total chol), LBDHDD (HDL)
    * "Standard Biochemistry Profile"    -> gives LBXSCR (serum creatinine)  [strongest add]
    (optional, fasting subsample only:) LBXGLU fasting glucose, LBXTR triglycerides, LBDLDL LDL
No code change needed - just add the files and re-run.

Outputs (DATA/processed):  uci_baseline.csv , uci_mean_enrichment.csv , uci_mean_clinical.csv

Run:  python 01_build_datasets_v2.py
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
os.makedirs(PROCESSED, exist_ok=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED)
MIN_STRATUM = 30
EXPIRED_CODES = {11, 13, 14, 19, 20, 21}

# Candidate NHANES proxy variables -> friendly name used in the output columns.
# The script keeps whichever of these are found in your merged NHANES data.
CANDIDATE_VARS = {
    "BMXBMI":        "BMI",           # already have (Body Measures, BMX)
    "Average_BPXSY": "SBP",           # already have (Blood Pressure, BPX)
    "LBXGH":         "HbA1c",         # already have (Glycohemoglobin, LAB10/GHB)
    # ---- strong adds, consistent variable names across 1999-2010 ----
    "LBXSCR":        "Creatinine",    # file: Standard Biochemistry Profile (LAB18 / L40 / BIOPRO)  <- strongest
    "LBXSAL":        "Albumin",       # file: Standard Biochemistry Profile (frailty / nutrition)
    "LBXTC":         "TotalChol",     # file: Cholesterol - Total & HDL (LAB13 / L13 / TCHOL)
    "LBXHGB":        "Hemoglobin",    # file: Complete Blood Count (LAB25 / CBC)  (anemia)
    "LBXCRP":        "CRP",           # file: C-Reactive Protein (LAB11 / CRP)   (inflammation)
    # ---- optional; HDL name changed across cycles, fasting labs are subsample-only ----
    "LBDHDD":        "HDL",           # Cholesterol - Total & HDL (2003+; 1999-2002 uses LBDHDL)
    "LBXGLU":        "FastGlucose",   # fasting subsample only
    "LBXTR":         "Triglyceride",  # fasting subsample only
    "LBDLDL":        "LDL",           # fasting subsample only
}

# ----------------------------------------------------------------------
def load_nhanes():
    frames = []
    for cycle_dir in sorted(glob.glob(os.path.join(NHANES, "*"))):
        if not os.path.isdir(cycle_dir):
            continue
        merged = None
        for xpt in sorted(glob.glob(os.path.join(cycle_dir, "*.xpt"))):
            df = pd.read_sas(xpt, format="xport")
            if "SEQN" not in df.columns:
                continue
            merged = df if merged is None else merged.merge(df, on="SEQN", how="outer")
        if merged is not None:
            frames.append(merged)
    nh = pd.concat(frames, ignore_index=True)

    bp_cols = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"] if c in nh.columns]
    if bp_cols:
        nh["Average_BPXSY"] = nh[bp_cols].mean(axis=1)

    if "DIQ050" not in nh.columns:
        nh["DIQ050"] = np.nan
    nh["on_insulin"] = (nh["DIQ050"] == 1).astype(int)

    present = [v for v in CANDIDATE_VARS if v in nh.columns]
    keep = ["RIAGENDR", "RIDAGEYR", "RIDRETH1", "DIQ010", "on_insulin"] + present
    nh = nh[[c for c in keep if c in nh.columns]].copy()

    # diabetic cohort
    nh = nh[(nh["DIQ010"] == 1) | (nh.get("LBXGH", pd.Series(np.nan, index=nh.index)) >= 6.5)]
    nh = nh.dropna(subset=["RIAGENDR", "RIDAGEYR", "RIDRETH1"])   # demographics required; labs may be partial

    nh = nh.rename(columns={"RIAGENDR": "gender", "RIDAGEYR": "age", "RIDRETH1": "race"})
    for c in ["gender", "age", "race"]:
        nh[c] = nh[c].astype(int)
    nh["age"]  = pd.cut(nh["age"], bins=[20, 40, 60, np.inf],
                        labels=["20-39", "40-59", ">=60"], right=False)
    nh["race"] = nh["race"].map({1: 2, 2: 2, 3: 3, 4: 4, 5: 5})
    nh = nh.dropna(subset=["age", "race"])
    nh["race"] = nh["race"].astype(int)
    return nh, present

def build_summaries(nh, VARS):
    """Demographic means, and clinical-state means with counts. NaNs are skipped per-variable."""
    demo = (nh.groupby(["age", "gender", "race"], observed=True)[VARS]
              .mean().reset_index().rename(columns={v: f"{v}_mean" for v in VARS}))
    gc = nh.groupby(["age", "gender", "race", "on_insulin"], observed=True)
    clin = gc[VARS].mean().reset_index().rename(columns={v: f"{v}_mean" for v in VARS})
    clin["n"] = gc.size().reset_index(name="n")["n"].values
    return demo, clin

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

def main():
    print("Loading NHANES ...")
    nh, VARS = load_nhanes()
    proxy_names = [CANDIDATE_VARS[v] for v in VARS]
    print(f"  Proxy variables detected ({len(VARS)}): {proxy_names}")
    demo, clin = build_summaries(nh, VARS)
    print(f"  NHANES diabetic cohort: {len(nh)} rows | {len(demo)} demographic strata | "
          f"{len(clin)} clinical-state strata")

    print("Loading UCI ...")
    uci = load_uci()
    print(f"  UCI encounters (after removing expired/hospice): {len(uci)} "
          f"({uci['patient_nbr'].nunique()} unique patients), pos rate {uci['readmitted'].mean():.3f}")

    baseline = uci.drop(columns=["on_insulin"])
    baseline.to_csv(os.path.join(PROCESSED, "uci_baseline.csv"), index=False)
    add_demographic(uci, demo, VARS).drop(columns=["on_insulin"]).to_csv(
        os.path.join(PROCESSED, "uci_mean_enrichment.csv"), index=False)
    add_clinical(uci, demo, clin, VARS).drop(columns=["on_insulin"]).to_csv(
        os.path.join(PROCESSED, "uci_mean_clinical.csv"), index=False)

    print("\nSaved to", PROCESSED, "with proxy columns:", [f"{v}_mean" for v in VARS])
    for f in ["uci_baseline.csv", "uci_mean_enrichment.csv", "uci_mean_clinical.csv"]:
        print("  -", f)

if __name__ == "__main__":
    main()
