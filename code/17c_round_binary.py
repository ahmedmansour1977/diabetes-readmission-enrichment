"""
STEP 17c - Which features give synthetic records away? (Diabetes 130-US Hospitals)
Pre-split SMOTE with ONLY the binary / one-hot indicator columns rounded back to 0 or 1;
count-valued columns keep their interpolated (fractional) values.
Usage:  python 17c_round_binary.py <uci_baseline.csv> <out_dir> [seed ...]
"""
import sys, os, time, importlib.util
import numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import mech
spec = importlib.util.spec_from_file_location("s17", os.path.join(here, "17_mechanism_primary.py"))
s17 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s17)
if __name__ == "__main__":
    path, out_dir = sys.argv[1], sys.argv[2]
    seeds = [int(s) for s in sys.argv[3:]] or [42, 1, 2]
    Xdf, y, g = s17.load_frame(path); X = Xdf.values.astype(float)
    print("binary columns:", int(mech.binary_columns(X).sum()), "of", X.shape[1])
    rows = []
    for seed in seeds:
        r = mech.run(X, y, g, "smote_round_bin", "xgb", seed, grouped=True); rows.append(r)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "mechanism_round_binary.csv"), index=False)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
