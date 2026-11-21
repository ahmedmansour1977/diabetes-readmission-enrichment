"""
STEP 17b - Is leakage needed at all? (Diabetes 130-US Hospitals)

Runs SMOTE strictly after the patient-grouped split, either in the training folds only
(correct use) or independently in the training folds and in the test folds. In the second
case no information crosses the validation boundary, yet synthetic positives are again
part of what is scored.

Usage:  python 17b_mechanism_postsplit.py <uci_baseline.csv> <out_dir> [seed ...]
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
    out_csv = os.path.join(out_dir, "mechanism_postsplit.csv")
    rows = pd.read_csv(out_csv).to_dict("records") if os.path.exists(out_csv) else []
    done = {(r["method"], r["learner"], r["seed"]) for r in rows}
    for seed in seeds:
        for learner in ("xgb", "lr"):
            for mode in ("train_only", "both_separately"):
                if ("postsplit_" + mode, learner, seed) in done: continue
                t = time.time()
                r = mech.run_postsplit(X, y, g, mode, learner, seed)
                r["seconds"] = round(time.time() - t, 1)
                rows.append(r); pd.DataFrame(rows).to_csv(out_csv, index=False)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
