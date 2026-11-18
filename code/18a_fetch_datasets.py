"""
STEP 18a - Download the six replication datasets from their public repositories.

Writes one CSV per dataset (features plus outcome columns, exactly as distributed) and a
small metadata file. Nothing is filtered or recoded here; that happens in step 18.

Needs:  pip install ucimlrepo scikit-learn pandas
Usage:  python 18a_fetch_datasets.py <data_dir>
"""
import sys, os, json
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_openml

UCI = [("cdc_dhi", 891), ("actg175", 890), ("thoracic", 277), ("ctg", 193), ("heart_failure", 519)]

if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); meta = {}
    for name, uci_id in UCI:
        d = fetch_ucirepo(id=uci_id)
        df = pd.concat([d.data.features, d.data.targets], axis=1)
        df.to_csv(os.path.join(out, name + ".csv"), index=False)
        meta[name] = dict(source="UCI Machine Learning Repository", id=uci_id, name=d.metadata.get("name"),
                          doi=d.metadata.get("dataset_doi"), rows=len(df), missing=int(df.isna().sum().sum()))
        print(name, df.shape)
    m = fetch_openml(data_id=310, as_frame=True, parser="auto")
    m.frame.to_csv(os.path.join(out, "mammography.csv"), index=False)
    meta["mammography"] = dict(source="OpenML", id=310, rows=len(m.frame), missing=int(m.frame.isna().sum().sum()))
    print("mammography", m.frame.shape)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
