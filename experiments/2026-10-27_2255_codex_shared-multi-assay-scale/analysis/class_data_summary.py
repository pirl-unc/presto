"""Describe cross-assay molecular overlap separately for each MHC class, TRAIN only."""

from collections import defaultdict
import json
import math
import numpy as np
from scipy.stats import spearmanr
from verify import F, PINS, UPLOAD, csv_write, write


def main():
    contexts = defaultdict(lambda: defaultdict(list))
    for source in PINS["manifests"]["multi-assay-s42"]["sources"]:
        with (UPLOAD / "inputs" / source["path"]).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["split"] != "train" or row["qualifier"] != "=":
                    continue
                mhc = row["mhc"][0]
                key = (row["peptide"], mhc["groove1"], mhc["groove2"])
                factor = {
                    "pM": 0.001,
                    "nM": 1,
                    "uM": 1e3,
                    "mM": 1e6,
                    "M": 1e9,
                    "min": 1,
                }[row["unit"]]
                contexts[(mhc["mhc_class"], row["endpoint"])][key].append(
                    math.log10(row["value"] * factor)
                )
    rows = []
    for cls in ("I", "II"):
        endpoints = sorted(e for c, e in contexts if c == cls)
        means = {
            e: {k: np.mean(v) for k, v in contexts[(cls, e)].items()} for e in endpoints
        }
        for a in endpoints:
            for b in endpoints:
                keys = sorted(set(means[a]) & set(means[b]))
                x = np.array([means[a][k] for k in keys])
                y = np.array([means[b][k] for k in keys])
                rho = (
                    float(spearmanr(x, y).statistic)
                    if len(x) >= 3 and np.ptp(x) and np.ptp(y)
                    else None
                )
                rows.append(
                    dict(
                        mhc_class=cls,
                        endpoint_a=a,
                        endpoint_b=b,
                        shared_exact_training_contexts=len(keys),
                        exact_training_contexts_a=len(means[a]),
                        exact_training_contexts_b=len(means[b]),
                        spearman_of_context_means=rho,
                    )
                )
    csv_write(F / "results/training-output-overlap-by-class.csv", rows)
    write(
        F / "results/training-output-overlap-by-class.json",
        dict(
            scope="TRAIN ONLY; exact pMHC match; means over independent observations. Assay conditions are not matched. Descriptive association, not endpoint equivalence or a predictive evaluation.",
            rows=rows,
        ),
    )
    for r in rows:
        if r["endpoint_a"] == "ic50" and r["endpoint_b"] != "ic50":
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
