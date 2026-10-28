"""Descriptive measured-label overlap on new TRAIN only; no holdout access."""

from collections import defaultdict
from itertools import combinations
import json
import math
import numpy as np
from scipy.stats import spearmanr, pearsonr
from verify import F, PINS, UPLOAD, csv_write, write


def main():
    groups = defaultdict(lambda: defaultdict(list))
    measurement_ids = defaultdict(set)
    peptide_sets = defaultdict(set)
    rows = 0
    for source in PINS["manifests"]["wide-six-natural-s42"]["sources"]:
        # This family's frozen transport files are partitioned by split in the name.
        if "-train-measured-" not in source["path"]:
            continue
        for line in (UPLOAD / "inputs" / source["path"]).open():
            r = json.loads(line)
            assert r["split"] == "train"
            rows += 1
            m = r["mhc"][0]
            cls, endpoint = m["mhc_class"], r["endpoint"]
            measurement_ids[(cls, endpoint)].add(
                r["provenance"]["original_measurement_id"]
            )
            peptide_sets[(cls, endpoint)].add(r["peptide"])
            if r["qualifier"] != "=":
                continue
            assert r["unit"] in ("nM", "min")
            key = (r["peptide"], m["groove1"], m["groove2"])
            groups[(cls, endpoint)][key].append(math.log10(r["value"]))
    assert rows == 107132
    result = []
    for cls in ("I", "II"):
        for a, b in combinations(
            PINS["manifests"]["wide-six-natural-s42"]["model"]["endpoints"], 2
        ):
            left, right = groups[cls, a], groups[cls, b]
            keys = sorted(set(left) & set(right))
            x = np.array([np.mean(left[k]) for k in keys])
            y = np.array([np.mean(right[k]) for k in keys])
            supported = len(keys) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0
            result.append(
                dict(
                    mhc_class=cls,
                    endpoint_a=a,
                    endpoint_b=b,
                    shared_exact_pmhc_contexts=len(keys),
                    shared_peptides=len(peptide_sets[cls, a] & peptide_sets[cls, b]),
                    shared_measurement_ids=len(
                        measurement_ids[cls, a] & measurement_ids[cls, b]
                    ),
                    spearman=float(spearmanr(x, y).statistic) if supported else None,
                    pearson=float(pearsonr(x, y).statistic) if supported else None,
                )
            )
    csv_write(F / "results/training-assay-overlap.csv", result)
    write(
        F / "results/training-assay-overlap.json",
        dict(
            rows=result,
            scope="New TRAIN measured rows only. Average exact log targets per peptide/groove/endpoint. Different assays and conditions are not matched; observational agreement does not establish equivalence, causal transfer, or independence.",
        ),
    )
    print(json.dumps([r for r in result if r["endpoint_a"] == "ic50"], indent=2))


if __name__ == "__main__":
    main()
