"""Training-only endpoint, molecular input and matched-context descriptions."""

from pathlib import Path
from collections import Counter, defaultdict
import csv, hashlib, json, math
import numpy as np
from scipy.stats import spearmanr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

F = Path(__file__).resolve().parents[1]
ROOT = F.parents[1]
UPLOAD = ROOT / "artifacts" / F.name / "upload"


def main():
    pins = json.loads((F / "reproduce/source-pins.json").read_text())
    counts = Counter()
    genes = Counter()
    contexts = defaultdict(lambda: defaultdict(list))
    peptides = defaultdict(set)
    pairs = defaultdict(set)
    lengths = defaultdict(list)
    for spec in pins["input_sources"]:
        if "-train.jsonl" not in spec["path"]:
            continue
        for line in (UPLOAD / "inputs" / spec["path"]).open():
            r = json.loads(line)
            e = r["endpoint"]
            m = r["mhc"][0]
            c = m["mhc_class"]
            counts[e, c, r["qualifier"]] += 1
            peptides[e].add(r["peptide"])
            pairs[e].add((m["groove1"], m["groove2"]))
            lengths[e].append(len(r["peptide"]))
            genes[
                e, "/".join(x.split("*")[0] for x in m["provenance"]["catalog_alleles"])
            ] += 1
            if r["qualifier"] != "=":
                continue
            key = hashlib.sha256(
                "|".join((r["peptide"], m["groove1"], m["groove2"], c)).encode()
            ).hexdigest()
            factor = {"pM": 0.001, "nM": 1, "uM": 1e3, "mM": 1e6, "M": 1e9, "min": 1}[
                r["unit"]
            ]
            contexts[e][key].append(math.log10(r["value"] * factor))
    endpoints = sorted(contexts)
    means = {e: {k: np.mean(v) for k, v in d.items()} for e, d in contexts.items()}
    rows = []
    matrix = np.zeros((len(endpoints), len(endpoints)), int)
    for i, a in enumerate(endpoints):
        for j, b in enumerate(endpoints):
            shared = sorted(set(means[a]) & set(means[b]))
            matrix[i, j] = len(shared)
            rho = None
            if len(shared) >= 3:
                x = [means[a][k] for k in shared]
                y = [means[b][k] for k in shared]
                if np.ptp(x) > 0 and np.ptp(y) > 0:
                    rho = float(spearmanr(x, y).statistic)
            rows.append(
                dict(
                    endpoint_a=a,
                    endpoint_b=b,
                    shared_exact_training_contexts=len(shared),
                    spearman_of_context_means=rho,
                )
            )
    result = dict(
        scope="TRAIN ONLY; matched peptide and groove pair, not matched assay conditions; descriptive, not a performance metric",
        counts=[
            dict(endpoint=k[0], mhc_class=k[1], qualifier=k[2], n=v)
            for k, v in sorted(counts.items())
        ],
        genes=[
            dict(endpoint=k[0], restriction_genes=k[1], n=v)
            for k, v in sorted(genes.items())
        ],
        input_diversity={
            e: dict(
                peptides=len(peptides[e]),
                groove_pairs=len(pairs[e]),
                minimum_length=min(lengths[e]),
                maximum_length=max(lengths[e]),
                median_length=float(np.median(lengths[e])),
            )
            for e in endpoints
        },
        overlap=rows,
    )
    (F / "results/training-data-summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    with (F / "results/training-output-overlap.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(np.log10(1 + matrix), cmap="Blues")
    ax.set_xticks(range(len(endpoints)), endpoints, rotation=40, ha="right")
    ax.set_yticks(range(len(endpoints)), endpoints)
    for i in range(len(endpoints)):
        for j in range(len(endpoints)):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:,}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > 3000 else "#1c3647",
                fontsize=9,
            )
    ax.set_title(
        "Exact training pMHC contexts shared across outputs\nDistinct readouts; assay conditions are not matched"
    )
    fig.colorbar(im, ax=ax, label="log10(1 + shared contexts)")
    fig.tight_layout()
    fig.savefig(F / "results/training-output-overlap.png", dpi=170)
    fig.savefig(F / "results/training-output-overlap.pdf")
    plt.close(fig)
    print(json.dumps(result["input_diversity"], indent=2))


if __name__ == "__main__":
    main()
