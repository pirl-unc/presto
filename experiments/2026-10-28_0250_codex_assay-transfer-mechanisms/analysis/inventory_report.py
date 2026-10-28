"""Training-only composition report; never reads fitted predictions."""

from pathlib import Path
from collections import Counter
import csv
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

F = Path(__file__).resolve().parents[1]
x = json.loads((F / "results/original-training-inventory.json").read_text())
endpoints = ("ic50", "kd", "ec50", "kd_via_ic50", "kd_via_ec50", "half_life")
counts = Counter()
exact = Counter()
for row in x["by_endpoint_class_qualifier"]:
    key = (row["endpoint"], row["mhc_class"])
    counts[key] += row["n"]
    if row["qualifier"] == "=":
        exact[key] += row["n"]
rows = [
    dict(
        endpoint=e,
        mhc_class=c,
        observations=counts[e, c],
        exact=exact[e, c],
        bounds=counts[e, c] - exact[e, c],
    )
    for e in endpoints
    for c in ("I", "II")
]
gene_counts = Counter()
with (F / "results/original-training-alleles.csv").open() as stream:
    for row in csv.DictReader(stream):
        genes = {
            " / ".join(name.split("*")[0] for name in alias.split(" / "))
            for alias in row["catalog_names"].split("; ")
        }
        assert len(genes) == 1, (
            "cross-gene identical grooves require original row attribution"
        )
        gene = next(iter(genes))
        for endpoint in endpoints:
            gene_counts[(row["mhc_class"], gene, endpoint)] += int(row[endpoint])
for row in x["genes"]:
    assert (
        sum(gene_counts[row["mhc_class"], row["genes"], e] for e in endpoints)
        == row["n"]
    )
gene_rows = [
    dict(mhc_class=c, genes=g, **{e: gene_counts[c, g, e] for e in endpoints})
    for c, g in sorted({(c, g) for c, g, e in gene_counts})
]
with (F / "results/original-training-gene-endpoint.csv").open("w") as out:
    w = csv.DictWriter(out, fieldnames=list(gene_rows[0]))
    w.writeheader()
    w.writerows(gene_rows)
with (F / "results/original-training-composition.csv").open("w") as out:
    w = csv.DictWriter(out, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
fig, axs = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
labels = ("IC50", "KD", "EC50", "KD via IC50", "KD via EC50", "Half-life")
for cls, ax, color in zip(("I", "II"), axs, ("#21618c", "#a14628"), strict=True):
    a = np.array([exact[e, cls] for e in endpoints])
    b = np.array([counts[e, cls] for e in endpoints]) - a
    pos = np.arange(len(endpoints))
    ax.barh(pos, a, color=color, label="Exact values")
    ax.barh(pos, b, left=a, color=color, alpha=0.35, hatch="//", label="Bounds")
    for i, v in enumerate(a + b):
        ax.text(v + 400, i, f"{v:,}", va="center", fontsize=9)
    ax.set_yticks(pos, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 49000)
    ax.set(
        title=f"MHC-{cls}: {sum(a + b):,} observations",
        xlabel="Training observations (same scale)",
    )
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.15)
fig.suptitle(
    "Original six-output training mix: 82.1% MHC-I / 17.9% MHC-II\nAll human purified pMHC assays; no mass-spec or processing labels",
    fontsize=13,
)
fig.savefig(F / "results/original-training-composition.png", dpi=180)
fig.savefig(F / "results/original-training-composition.pdf")
plt.close(fig)
source = Counter()
for r in x["source_study"]:
    source[r["source"]] += r["n"]
print(
    json.dumps(
        dict(
            source_counts=source,
            study_source_groups=len(x["source_study"]),
            top_study_share=x["source_study"][0]["n"] / x["observations"],
            exact=sum(exact.values()),
            bounds=sum(counts.values()) - sum(exact.values()),
        ),
        indent=2,
    )
)
