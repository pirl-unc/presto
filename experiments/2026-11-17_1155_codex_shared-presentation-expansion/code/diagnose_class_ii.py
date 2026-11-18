"""Descriptive class-I/II slices, kept separate from checkpoint selection."""

import gzip
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

E = Path(__file__).resolve().parents[1]


def main():
    all_rows = []
    paths = list((E / "results/evaluation").glob("*/*-predictions.jsonl.gz"))
    paths += list((E / "results/fits").glob("*/raw/validation-predictions.jsonl.gz"))
    for path in paths:
        model = path.parents[1].name if path.parent.name == "raw" else path.parent.name
        if model.endswith(("-n8", "-n12", "-n16")):
            continue
        with gzip.open(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                if r["endpoint"] != "elution_observation":
                    continue
                assert set(r["mhc_classes"]) in ({"I"}, {"II"})
                cls = r["mhc_classes"][0]
                o = r["observation"]
                p = o["provenance"]
                c = o["context"]
                population = (
                    (
                        "expanded"
                        if p.get("source_family")
                        else "apc"
                        if r["source"] == "hitlist_native_ms"
                        else "broad"
                    )
                    if path.parent.name == "raw" or path.name.startswith("validation-")
                    else path.name.split("-")[0]
                )
                loci = set()
                for m in o["mhc"]:
                    for a in m.get("provenance", {}).get("catalog_alleles", []):
                        loci.update(
                            locus
                            for locus in (
                                ("DR", "DP", "DQ")
                                if cls == "II"
                                else ("A", "B", "C", "E", "F", "G")
                            )
                            if a.startswith("HLA-" + locus)
                        )
                cell = c.get("cell_type_override") or c.get("cell_type")
                for omitted in c.get("provenance", {}).get("unrepresented_apc_context", []):
                    cell = omitted.get("cell_type_override") or omitted.get("cell_type") or cell
                all_rows.append(
                    {
                        "model": model,
                        "class": cls,
                        "population": population,
                        "id": r["observation_id"],
                        "target": r["target"],
                        "prediction": r["prediction"],
                        "loss": r["loss"],
                        "locus": "+".join(sorted(loci)) or "unrecorded",
                        "candidates": str(len(o["mhc"])),
                        "length": (
                            (
                                "8-9"
                                if len(o["peptide"]) <= 9
                                else "10-11"
                                if len(o["peptide"]) <= 11
                                else "12+"
                            )
                            if cls == "I"
                            else (
                                "9-14"
                                if len(o["peptide"]) <= 14
                                else "15-20"
                                if len(o["peptide"]) <= 20
                                else "21+"
                            )
                        ),
                        "reported_cell": cell or "unknown",
                        "flanks": "known"
                        if c.get("source_n") is not None and c.get("source_c") is not None
                        else "masked",
                        "study": str(p.get("study_id") or p.get("pmid")),
                    }
                )
    d = pd.DataFrame(all_rows)
    summaries = []
    for field in ("locus", "candidates", "length", "reported_cell", "flanks", "study"):
        for (model, cls, pop, value), g in d.groupby(["model", "class", "population", field]):
            if g.target.nunique() != 2:
                continue
            summaries.append(
                {
                    "model": model,
                    "class": cls,
                    "population": pop,
                    "slice": field,
                    "value": value,
                    "rows": len(g),
                    "positives": int(g.target.sum()),
                    "loss": g.loss.mean(),
                    "auroc": roc_auc_score(g.target, g.prediction),
                    "auprc": average_precision_score(g.target, g.prediction),
                }
            )
    summary = pd.DataFrame(summaries)
    for cls in ("I", "II"):
        out = E / f"results/analysis/class-{cls}-slices.csv"
        summary[summary["class"] == cls].to_csv(out, index=False)
        print(out)


if __name__ == "__main__":
    main()
