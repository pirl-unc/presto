"""Sequence-only development-split audit; no predictions, target selection or resplitting."""

from pathlib import Path
import hashlib
import json

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
INPUTS = ROOT / "artifacts" / FAMILY.name / "inputs"


def collect(split):
    rows = []
    with (INPUTS / f"joint-{split}.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            mhc = row["mhc"][0]
            rows.append(
                (
                    row["peptide"],
                    mhc["mhc_class"],
                    hashlib.sha256(
                        (mhc["groove1"] + "|" + mhc["groove2"]).encode()
                    ).hexdigest(),
                )
            )
    return rows


def main():
    train = collect("train")
    exact = {c: {p for p, k, _ in train if k == c} for c in ("I", "II")}
    cores = {c: {p[i : i + 9] for p in exact[c] for i in range(len(p) - 8)} for c in exact}
    pairs = {c: {m for _, k, m in train if k == c} for c in exact}
    result = {
        "scope": "sequence-only audit before predictive fitting; original splits retained",
        "nine_mer_definition": "any contiguous 9aa substring, not an experimentally assigned binding core",
        "training_shared_exact_peptides_between_classes": len(exact["I"] & exact["II"]),
        "training_shared_nine_mer_substrings_between_classes": len(cores["I"] & cores["II"]),
        "subsets": {},
    }
    for split in ("validation", "test"):
        rows = collect(split)
        for target in ("I", "II"):
            subset = [(p, m) for p, k, m in rows if k == target]
            for source in ("I", "II"):
                n = sum(
                    any(p[i : i + 9] in cores[source] for i in range(len(p) - 8))
                    for p, _ in subset
                )
                result["subsets"][f"{source}-train_to_{target}-{split}"] = dict(
                    observations=len(subset),
                    exact_peptide_seen=sum(p in exact[source] for p, _ in subset),
                    any_nine_mer_seen=n,
                    any_nine_mer_fraction=n / len(subset),
                    mhc_pair_seen=sum(m in pairs[source] for _, m in subset),
                )
    (FAMILY / "results" / "sequence-overlap-audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
