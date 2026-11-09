"""Input-only projection from already canonical construct/context tables."""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

FAMILY = Path(__file__).resolve().parents[1]
RECIPE = Path("/private/tmp/hitlist-digestion-209-review-candidate")
PRIOR = FAMILY.parent / "2026-10-23_1202_codex_biological-cut-rates/results/fits.json"
TRAIN = ["FFH_12p5", "FFH_6p25", "H2A", "H2B", "IL37b", "LEDGF", "hTau"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert (
        sha(RECIPE / "manifest.json")
        == "5f616170b5514cb4d8f0c7b7843d232959e9514eebbca6483f976e17b0846615"
    )
    manifest = json.loads((RECIPE / "manifest.json").read_text())
    loaded = {}
    for name in ["constructs", "contexts"]:
        key = f"source_tables/{name}.parquet"
        assert sha(RECIPE / key) == manifest["files"][key]["sha256"]
        loaded[name] = pq.read_table(RECIPE / key).to_pylist()
    constructs = {r["construct_id"]: r for r in loaded["constructs"]}
    contexts = {r["condition_id"]: r for r in loaded["contexts"]}
    assert (
        sha(PRIOR) == "2400f214141b6ab421cc0b5168ec08b3209507b62b7cf0a590fd719937bd267f"
    )
    coefficients = json.loads(PRIOR.read_text())["sequence_context"]["coefficients"]
    cases = []
    for name in TRAIN:
        context = contexts["soh2024_digestion:condition:aSPIRE_" + name]
        construct = constructs[context["construct_id"]]
        cases.append(
            dict(
                condition=context["condition_id"],
                construct_id=context["construct_id"],
                sequence=construct["sequence"],
                sequence_sha256=construct["sequence_sha256"],
                substrate_concentration_um=context["substrate_concentration_um"],
                enzyme_substrate_denominator=context["enzyme_substrate_denominator"],
                n_residues=construct["n_residues"],
                enzyme_species=context["enzyme_species"],
                substrate_species=construct["substrate_species"],
            )
        )
    result = dict(
        cases=cases,
        initial_coefficients=coefficients,
        recipe_manifest_sha256=sha(RECIPE / "manifest.json"),
        read_files={
            str(RECIPE / key): sha(RECIPE / key)
            for key in [
                "manifest.json",
                "source_tables/constructs.parquet",
                "source_tables/contexts.parquet",
            ]
        },
        initializer_path=str(PRIOR),
        initializer_sha256=sha(PRIOR),
        source_targets_loaded=False,
        split_scope="Fixed seven existing training conditions; no examples/quantity/validation/test target files read",
    )
    (FAMILY / "results/runtime-inputs.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print([(r["condition"], r["n_residues"]) for r in cases], flush=True)


if __name__ == "__main__":
    main()
