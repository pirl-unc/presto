"""Independent prepared-cohort audit using position-deletion Hamming witnesses."""

import gzip
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from prepare import PARENT_DATA, PRIOR, rows, sha
from presto.data.modular_hitlist_ms import source_cell_type
from presto.data.presentation import map_peptides, read_gencode_proteins


class WitnessIndex:
    def __init__(self, sequences):
        self.exact = set(sequences)
        self.cores = {p[i : i + 8] for p in self.exact for i in range(len(p) - 7)}
        self.masks = set()
        for p in self.exact:
            for i in range(len(p)):
                self.masks.add((len(p), i, p[:i] + p[i + 1 :]))

    def reason(self, p):
        if p in self.exact:
            return "exact"
        if any(p[i : i + 8] in self.cores for i in range(len(p) - 7)):
            return "shared8mer"
        if any((len(p), i, p[:i] + p[i + 1 :]) in self.masks for i in range(len(p))):
            return "hamming1"
        return None


def audit(root, inputs):
    start = time.monotonic()
    d = root / "inputs"
    report = json.loads((root / "preparation.json").read_text())
    if report["status"] != "prepared":
        raise ValueError("preparation not complete")
    historical_sources = {
        PARENT_DATA
        / "presentation-train.jsonl": "6609ee375f6bb117f19035560daa6cf50a63b9c229fa46ae24e773afebdda2fc",
        PARENT_DATA
        / "concentrations-s42-concentrations-s42-train.jsonl": "8593fbd416a489b46397c4fae288d5916ad55d2febec7f7798bc4bb3f788ec1c",
        PRIOR
        / "inputs/apc-train.jsonl": "7fb6794df84de0a044303ed09c6d0d6efdf1b9762b7f0f376e21ed88c2622411",
    }
    historical_sources[
        Path(
            "/volume/2026-11-07_1720_codex_affinity-development/prepared/concentrations-s42-train.jsonl"
        )
    ] = "acc22d1481ce8e0b45a6df86ca9a62e70721752955070b7eb26b85e48361635d"
    for path, digest in historical_sources.items():
        if sha(path) != digest:
            raise ValueError(
                "historical TRAIN differs from checkpoint source contract: " + str(path)
            )
    for name, record in report["files"].items():
        if sha(d / name) != record["sha256"]:
            raise ValueError("source hash mismatch " + name)
    held = set()
    held_groups = set()
    for name in [
        "affinity-validation.jsonl",
        "apc-validation.jsonl",
        "broad-validation.jsonl",
        "expanded-validation.jsonl",
    ]:
        for r in rows(d / name):
            held.add(r["peptide"])
            held_groups.add(r["group_id"])
    for r in rows(PRIOR / "inputs/matching-candidates.jsonl"):
        held.update(q["peptide"] for q in r["options"])
    with gzip.open(inputs / "sealed-identity-embargo.txt.gz", "rt") as f:
        held.update(p.strip() for p in f if p.strip())
    protected = WitnessIndex(held)
    observed = set(
        pq.read_table(inputs / "observed-identities.parquet")["peptide"].to_pylist()
    )
    results = {}
    cache = {}
    for name in [
        "affinity-train.jsonl",
        "apc-train.jsonl",
        "broad-train.jsonl",
        "expanded-train.jsonl",
    ]:
        errors = Counter()
        examples = []
        ids = set()
        count = 0
        pair_rows = defaultdict(list)
        for r in rows(d / name):
            count += 1
            p = r["peptide"]
            if p not in cache:
                cache[p] = protected.reason(p)
            reason = cache[p]
            if reason:
                errors["related_" + reason] += 1
                if len(examples) < 5:
                    examples.append({"id": r["observation_id"], "reason": reason})
            if r["split"] != "train":
                errors["wrong_split"] += 1
            if r["group_id"] in held_groups:
                errors["held_group"] += 1
            if r["observation_id"] in ids:
                errors["duplicate_id"] += 1
            ids.add(r["observation_id"])
            if name == "expanded-train.jsonl" and r["context"].get(
                "cell_type"
            ) != source_cell_type(r["provenance"]["raw_record"]):
                errors["reported_cell_identity_mismatch"] += 1
            if r["endpoint"] == "elution_observation" and name != "broad-train.jsonl":
                pair = r["provenance"].get("pair_id")
                if not pair:
                    errors["missing_pair_id"] += 1
                else:
                    # Keep the small identity/context signature, not full rows.
                    ctx = {
                        k: v
                        for k, v in r["context"].items()
                        if k
                        not in {
                            "source_n",
                            "source_c",
                            "n_terminus",
                            "c_terminus",
                            "provenance",
                        }
                    }
                    pair_rows[pair].append(
                        (
                            r["value"],
                            len(p),
                            r["group_id"],
                            json.dumps(r["mhc"], sort_keys=True),
                            json.dumps(ctx, sort_keys=True),
                        )
                    )
                if r["value"] == 0 and p in observed:
                    errors["observed_negative"] += 1
        for members in pair_rows.values():
            if (
                len(members) != 2
                or {m[0] for m in members} != {0.0, 1.0}
                or members[0][1:] != members[1][1:]
            ):
                errors["pair_mismatch"] += 1
        results[name] = {
            "rows": count,
            "errors": dict(errors),
            "examples": examples,
            "pair_audit": "all complete native pairs"
            if name in {"expanded-train.jsonl", "apc-train.jsonl"}
            else "not a paired native population",
        }
        print(name, results[name], flush=True)
    # New validation must be novel to all known historical parent/default TRAIN sequences.
    newval_rows = list(rows(d / "expanded-validation.jsonl"))
    newval = [r["peptide"] for r in newval_rows]
    historical = set()
    for p in historical_sources:
        for r in rows(p):
            historical.add(r["peptide"])
    del protected, cache, pair_rows, ids
    historical_index = WitnessIndex(historical)
    exposure = Counter(historical_index.reason(p) for p in newval)
    exposure.pop(None, None)
    families = json.loads((root / "source-family-map.json").read_text())
    reference = read_gencode_proteins(inputs / "proteins.fa.gz")
    historical_hits = map_peptides(historical, reference)
    historical_families = {
        families[p]
        for occurrences in historical_hits.values()
        for p, _ in occurrences
        if p in families
    }
    exposed_families = sorted(
        {r["group_id"] for r in newval_rows} & historical_families
    )
    result = {
        "status": "failed"
        if exposure or exposed_families or any(x["errors"] for x in results.values())
        else "passed",
        "files": results,
        "new_validation_historical_exposure": dict(exposure),
        "new_validation_historical_family_exposure": exposed_families,
        "new_validation_rows": len(newval),
        "protected_identity_count": len(held),
        "seconds": time.monotonic() - start,
        "test_accessed": False,
        "historical_source_hashes": {str(p): h for p, h in historical_sources.items()},
        "method": "Exact/shared8mer plus position-deletion Hamming witnesses, independently implemented from preparation half-string indexing",
    }
    (root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "passed":
        raise ValueError("prepared cohort failed independent audit")
    return result
