"""Pair-preserving APC-only validation perturbations; never alter source context."""

import copy
import hashlib
import json
import random
from collections import defaultdict

FIELDS = (
    "components",
    "component_overrides",
    "cell_type",
    "cell_type_override",
    "cytokines",
    "cytokine_overrides",
    "interventions",
    "intervention_overrides",
    "processing_pathways",
    "pathway_overrides",
)


def signature(context):
    return {
        k: copy.deepcopy(context.get(k, None if k.startswith("cell_type") else {})) for k in FIELDS
    }


def functional(value):
    if isinstance(value, dict):
        return {k: functional(v) for k, v in value.items() if k not in ("evidence", "provenance")}
    if isinstance(value, list):
        return [functional(v) for v in value]
    return value


def canonical(value):
    return json.dumps(functional(value), sort_keys=True, separators=(",", ":"))


def controlled(records, mode, seed=20261117):
    if mode not in ("original", "unknown", "within-study-shuffle"):
        raise ValueError(mode)
    pairs = defaultdict(list)
    for row in records:
        assert row["split"] == "validation"
        pairs[row["provenance"]["pair_id"]].append(row)
    groups, signatures = defaultdict(list), {}
    for pair, members in pairs.items():
        assert len(members) == 2 and sorted(r["value"] for r in members) == [0, 1]
        a, b = members
        assert canonical(signature(a["context"])) == canonical(signature(b["context"]))
        classes = {m.get("class_override") or m["mhc_class"] for r in members for m in r["mhc"]}
        assert len(classes) == 1
        studies = {r["provenance"].get("study_id") or r["provenance"].get("pmid") for r in members}
        assert len(studies) == 1 and None not in studies
        groups[(next(iter(classes)), str(next(iter(studies))))].append(pair)
        signatures[pair] = signature(a["context"])
    donors = {}
    group_report = []
    for key, ids in sorted(groups.items()):
        ids = sorted(ids)
        shuffled = ids.copy()
        stable_seed = int.from_bytes(
            hashlib.sha256(json.dumps([seed, key]).encode()).digest()[:8], "big"
        )
        random.Random(stable_seed).shuffle(shuffled)
        donors.update(zip(ids, shuffled, strict=True))
        group_report.append(
            {
                "class": key[0],
                "study": key[1],
                "pairs": len(ids),
                "distinct_apc_signatures": len({canonical(signatures[i]) for i in ids}),
            }
        )
    result, changed_pairs = [], set()
    for source in records:
        row = copy.deepcopy(source)
        pair = row["provenance"]["pair_id"]
        replacement = (
            signatures[pair]
            if mode == "original"
            else signature({})
            if mode == "unknown"
            else signatures[donors[pair]]
        )
        if canonical(replacement) != canonical(signatures[pair]):
            changed_pairs.add(pair)
        row["context"].update(copy.deepcopy(replacement))
        row["provenance"]["apc_input_control"] = {
            "mode": mode,
            "seed": seed,
            "donor_pair": donors[pair] if mode == "within-study-shuffle" else None,
            "original_apc_context": signatures[pair],
        }
        # Verify every biological/outcome field outside APC is exactly preserved.
        a, b = copy.deepcopy(source), copy.deepcopy(row)
        b["provenance"].pop("apc_input_control")
        for record in (a, b):
            for field in FIELDS:
                record["context"].pop(field, None)
        assert a == b
        result.append(row)
    mapping = {pair: donors[pair] for pair in sorted(donors)}
    return result, {
        "mode": mode,
        "seed": seed,
        "rows": len(records),
        "pairs": len(pairs),
        "changed_pairs_before_vocab_projection": len(changed_pairs),
        "groups": group_report,
        "donor_mapping": mapping if mode == "within-study-shuffle" else None,
        "non_apc_invariants_verified": True,
    }
