"""Separate occurrence scanning and feature reconstruction for qualification."""

import json
from collections import defaultdict

import numpy as np

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def reconstruct(keys, parents):
    result = {}
    origin_sets = {}
    for k in keys:
        peptide, ids = json.loads(k)
        accum = np.zeros(46)
        origins = set()
        for pid in ids:
            seq = parents[pid]
            matches = [
                i
                for i in range(len(seq) - len(peptide) + 1)
                if seq[i : i + len(peptide)] == peptide
            ]
            assert matches
            for i in matches:
                end = i + len(peptide)
                term = np.zeros(46)
                term[:4] = [
                    np.log(1 + i),
                    np.log(1 + len(seq) - end),
                    int(i == 0),
                    int(end == len(seq)),
                ]
                term[4 + (ALPHABET.index(seq[i - 1]) if i else 20)] = 1
                term[25 + (ALPHABET.index(seq[end]) if end < len(seq) else 20)] = 1
                accum += term / len(ids) / len(matches)
                origins.add((pid, i, end))
        result[k] = accum
        origin_sets[k] = origins
    return result, origin_sets


def source_origin_sets(occurrences, reference_parent, source_ids):
    result = defaultdict(set)
    for row in occurrences:
        if row["source_record_id"] not in source_ids:
            continue
        pid = reference_parent.get(row["reference_record_id"])
        if pid is not None and row["start_1based"] is not None:
            result[row["source_record_id"]].add((pid, row["start_1based"] - 1, row["end_1based"]))
    return result


def fixture_check(producer):
    # Two occurrences in one parent and one terminal occurrence in another.
    parents = {"a": "ACAC", "b": "ACD"}
    k = json.dumps(["AC", ["a", "b"]], separators=(",", ":"))
    expected, origins = reconstruct([k], parents)
    actual, lineage = producer("AC", ["a", "b"], parents)
    assert np.max(abs(expected[k] - actual)) < 1e-14
    assert origins[k] == {("a", 0, 2), ("a", 2, 4), ("b", 0, 2)}
    assert sorted(x["weight"] for x in lineage) == [0.25, 0.25, 0.5]
    assert actual[2] == 0.75 and actual[3] == 0.25
    return {
        "status": "passed",
        "weights": [0.25, 0.25, 0.5],
        "n_terminal_fraction": 0.75,
        "c_terminal_fraction": 0.25,
    }
