"""Independent Biopython parse, Aho-Corasick scan, source replay and graph audit."""

import csv
import gzip
import hashlib
import json
import re
import shutil
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import Bio
from Bio import SeqIO, SwissProt

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
RESULTS = FAMILY / "results"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def readjsonl(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def readcsv(path, delimiter=","):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def direct_positions(sequence, peptide):
    return [
        match.start() + 1
        for match in re.finditer("(?=" + re.escape(peptide) + ")", sequence)
    ]


def automaton(patterns):
    """Independent character automaton, unlike the producer's six-mer index."""
    children, failure, outputs = [{}], [0], [[]]
    for pid, pattern in patterns.items():
        node = 0
        for letter in pattern:
            if letter not in children[node]:
                children[node][letter] = len(children)
                children.append({})
                failure.append(0)
                outputs.append([])
            node = children[node][letter]
        outputs[node].append(pid)
    queue = deque(children[0].values())
    while queue:
        parent = queue.popleft()
        for letter, child in children[parent].items():
            queue.append(child)
            target = failure[parent]
            while target and letter not in children[target]:
                target = failure[target]
            failure[child] = children[target].get(letter, 0)
            outputs[child].extend(outputs[failure[child]])
    return children, failure, outputs


def scan(sequences, patterns):
    children, failure, outputs = automaton(patterns)
    hits = set()
    for accession, sequence in sequences.items():
        node = 0
        for end0, letter in enumerate(sequence):
            while node and letter not in children[node]:
                node = failure[node]
            node = children[node].get(letter, 0)
            for pid in outputs[node]:
                hits.add((pid, accession, end0 - len(patterns[pid]) + 2))
    return hits


def graph_components(hits):
    peptide_parents, parent_peptides = defaultdict(set), defaultdict(set)
    for pid, parent, _ in hits:
        peptide_parents[pid].add(parent)
        parent_peptides[parent].add(pid)
    assignments = {}
    for pid in sorted(peptide_parents, key=int):
        if pid in assignments:
            continue
        queue, visited_parents = [pid], set()
        assignments[pid] = pid
        while queue:
            p = queue.pop()
            for parent in peptide_parents[p] - visited_parents:
                visited_parents.add(parent)
                for neighbor in parent_peptides[parent]:
                    if neighbor not in assignments:
                        assignments[neighbor] = pid
                        queue.append(neighbor)
    return assignments


def main():
    tick = time.monotonic()
    assert not (RESULTS / "independent-verification.json").exists()
    hashes = json.loads((RESULTS / "output_hashes.json").read_text())
    for name, digest in hashes.items():
        assert sha(RESULTS / name) == digest
    assets = json.loads((FAMILY / "reproduce/reference_assets.json").read_text())
    paths = json.loads((RESULTS / "input_paths.json").read_text())
    for name, path in paths["references"].items():
        assert sha(path) == assets["assets"][name]["sha256"]
    source_hashes = json.loads(
        (
            ROOT
            / "experiments/2026-10-22_1630_codex_mapp-source-admission/reproduce/archive_members.json"
        ).read_text()
    )
    for name, path in paths["archive_members"].items():
        assert sha(path) == source_hashes[name]["sha256"]
    for key in ("support", "membership"):
        assert sha(paths[key]) == paths[key + "_sha256"]
    source = readcsv(paths["archive_members"]["txt/peptides.txt"], "\t")
    by_id = {r["id"]: r for r in source}
    assert len(by_id) == len(source) == 10673
    preserved = readjsonl(RESULTS / "source_peptides.jsonl.gz")
    assert len(preserved) == len(source)
    for index, (r, frozen) in enumerate(zip(source, preserved, strict=True)):
        assert frozen["source_row"] == index + 2
        assert (
            frozen["full_source_row_sha256"]
            == hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
        )
        assert all(r[k] == value for k, value in frozen["fields"].items())
    support, publisher = readcsv(paths["support"]), readcsv(paths["membership"])
    subsets = {
        "all": set(by_id),
        "clean_source": {
            r["id"]
            for r in source
            if not r["Reverse"] and not r["Potential contaminant"]
        },
        "reverse": {r["id"] for r in source if r["Reverse"] == "+"},
        "contaminant": {r["id"] for r in source if r["Potential contaminant"] == "+"},
        "structural_contrast": {
            r["peptide_id"]
            for r in support
            if r["structural_contrast_support"] == "True"
        },
    }
    for condition in ("UT", "TNF+IFN"):
        subsets["publisher_" + condition] = {
            r["source_peptide_id"]
            for r in publisher
            if r["condition"] == condition and r["publisher_present"] == "True"
        }
    peptides = {pid: r["Sequence"] for pid, r in by_id.items()}
    toy_patterns = {"one": "ABA", "two": "BA", "three": "ABABA", "four": "X"}
    toy_sequences = {"a": "ABABA", "b": "BAX", "empty": ""}
    expected_toy = {
        (pid, acc, p)
        for pid, pep in toy_patterns.items()
        for acc, seq in toy_sequences.items()
        for p in direct_positions(seq, pep)
    }
    assert scan(toy_sequences, toy_patterns) == expected_toy
    totals = {}
    report = json.loads((RESULTS / "summary.json").read_text())
    for generation in report["references"]:
        reference = {}
        aliases = defaultdict(set)
        if generation == "reviewed_2023_canonical":
            with gzip.open(
                paths["references"]["human-reviewed-2023.dat.gz"], "rt"
            ) as stream:
                for r in SwissProt.parse(stream):
                    assert r.accessions[0] not in reference
                    assert r.taxonomy_id == ["9606"]
                    reference[r.accessions[0]] = {
                        "sequence": r.sequence,
                        "entry": r.entry_name,
                        "aliases": r.accessions[1:],
                        "length": r.sequence_length,
                        "sequence_version": r.sequence_update[1],
                    }
                    for alias in r.accessions:
                        aliases[alias].add(r.accessions[0])
        else:
            for r in SeqIO.parse(
                paths["references"]["human-reviewed-2026_03-isoforms.fasta"], "fasta"
            ):
                acc = r.id.split("|")[1]
                assert acc not in reference
                reference[acc] = {
                    "sequence": str(r.seq),
                    "entry": r.id.split("|")[2],
                    "aliases": [],
                    "length": len(r.seq),
                    "header": r.description,
                }
                aliases[acc].add(acc)
        inventory = readjsonl(RESULTS / f"{generation}_entries.jsonl.gz")
        assert (
            len(reference)
            == len(inventory)
            == report["references"][generation]["entries"]
        )
        assert {r["accession"] for r in inventory} == set(reference)
        for r in inventory:
            expected = reference[r["accession"]]
            for field, value in expected.items():
                if field != "sequence":
                    assert r[field] == value, (generation, r["accession"], field)
            assert (
                r["sequence_sha256"]
                == hashlib.sha256(expected["sequence"].encode()).hexdigest()
            )
        sequences = {acc: r["sequence"] for acc, r in reference.items()}
        all_hits = scan(sequences, peptides)
        emitted = readcsv(RESULTS / f"{generation}_matches.csv.gz")
        observed = {
            (r["peptide_id"], r["accession"], int(r["start1"])) for r in emitted
        }
        assert len(observed) == len(emitted) and observed == all_hits
        per_peptide = defaultdict(lambda: defaultdict(list))
        for pid, acc, start in sorted(all_hits):
            per_peptide[pid][acc].append(start)
        for r in emitted:
            pid, acc, start, end = (
                r["peptide_id"],
                r["accession"],
                int(r["start1"]),
                int(r["end1"]),
            )
            seq = sequences[acc]
            assert seq[start - 1 : end] == peptides[pid] and end - start + 1 == len(
                peptides[pid]
            )
            assert r["left"] == (seq[start - 2] if start > 1 else "-")
            assert r["right"] == (seq[end] if end < len(seq) else "-")
            assert r["sequence_sha256"] == hashlib.sha256(seq.encode()).hexdigest()
        matched = readjsonl(RESULTS / f"{generation}_matched_parents.jsonl.gz")
        assert {r["accession"] for r in matched} == {acc for _, acc, _ in all_hits}
        for r in matched:
            assert r["sequence"] == sequences[r["accession"]]
        source_mappings = readjsonl(RESULTS / f"{generation}_source_mappings.jsonl.gz")
        mapping_lookup = {
            (r["peptide_id"], r["source_accession"]): r for r in source_mappings
        }
        expected_keys = {
            (pid, accession)
            for pid, r in by_id.items()
            for accession in set(
                [*r["Proteins"].split(";"), r["Leading razor protein"]]
            )
            if accession
        }
        assert (
            len(mapping_lookup) == len(source_mappings)
            and set(mapping_lookup) == expected_keys
        )
        positive_candidates, negative_candidates = 0, 0
        for (pid, accession), mapping in mapping_lookup.items():
            src = by_id[pid]
            candidates = aliases.get(accession, set())
            assert mapping["listed_in_proteins"] == (
                accession in src["Proteins"].split(";")
            )
            assert mapping["leading_razor"] == (
                accession == src["Leading razor protein"]
            )
            assert mapping["status"] == ("resolved" if candidates else "missing_id")
            assert [r["accession"] for r in mapping["candidates"]] == sorted(candidates)
            for candidate in mapping["candidates"]:
                acc = candidate["accession"]
                seq = sequences[acc]
                positions = direct_positions(seq, peptides[pid])
                assert candidate["positions_start1"] == positions
                positive_candidates += bool(positions)
                negative_candidates += not positions
                assert candidate["identity_kind"] == (
                    "exact_primary_or_isoform"
                    if acc == accession
                    else "recorded_secondary"
                )
                coordinate = candidate["source_coordinate_comparison"]
                if (
                    mapping["leading_razor"]
                    and src["Start position"].isdigit()
                    and src["End position"].isdigit()
                ):
                    start, end = int(src["Start position"]), int(src["End position"])
                    valid = 1 <= start <= end <= len(seq)
                    expected = {
                        "in_range": valid,
                        "peptide_equal": valid
                        and seq[start - 1 : end] == peptides[pid],
                        "left_equal": valid
                        and src["Amino acid before"]
                        == (seq[start - 2] if start > 1 else "-"),
                        "right_equal": valid
                        and src["Amino acid after"]
                        == (seq[end] if end < len(seq) else "-"),
                    }
                    assert coordinate == expected
                else:
                    assert coordinate is None
        assignment = graph_components(all_hits)
        summaries = readcsv(RESULTS / f"{generation}_peptides.csv.gz")
        assert len(summaries) == len(peptides)
        native_summaries = []
        for r in summaries:
            pid = r["peptide_id"]
            src = by_id[pid]
            source_ids = set(
                [*src["Proteins"].split(";"), src["Leading razor protein"]]
            ) - {""}
            resolved = set().union(*(aliases.get(acc, set()) for acc in source_ids))
            matches = per_peptide[pid]
            lead_acc = aliases.get(src["Leading razor protein"], set())
            lead_mapping = mapping_lookup.get(
                (pid, src["Leading razor protein"]), {"candidates": []}
            )
            coords = [
                c["source_coordinate_comparison"]
                for c in lead_mapping["candidates"]
                if c["source_coordinate_comparison"] is not None
            ]
            expected = {
                "peptide_id": pid,
                "sequence": peptides[pid],
                "source_id_count": len(source_ids),
                "source_missing_id_count": sum(not aliases.get(a) for a in source_ids),
                "source_resolved_id_count": len(resolved),
                "source_matched_identity_count": len(set(matches) & resolved),
                "reference_identity_count": len(matches),
                "reference_sequence_count": len({sequences[a] for a in matches}),
                "reference_occurrence_count": sum(len(v) for v in matches.values()),
                "additional_identity_count": len(set(matches) - resolved),
                "leading_has_match": bool(set(matches) & lead_acc),
                "leading_coordinate_match": any(c["peptide_equal"] for c in coords),
                "leading_coordinate_and_flanks_match": any(
                    all(c.values()) for c in coords
                ),
                "leading_coordinate_conflict": bool(coords)
                and not any(c["peptide_equal"] for c in coords),
                "component": assignment.get(pid, ""),
            }
            assert r == {k: str(v) for k, v in expected.items()}, (generation, pid)
            native_summaries.append(expected)
        for subset_report in report["subset_summaries"]:
            if subset_report["reference"] != generation:
                continue
            ids = subsets[subset_report["subset"]]
            selected = [r for r in native_summaries if r["peptide_id"] in ids]
            parent_ids = {acc for pid in ids for acc in per_peptide[pid]}
            comp_counts = Counter(assignment[pid] for pid in ids if pid in assignment)
            expected = {
                "reference": generation,
                "subset": subset_report["subset"],
                "peptides": len(ids),
                "matched_peptides": sum(bool(per_peptide[pid]) for pid in ids),
                "source_listed_matched_peptides": sum(
                    r["source_matched_identity_count"] > 0 for r in selected
                ),
                "leading_matched_peptides": sum(
                    r["leading_has_match"] for r in selected
                ),
                "leading_coordinate_match": sum(
                    r["leading_coordinate_match"] for r in selected
                ),
                "leading_coordinate_flanks_match": sum(
                    r["leading_coordinate_and_flanks_match"] for r in selected
                ),
                "leading_coordinate_conflict": sum(
                    r["leading_coordinate_conflict"] for r in selected
                ),
                "one_reference_identity": sum(
                    r["reference_identity_count"] == 1 for r in selected
                ),
                "one_reference_sequence": sum(
                    r["reference_sequence_count"] == 1 for r in selected
                ),
                "multiple_reference_identities": sum(
                    r["reference_identity_count"] > 1 for r in selected
                ),
                "multiple_reference_sequences": sum(
                    r["reference_sequence_count"] > 1 for r in selected
                ),
                "additional_identity_peptides": sum(
                    r["additional_identity_count"] > 0 for r in selected
                ),
                "matched_parent_identities": len(parent_ids),
                "matched_parent_sequences": len({sequences[a] for a in parent_ids}),
                "positions": sum(r["reference_occurrence_count"] for r in selected),
                "components": len(comp_counts),
                "largest_component_peptides": max(comp_counts.values(), default=0),
                "multi_peptide_components": sum(n > 1 for n in comp_counts.values()),
            }
            assert expected == subset_report, (generation, subset_report["subset"])
        totals[generation] = {
            "reference_records": len(reference),
            "complete_scan_matches": len(all_hits),
            "source_accession_rows": len(source_mappings),
            "source_positive_candidates": positive_candidates,
            "source_negative_candidates": negative_candidates,
            "peptide_summary_rows": len(summaries),
            "component_assignments": len(assignment),
        }
        print(generation, "verified", totals[generation], flush=True)
    shutil.copy2(__file__, FAMILY / "reproduce/source/verify.py")
    output = {
        "passed": True,
        "biopython": Bio.__version__,
        "input_hashes_checked": 9,
        "output_hashes_checked": len(hashes),
        "source_rows": len(source),
        "references": totals,
        "complete_scan_method": "independent character-level Aho-Corasick, all reference residues",
        "component_method": "independent bipartite graph traversal, all-source mappings",
        "source_candidate_method": "overlapping regular-expression lookahead, all positive and negative candidates",
        "verifier_sha256": sha(__file__),
        "elapsed_seconds": time.monotonic() - tick,
    }
    (RESULTS / "independent-verification.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
