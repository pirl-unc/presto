"""Independent full reference parsing, all-hit replay and quantity conservation."""

import csv
import gzip
import hashlib
import importlib.util
import json
import re
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import Bio
import pyarrow.parquet as pq
from Bio import SeqIO, SwissProt

FAMILY = Path(__file__).resolve().parents[1]
ROOT = Path("/private/tmp/hitlist-v2-mapp-reference-second")
OLD = Path("/private/tmp/hitlist-v2-mapp-external-first")
SOURCE = ROOT / "external/pxd009092"
PRIOR = FAMILY.parent / "2026-10-22_1845_codex_mapp-parent-reference/results"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def table(name, reference=None):
    filters = [("reference_id", "=", reference)] if reference else None
    return pq.read_table(
        SOURCE / "typed" / f"{name}.parquet", filters=filters
    ).to_pylist()


def keyed(rows, keys):
    result = {tuple(r[k] for k in keys): r for r in rows}
    assert len(result) == len(rows), keys
    return result


def readcsv(path):
    with gzip.open(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def readjsonl(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(x) for x in stream]


def main():
    tick = time.monotonic()
    assert not (FAMILY / "results/independent-verification.json").exists()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    files = {
        str(p.relative_to(ROOT))
        for p in ROOT.rglob("*")
        if p.is_file() and p != ROOT / "manifest.json"
    }
    assert files == set(manifest["files"])
    for name, expected in manifest["files"].items():
        path = ROOT / name
        assert (
            sha(path) == expected["sha256"]
            and path.stat().st_size == expected["size_bytes"]
        ), name
    old_manifest = json.loads((OLD / "manifest.json").read_text())
    unchanged = {}
    for name, expected in old_manifest["files"].items():
        if name.startswith(("external/pxd009092/typed/", "external/pxd057588/")):
            assert sha(OLD / name) == expected["sha256"] == sha(ROOT / name), name
            unchanged[name] = expected["sha256"]
    assert sum(n.startswith("external/pxd009092/typed/") for n in unchanged) == 11
    prior_hashes = json.loads((PRIOR / "output_hashes.json").read_text())
    for name, expected in prior_hashes.items():
        assert sha(PRIOR / name) == expected, name
    assert json.loads((PRIOR / "independent-verification.json").read_text())
    # This frozen verifier uses BioPython and a character-level Aho-Corasick
    # scanner, independently of Hitlist's standard-library parser/prefix scan.
    spec = importlib.util.spec_from_file_location(
        "independent_scan", FAMILY / "reproduce/source/prior_reference_verifier.py"
    )
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    source_rows = {
        r["fields"]["id"]: r["fields"]
        for r in readjsonl(PRIOR / "source_peptides.jsonl.gz")
    }
    patterns = {pid: r["Sequence"] for pid, r in source_rows.items()}
    source_peptides = {r["peptide_row_id"]: r for r in table("peptide_rows")}
    assert len(source_peptides) == len(patterns) == 10673
    for pid, r in source_rows.items():
        actual = source_peptides["pxd009092:peptide:" + pid]
        for emitted, original in [
            ("sequence_raw", "Sequence"),
            ("parent_labels_raw", "Proteins"),
            ("leading_parent_raw", "Leading razor protein"),
            ("n_flank_raw", "Amino acid before"),
            ("c_flank_raw", "Amino acid after"),
        ]:
            assert actual[emitted] == r[original]
    references = table("sequence_references")
    contract = json.loads((SOURCE / "contracts/contract.json").read_text())
    assert keyed(references, ["reference_id"]) == keyed(
        contract["references"], ["reference_id"]
    )
    summaries = []
    all_resolutions = table("source_parent_resolutions")
    for ref in references:
        rid = ref["reference_id"]
        path = SOURCE / "assets" / ref["asset_path"]
        assert (
            sha(path) == ref["asset_sha256"]
            and path.stat().st_size == ref["asset_size_bytes"]
        )
        proteins, aliases = {}, defaultdict(set)
        if ref["reference_format"] == "uniprot_dat_gzip":
            with gzip.open(path, "rt") as stream:
                for r in SwissProt.parse(stream):
                    acc = r.accessions[0]
                    assert acc not in proteins and r.taxonomy_id == ["9606"]
                    proteins[acc] = (r.sequence, r.sequence_update[1], None)
                    for alias in r.accessions:
                        aliases[alias].add(acc)
        else:
            for r in SeqIO.parse(path, "fasta"):
                acc = r.id.split("|")[1]
                assert acc not in proteins
                version = re.search(r"\bSV=(\d+)", r.description)
                proteins[acc] = (
                    str(r.seq),
                    int(version.group(1)) if version else None,
                    r.description,
                )
                aliases[acc].add(acc)
        reference_rows = table("reference_proteins", rid)
        assert len(reference_rows) == len(proteins)
        inventory = {
            r["accession"]: r for r in readjsonl(PRIOR / f"{rid}_entries.jsonl.gz")
        }
        assert set(inventory) == set(proteins)
        for actual in reference_rows:
            acc = actual["accession"]
            seq, version, header = proteins[acc]
            assert actual["reference_record_id"] == f"{rid}:{acc}"
            assert (
                actual["sequence"] == seq
                and actual["sequence_version"] == version
                and actual["header_raw"] == header
            )
            assert actual["sequence_length"] == len(seq)
            assert (
                actual["sequence_sha256"]
                == hashlib.sha256(seq.encode()).hexdigest()
                == inventory[acc]["sequence_sha256"]
            )
            metadata = json.loads(actual["metadata_json"])
            expected = inventory[acc]
            if header is None:
                assert metadata == {
                    "id_lines": [expected["id_line"]],
                    "dates": expected["dates"],
                    "taxonomy": expected["taxonomy"],
                }
            else:
                taxonomy = re.search(r"\bOX=(\d+)", header)
                assert metadata == {
                    "entry": expected["entry"],
                    "taxonomy": taxonomy.group(1) if taxonomy else None,
                }
        accession_rows = table("reference_accessions", rid)
        expected_aliases = {
            (
                rid,
                f"{rid}:{acc}",
                alias,
                "primary" if alias == acc else "recorded_secondary",
            )
            for alias, candidates in aliases.items()
            for acc in candidates
        }
        assert len(accession_rows) == len(expected_aliases)
        assert {
            tuple(
                r[k]
                for k in (
                    "reference_id",
                    "reference_record_id",
                    "accession_raw",
                    "accession_role",
                )
            )
            for r in accession_rows
        } == expected_aliases
        hits = scanner.scan({a: p[0] for a, p in proteins.items()}, patterns)
        previous_hits = readcsv(PRIOR / f"{rid}_matches.csv.gz")
        assert hits == {
            (r["peptide_id"], r["accession"], int(r["start1"])) for r in previous_hits
        }
        occurrences = table("peptide_reference_occurrences", rid)
        assert len(occurrences) == len(hits)
        assert {
            (
                r["peptide_row_id"].rsplit(":", 1)[1],
                r["reference_record_id"][len(rid) + 1 :],
                r["start_1based"],
            )
            for r in occurrences
        } == hits
        by_peptide, sequence_owners = defaultdict(list), defaultdict(set)
        for r in occurrences:
            pid = r["peptide_row_id"].rsplit(":", 1)[1]
            acc = r["reference_record_id"][len(rid) + 1 :]
            seq = proteins[acc][0]
            start, end = r["start_1based"], r["end_1based"]
            assert r["occurrence_id"] == f"pxd009092:peptide:{pid}:{rid}:{acc}:{start}"
            assert seq[start - 1 : end] == patterns[pid] and end - start + 1 == len(
                patterns[pid]
            )
            assert r["n_flank"] == (seq[start - 2] if start > 1 else "-")
            assert r["c_flank"] == (seq[end] if end < len(seq) else "-")
            src = source_rows[pid]
            listed = set().union(
                *(aliases.get(a, set()) for a in src["Proteins"].split(";") if a)
            )
            assert r["source_listed_resolved"] == (acc in listed)
            assert r["source_leading_resolved"] == (
                acc in aliases.get(src["Leading razor protein"], set())
            )
            by_peptide[pid].append((acc, start))
            sequence_owners[seq].add(pid)
        claims = keyed(
            table("source_parent_claims", rid),
            ["peptide_row_id", "source_accession_raw"],
        )
        resolutions = keyed(
            [
                r
                for r in all_resolutions
                if r["reference_record_id"].startswith(rid + ":")
            ],
            ["claim_id", "reference_record_id"],
        )
        mappings = readjsonl(PRIOR / f"{rid}_source_mappings.jsonl.gz")
        assert len(claims) == len(mappings)
        expected_resolutions = 0
        for previous in mappings:
            pid, acc = previous["peptide_id"], previous["source_accession"]
            key = "pxd009092:peptide:" + pid
            cid = f"{key}:reference:{rid}:source_parent:{acc}"
            assert claims[key, acc] == {
                "claim_id": cid,
                "peptide_row_id": key,
                "reference_id": rid,
                "source_accession_raw": acc,
                "resolution_status": previous["status"],
                "n_candidate_records": len(previous["candidates"]),
                "listed_in_source": previous["listed_in_proteins"],
                "source_leading": previous["leading_razor"],
            }
            for candidate in previous["candidates"]:
                actual = resolutions[cid, f"{rid}:{candidate['accession']}"]
                positions = candidate["positions_start1"]
                assert positions == sorted(
                    start for a, start in by_peptide[pid] if a == candidate["accession"]
                )
                expected_resolutions += 1
                assert actual["accession_role"] == (
                    "primary" if candidate["accession"] == acc else "recorded_secondary"
                )
                assert actual["n_exact_occurrences"] == len(positions)
                assert actual["occurrence_status"] == (
                    "exact_occurrence" if positions else "resolved_sequence_absent"
                )
                comparison = candidate["source_coordinate_comparison"]
                if comparison is None:
                    status = "not_leading_source_claim"
                elif not comparison["in_range"]:
                    status = "out_of_range"
                elif not comparison["peptide_equal"]:
                    status = "peptide_mismatch"
                elif all(comparison.values()):
                    status = "peptide_and_flanks_match"
                else:
                    status = "peptide_match_flank_mismatch"
                assert actual["coordinate_status"] == status
                for field in ("in_range", "peptide_equal", "left_equal", "right_equal"):
                    assert actual["coordinate_" + field] == (
                        comparison[field] if comparison else None
                    )
        assert len(resolutions) == expected_resolutions
        # Independent graph traversal over literal full sequences, including unmatched singletons.
        component = {}
        for pid in patterns:
            if pid in component:
                continue
            queue, members, visited_sequences = [pid], {pid}, set()
            while queue:
                p = queue.pop()
                for acc, _ in by_peptide[p]:
                    seq = proteins[acc][0]
                    if seq in visited_sequences:
                        continue
                    visited_sequences.add(seq)
                    for neighbor in sequence_owners[seq] - members:
                        members.add(neighbor)
                        queue.append(neighbor)
            label = min("pxd009092:peptide:" + p for p in members)
            for p in members:
                component[p] = rid + ":component:" + label
        coverage = table("peptide_reference_coverage", rid)
        previous_coverage = {
            r["peptide_id"]: r for r in readcsv(PRIOR / f"{rid}_peptides.csv.gz")
        }
        assert len(coverage) == len(patterns)
        integer_fields = {
            "n_source_claims": "source_id_count",
            "n_missing_claims": "source_missing_id_count",
            "n_source_resolved_records": "source_resolved_id_count",
            "n_source_matched_records": "source_matched_identity_count",
            "n_reference_records": "reference_identity_count",
            "n_reference_sequences": "reference_sequence_count",
            "n_occurrences": "reference_occurrence_count",
            "n_additional_records": "additional_identity_count",
        }
        flag_fields = {
            "leading_has_occurrence": "leading_has_match",
            "leading_coordinate_match": "leading_coordinate_match",
            "leading_coordinate_flanks_match": "leading_coordinate_and_flanks_match",
            "leading_coordinate_conflict": "leading_coordinate_conflict",
        }
        for actual in coverage:
            pid = actual["peptide_row_id"].rsplit(":", 1)[1]
            previous = previous_coverage[pid]
            for new, old in integer_fields.items():
                assert actual[new] == int(previous[old]), (
                    rid,
                    pid,
                    new,
                    actual[new],
                    previous[old],
                )
            for new, old in flag_fields.items():
                assert actual[new] == (previous[old] == "True"), (rid, pid, new)
            assert actual["mapping_status"] == (
                "exact_occurrence" if by_peptide[pid] else "no_exact_occurrence"
            )
            assert actual["shared_sequence_component_id"] == component[pid]
        summaries.append(
            {
                "reference_id": rid,
                "n_proteins": len(proteins),
                "n_accession_claims": len(accession_rows),
                "n_occurrences": len(occurrences),
                "n_source_claims": len(claims),
                "n_source_resolutions": len(resolutions),
                "n_mapped_peptides": sum(bool(by_peptide[p]) for p in patterns),
                "n_sequence_components_including_unmatched": len(
                    set(component.values())
                ),
                "source_resolution_status": dict(
                    Counter(r["resolution_status"] for r in claims.values())
                ),
                "candidate_occurrence_status": dict(
                    Counter(r["occurrence_status"] for r in resolutions.values())
                ),
                "coordinate_status": dict(
                    Counter(r["coordinate_status"] for r in resolutions.values())
                ),
            }
        )
    report = {
        "status": "passed",
        "seconds": time.monotonic() - tick,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "biopython_version": Bio.__version__,
        "n_package_files": len(files),
        "unchanged_prior_source_artifacts": unchanged,
        "prior_output_hashes": prior_hashes,
        "references": summaries,
        "scope": "All reference sequences/aliases/occurrences/source resolutions/coverage independently checked; all eleven original MAPP typed tables and PXD057588 source artifacts byte-identical. No fit, split, precursor or concentration claims.",
    }
    (FAMILY / "results/independent-verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("unchanged_prior_source_artifacts", "prior_output_hashes")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
