"""Exact reference/position audit; no biological admission or model fitting."""

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import datacache

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
RESULTS = FAMILY / "results"
SOURCE = ROOT / "experiments/2026-10-22_1630_codex_mapp-source-admission"
PUBLISHER = ROOT / "experiments/2026-10-22_1736_codex_mapp-publisher-reconciliation"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def jsonl(path, rows):
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def csvout(path, rows):
    with gzip.open(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rows(path, delimiter=","):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def record(accession, sequence, **metadata):
    assert sequence and re.fullmatch("[A-Z]+", sequence), accession
    return {
        "accession": accession,
        "sequence": sequence,
        "length": len(sequence),
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        **metadata,
    }


def parse_dat(path):
    out, lines = [], []
    with gzip.open(path, "rt") as stream:
        for line in stream:
            if line.strip() != "//":
                lines.append(line.rstrip("\n"))
                continue
            tags = defaultdict(list)
            sequence = []
            in_sequence = False
            for item in lines:
                if in_sequence:
                    sequence.append("".join(item.split()))
                else:
                    tags[item[:2]].append(item[5:])
                    in_sequence = item.startswith("SQ")
            accessions = " ".join(tags["AC"]).replace(";", " ").split()
            seq = "".join(sequence)
            assert int(re.search(r"(\d+) AA", tags["SQ"][0]).group(1)) == len(seq)
            seq_version = [
                int(re.search(r"sequence version (\d+)", x).group(1))
                for x in tags["DT"]
                if "sequence version " in x
            ]
            assert len(seq_version) == 1
            out.append(
                record(
                    accessions[0],
                    seq,
                    aliases=accessions[1:],
                    entry=tags["ID"][0].split()[0],
                    id_line=tags["ID"][0],
                    taxonomy=tags["OX"],
                    dates=tags["DT"],
                    sequence_version=seq_version[0],
                    header=None,
                )
            )
            lines = []
    assert not lines, "unterminated DAT record"
    return out


def parse_fasta(path):
    out, header, sequence = [], None, []

    def finish():
        if header is None:
            return
        fields = header.split("|", 2)
        assert len(fields) == 3 and fields[0] == "sp"
        version = re.search(r"\bSV=(\d+)", header)
        taxonomy = re.search(r"\bOX=(\d+)", header)
        out.append(
            record(
                fields[1],
                "".join(sequence),
                aliases=[],
                entry=fields[2].split()[0],
                header=header,
                id_line=None,
                dates=[],
                taxonomy=[taxonomy.group(1)] if taxonomy else [],
                sequence_version=int(version.group(1)) if version else None,
            )
        )

    with Path(path).open() as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish()
                header, sequence = line[1:], []
            else:
                assert header is not None
                sequence.append(line)
    finish()
    return out


def occurrences(sequence, peptide):
    positions, start = [], 0
    while True:
        start = sequence.find(peptide, start)
        if start < 0:
            return positions
        positions.append(start + 1)
        start += 1


def scan(reference, peptides, prefix_length=6):
    index = defaultdict(list)
    for pid, peptide in peptides.items():
        assert len(peptide) >= prefix_length
        index[peptide[:prefix_length]].append((pid, peptide))
    found = defaultdict(list)
    for parent in reference:
        sequence = parent["sequence"]
        for start in range(len(sequence) - prefix_length + 1):
            for pid, peptide in index.get(sequence[start : start + prefix_length], ()):
                if sequence.startswith(peptide, start):
                    found[pid].append((parent["accession"], start + 1))
    return found


def scanner_checks():
    peptides = {
        "a": "AAAAAA",
        "b": "AAAAAAA",
        "c": "BCDEFG",
        "d": "CDEFGH",
        "e": "ZZZZZZ",
        "f": "AAAAAABCDEFGH",
    }
    refs = [
        record("one", "AAAAAABCDEFGH"),
        record("two", "AAAAAAA"),
        record("short", "AAAAA"),
    ]
    found = scan(refs, peptides)
    expected = {
        pid: [
            (r["accession"], start)
            for r in refs
            for start in occurrences(r["sequence"], p)
        ]
        for pid, p in peptides.items()
    }
    assert {pid: sorted(found[pid]) for pid in peptides} == {
        pid: sorted(value) for pid, value in expected.items()
    }
    assert len(found["a"]) == 3 and len(found["b"]) == 1
    return {
        "passed": True,
        "peptides": len(peptides),
        "references": len(refs),
        "cases": "overlap, nested peptide prefixes, both termini, absent, shorter reference",
    }


def components(found):
    roots = {pid: pid for pid in found if found[pid]}

    def root(pid):
        while roots[pid] != pid:
            roots[pid] = roots[roots[pid]]
            pid = roots[pid]
        return pid

    owner = {}
    for pid in sorted(roots, key=int):
        for accession, _ in found[pid]:
            if accession in owner:
                a, b = root(pid), root(owner[accession])
                roots[max(a, b, key=int)] = min(a, b, key=int)
            else:
                owner[accession] = pid
    return {pid: root(pid) for pid in roots}


def main():
    tick = time.monotonic()
    assert not (RESULTS / "summary.json").exists(), "preserve completed audit"
    launch = {
        "agent_model": "Codex / GPT-6",
        "started": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "python": sys.version,
        "executable": sys.executable,
        "cwd": str(ROOT),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ),
        "environment_overrides": {},
        "code_sha256": sha(__file__),
        "model": None,
        "training": None,
        "pretraining": None,
        "loss": None,
        "device": "local CPU",
        "synthetic_data": "scanner software checks only",
        "predictive_split": None,
        "no_prediction_dump_reason": "reference source audit, no fitted model",
    }
    dump(FAMILY / "reproduce/launch.json", launch)
    shutil.copy2(__file__, FAMILY / "reproduce/source/audit.py")
    assets = json.loads((FAMILY / "reproduce/reference_assets.json").read_text())
    key = hashlib.sha256(
        json.dumps(assets["assets"], sort_keys=True).encode()
    ).hexdigest()
    installed = datacache.install_bundle(
        ROOT / "data/availability_sources" / key,
        assets["assets"],
        source_paths=assets["seed_paths"],
    )
    for name, path in installed.items():
        assert sha(path) == assets["assets"][name]["sha256"]
    original = json.loads((SOURCE / "results/input_paths.json").read_text())[
        "archive_members"
    ]
    hashes = json.loads((SOURCE / "reproduce/archive_members.json").read_text())
    inputs = {name: ROOT / path for name, path in original.items()}
    for name, path in inputs.items():
        assert sha(path) == hashes[name]["sha256"]
    sources = rows(inputs["txt/peptides.txt"], "\t")
    peptides = {r["id"]: r["Sequence"] for r in sources}
    assert len(peptides) == len(sources) == 10673 and len(
        set(peptides.values())
    ) == len(peptides)
    support_path = SOURCE / "results/peptide_support_ledger.csv.gz"
    membership_path = PUBLISHER / "results/publisher_membership.csv.gz"
    support = rows(support_path)
    membership = rows(membership_path)
    for r in support:
        assert peptides[r["peptide_id"]] == r["sequence"]
    for r in membership:
        assert peptides[r["source_peptide_id"]] == r["sequence"]
    subsets = {
        "all": set(peptides),
        "clean_source": {
            r["id"]
            for r in sources
            if not r["Reverse"] and not r["Potential contaminant"]
        },
        "reverse": {r["id"] for r in sources if r["Reverse"] == "+"},
        "contaminant": {r["id"] for r in sources if r["Potential contaminant"] == "+"},
        "structural_contrast": {
            r["peptide_id"]
            for r in support
            if r["structural_contrast_support"] == "True"
        },
    }
    for condition in ("UT", "TNF+IFN"):
        subsets["publisher_" + condition] = {
            r["source_peptide_id"]
            for r in membership
            if r["condition"] == condition and r["publisher_present"] == "True"
        }
    fields = [
        "id",
        "Sequence",
        "Proteins",
        "Leading razor protein",
        "Start position",
        "End position",
        "Amino acid before",
        "Amino acid after",
        "N-term cleavage window",
        "C-term cleavage window",
        "Protein group IDs",
        "Unique (Groups)",
        "Unique (Proteins)",
        "Reverse",
        "Potential contaminant",
        "PEP",
        "Length",
    ]
    jsonl(
        RESULTS / "source_peptides.jsonl.gz",
        [
            {
                "source_row": index + 2,
                "fields": {field: r[field] for field in fields},
                "full_source_row_sha256": hashlib.sha256(
                    json.dumps(r, sort_keys=True).encode()
                ).hexdigest(),
            }
            for index, r in enumerate(sources)
        ],
    )
    dump(
        RESULTS / "input_paths.json",
        {
            "references": {k: str(v) for k, v in installed.items()},
            "archive_members": {k: str(v) for k, v in inputs.items()},
            "support": str(support_path),
            "membership": str(membership_path),
            "support_sha256": sha(support_path),
            "membership_sha256": sha(membership_path),
        },
    )
    dump(RESULTS / "scanner_checks.json", scanner_checks())
    refs = {
        "reviewed_2023_canonical": parse_dat(installed["human-reviewed-2023.dat.gz"]),
        "reviewed_2026_03_isoforms": parse_fasta(
            installed["human-reviewed-2026_03-isoforms.fasta"]
        ),
    }
    summary, counts = {}, []
    for generation, reference in refs.items():
        print("mapping", generation, len(reference), flush=True)
        records = {r["accession"]: r for r in reference}
        assert len(records) == len(reference), "duplicate primary IDs"
        aliases = defaultdict(set)
        for r in reference:
            for accession in [r["accession"], *r["aliases"]]:
                aliases[accession].add(r["accession"])
        inventory = [{k: v for k, v in r.items() if k != "sequence"} for r in reference]
        jsonl(RESULTS / f"{generation}_entries.jsonl.gz", inventory)
        found = scan(reference, peptides)
        comp = components(found)
        source_map, matches, peptide_summary = [], [], []
        for r in sources:
            pid, peptide = r["id"], r["Sequence"]
            source_ids = r["Proteins"].split(";") if r["Proteins"] else []
            leading = r["Leading razor protein"]
            listed_ids = list(
                dict.fromkeys([*source_ids, *([leading] if leading else [])])
            )
            by_acc = defaultdict(list)
            for acc, start in found[pid]:
                by_acc[acc].append(start)
            listed_resolved, leading_matches = set(), []
            missing_ids = 0
            for accession in listed_ids:
                candidates = sorted(aliases.get(accession, ()))
                missing_ids += not candidates
                listed_resolved.update(candidates)
                evidence = []
                for candidate in candidates:
                    ref = records[candidate]
                    seq = ref["sequence"]
                    positions = by_acc[candidate]
                    coordinate = None
                    if accession == leading:
                        if (
                            r["Start position"].isdigit()
                            and r["End position"].isdigit()
                        ):
                            start, end = (
                                int(r["Start position"]),
                                int(r["End position"]),
                            )
                            in_range = 1 <= start <= end <= len(seq)
                            coordinate = {
                                "in_range": in_range,
                                "peptide_equal": in_range
                                and seq[start - 1 : end] == peptide,
                                "left_equal": in_range
                                and r["Amino acid before"]
                                == (seq[start - 2] if start > 1 else "-"),
                                "right_equal": in_range
                                and r["Amino acid after"]
                                == (seq[end] if end < len(seq) else "-"),
                            }
                        leading_matches.append(
                            {
                                "accession": candidate,
                                "positions": positions,
                                "coordinate": coordinate,
                            }
                        )
                    evidence.append(
                        {
                            "accession": candidate,
                            "identity_kind": "exact_primary_or_isoform"
                            if candidate == accession
                            else "recorded_secondary",
                            "positions_start1": positions,
                            "source_coordinate_comparison": coordinate,
                        }
                    )
                source_map.append(
                    {
                        "peptide_id": pid,
                        "source_accession": accession,
                        "listed_in_proteins": accession in source_ids,
                        "leading_razor": accession == leading,
                        "status": "missing_id" if not candidates else "resolved",
                        "candidates": evidence,
                    }
                )
            matched_ids = {acc for acc, _ in found[pid]}
            for acc, start in found[pid]:
                ref, end = records[acc], start + len(peptide) - 1
                seq = ref["sequence"]
                matches.append(
                    {
                        "peptide_id": pid,
                        "accession": acc,
                        "start1": start,
                        "end1": end,
                        "left": seq[start - 2] if start > 1 else "-",
                        "right": seq[end] if end < len(seq) else "-",
                        "sequence_sha256": ref["sequence_sha256"],
                        "source_listed_resolved": acc in listed_resolved,
                        "source_leading_resolved": acc in aliases.get(leading, set()),
                    }
                )
            coord = [
                x["coordinate"] for x in leading_matches if x["coordinate"] is not None
            ]
            peptide_summary.append(
                {
                    "peptide_id": pid,
                    "sequence": peptide,
                    "source_id_count": len(listed_ids),
                    "source_missing_id_count": missing_ids,
                    "source_resolved_id_count": len(listed_resolved),
                    "source_matched_identity_count": len(matched_ids & listed_resolved),
                    "reference_identity_count": len(matched_ids),
                    "reference_sequence_count": len(
                        {records[a]["sequence_sha256"] for a in matched_ids}
                    ),
                    "reference_occurrence_count": len(found[pid]),
                    "additional_identity_count": len(matched_ids - listed_resolved),
                    "leading_has_match": any(x["positions"] for x in leading_matches),
                    "leading_coordinate_match": any(x["peptide_equal"] for x in coord),
                    "leading_coordinate_and_flanks_match": any(
                        all(x.values()) for x in coord
                    ),
                    "leading_coordinate_conflict": bool(coord)
                    and not any(x["peptide_equal"] for x in coord),
                    "component": comp.get(pid, ""),
                }
            )
        jsonl(RESULTS / f"{generation}_source_mappings.jsonl.gz", source_map)
        csvout(RESULTS / f"{generation}_matches.csv.gz", matches)
        csvout(RESULTS / f"{generation}_peptides.csv.gz", peptide_summary)
        matched_parents = {m["accession"] for m in matches}
        jsonl(
            RESULTS / f"{generation}_matched_parents.jsonl.gz",
            [r for r in reference if r["accession"] in matched_parents],
        )
        for subset, ids in subsets.items():
            selected = [r for r in peptide_summary if r["peptide_id"] in ids]
            component_counts = Counter(
                r["component"] for r in selected if r["component"] != ""
            )
            selected_parent_ids = {acc for pid in ids for acc, _ in found[pid]}
            counts.append(
                {
                    "reference": generation,
                    "subset": subset,
                    "peptides": len(ids),
                    "matched_peptides": sum(
                        r["reference_identity_count"] > 0 for r in selected
                    ),
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
                    "matched_parent_identities": len(selected_parent_ids),
                    "matched_parent_sequences": len(
                        {records[a]["sequence_sha256"] for a in selected_parent_ids}
                    ),
                    "positions": sum(r["reference_occurrence_count"] for r in selected),
                    "components": len(component_counts),
                    "largest_component_peptides": max(
                        component_counts.values(), default=0
                    ),
                    "multi_peptide_components": sum(
                        x > 1 for x in component_counts.values()
                    ),
                }
            )
        summary[generation] = {
            "entries": len(reference),
            "residues": sum(r["length"] for r in reference),
            "distinct_full_sequences": len({r["sequence_sha256"] for r in reference}),
            "isoform_ids": sum("-" in r["accession"] for r in reference),
            "aliases_with_multiple_candidates": sum(
                len(x) > 1 for x in aliases.values()
            ),
            "source_accession_rows": len(source_map),
            "emitted_matches": len(matches),
            "matched_parent_entries": len(matched_parents),
            "source_accession_status": dict(Counter(r["status"] for r in source_map)),
        }
    csvout(RESULTS / "subset_summary.csv.gz", counts)
    dump(
        RESULTS / "summary.json",
        {
            "references": summary,
            "subset_summaries": counts,
            "elapsed_seconds": time.monotonic() - tick,
            "status": "mapping_complete_verification_pending",
            "source_reference": "Human_UP_2016_02.fasta; exact bytes unavailable",
            "interpretation": "reference-scoped exact matches, not unique biological origin or homology independence",
        },
    )
    outputs = {
        p.name: sha(p)
        for p in RESULTS.iterdir()
        if p.is_file() and p.name != "output_hashes.json"
    }
    dump(RESULTS / "output_hashes.json", outputs)
    print(
        json.dumps({"references": summary, "elapsed_seconds": time.monotonic() - tick}),
        flush=True,
    )


if __name__ == "__main__":
    main()
