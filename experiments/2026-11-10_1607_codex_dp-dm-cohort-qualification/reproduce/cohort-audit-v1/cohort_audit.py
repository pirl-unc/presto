"""Exact/core8/Hamming1 source audit, with a distinct substitution replay."""

import gzip
import hashlib
import json
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

AA = "ACDEFGHIKLMNPQRSTVWY"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def masks(p):
    # Same-position deletion contract from data/netmhc_training.py.
    return ((len(p), i, p[:i] + p[i + 1 :]) for i in range(len(p)))


def core8(p):
    return {p[i : i + 8] for i in range(len(p) - 7)}


def primary(candidates, references):
    reverse_core, reverse_mask = defaultdict(set), defaultdict(set)
    for p in candidates:
        for c in core8(p):
            reverse_core[c].add(p)
        for m in masks(p):
            reverse_mask[m].add(p)
    found = {k: set() for k in ("exact", "core8", "hamming1")}
    for p in references:
        if p in candidates:
            found["exact"].add(p)
        for c in core8(p):
            found["core8"].update(reverse_core.get(c, ()))
        for m in masks(p):
            found["hamming1"].update(reverse_mask.get(m, ()))
    return found


def independent(candidates, references):
    # Enumerate substitutions, never construct deletion masks.
    cores = set()
    alphabet = set(AA)
    for p in references:
        alphabet.update(p)
        cores.update(p[i : i + 8] for i in range(len(p) - 7))
    lengths = {len(p) for p in references}
    found = {k: set() for k in ("exact", "core8", "hamming1")}
    for p in candidates:
        if p in references:
            found["exact"].add(p)
        if any(p[i : i + 8] in cores for i in range(len(p) - 7)):
            found["core8"].add(p)
        if len(p) in lengths and (
            p in references
            or any(
                p[:i] + a + p[i + 1 :] in references
                for i in range(len(p))
                for a in alphabet
                if a != p[i]
            )
        ):
            found["hamming1"].add(p)
    return found


def json_rows(path, fields):
    from identity_projection import project_top_level_json

    with gzip.open(path, "rb") as stream:
        for line in stream:
            yield project_top_level_json(line, frozenset(fields))


def audit(contract, volume_root, bundle_root, out):
    started = time.monotonic()
    root = Path(volume_root)
    bundle = Path(bundle_root)
    out = Path(out)
    out.mkdir(exist_ok=False, parents=True)
    sources = {}
    for name, pin in contract["inputs"].items():
        path = (root if pin["location"] == "volume" else bundle) / pin["path"]
        assert path.stat().st_size == pin["bytes"] and sha(path) == pin["sha256"], name
        sources[name] = path
    extracted = out / "source-projections"
    extracted.mkdir()
    with tarfile.open(sources["prior_bundle"], "r:gz") as archive:
        for name, pin in contract["archive_members"].items():
            member = archive.getmember(pin["path"])
            assert member.isfile()
            path = extracted / name
            with archive.extractfile(member) as source, path.open("wb") as target:
                while b := source.read(8 << 20):
                    target.write(b)
            assert (
                path.stat().st_size == pin["size_bytes"] and sha(path) == pin["sha256"]
            )
            sources[name] = path
    candidates = pq.read_table(
        sources["candidate_export"],
        columns=[
            "peptide",
            "sample_label",
            "mhc_allele_set",
            "condition_transduction",
            "evidence_row_id",
        ],
    ).to_pylist()
    peptides = {r["peptide"] for r in candidates}
    assert len(candidates) == 21547 and len(peptides) == 10143
    references = defaultdict(set)
    counts = Counter()
    source_matches = Counter()

    def add(group, p, identity=None):
        assert isinstance(p, str) and p, group
        references[group].add(p)
        counts[group] += 1
        if identity and any(
            token in json.dumps(identity, sort_keys=True)
            for token in ("33025005", "PXD017154")
        ):
            source_matches[group] += 1

    for batch in pq.ParquetFile(sources["presentation"]).iter_batches(
        columns=[
            "peptide",
            "partition",
            "study_id",
            "group_id",
            "sample_id",
            "original_measurement_id",
        ]
    ):
        for r in batch.to_pylist():
            add(
                "presentation:" + r["partition"],
                r["peptide"],
                {k: v for k, v in r.items() if k not in ("peptide", "partition")},
            )
    for batch in pq.ParquetFile(sources["atlas"]).iter_batches(
        columns=["peptide", "fold"]
    ):
        for r in batch.to_pylist():
            add("atlas:" + r["fold"], r["peptide"])
    with gzip.open(sources["presentation_reserved"], "rt") as stream:
        for line in stream:
            add("presentation:reserved_full", line.strip())
    study_closure = json.loads(sources["presentation_studies"].read_text())
    if any(
        t in json.dumps(study_closure, sort_keys=True)
        for t in ("33025005", "PXD017154")
    ):
        source_matches["presentation:reserved_full"] += 1
    affinity_manifest = json.loads(sources["affinity_manifest"].read_text())
    for entry in affinity_manifest["inputs_and_outputs"]:
        name = entry["name"]
        path = sources["affinity_manifest"].parent / entry["output"]
        assert path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"]
        for r in json_rows(
            path,
            [
                "peptide",
                "observation_id",
                "study_id",
                "group_id",
                "original_measurement_id",
                "source",
                "pmid",
                "reference_iri",
            ],
        ):
            add(
                "affinity:" + name,
                r["peptide"],
                {k: v for k, v in r.items() if k != "peptide"},
            )
        assert counts["affinity:" + name] == entry["rows"]
    for name, partition in [
        ("tcell_rows", "partition"),
        ("tcell_drops", "base_partition"),
    ]:
        for r in json_rows(
            sources[name], ["peptide", partition, "component_id", "assay_id"]
        ):
            add("tcell:" + r[partition], r["peptide"])
    for r in json_rows(
        sources["tcell_components"], ["component_id", "base_partition", "pmid", "doi"]
    ):
        if str(r.get("pmid")) == "33025005":
            source_matches["tcell:" + r["base_partition"]] += 1
    assert set(references) == set(contract["cohort_roles"]), sorted(references)
    summaries = {}
    flagsets = {}
    protected = set()
    in_train = set()
    for group in sorted(references):
        found = primary(peptides, references[group])
        check = independent(peptides, references[group])
        assert found == check, group
        union = set().union(*found.values())
        role = contract["cohort_roles"][group]
        if role == "protected":
            protected.update(union)
        if role == "train":
            in_train.update(union)
        if role == "protected" and source_matches[group]:
            protected.update(peptides)
        flagsets[group] = found
        summaries[group] = {
            "role": role,
            "input_rows": counts[group],
            "unique_reference_peptides": len(references[group]),
            "source_token_matches": source_matches[group],
            **{
                f"n_{k}_candidate_rows": sum(r["peptide"] in values for r in candidates)
                for k, values in found.items()
            },
            "n_union_candidate_rows": sum(r["peptide"] in union for r in candidates),
            "independent_replay": "passed",
        }
        print(json.dumps({"cohort": group, **summaries[group]}), flush=True)
    membership = []
    sample_counts = Counter()
    condition_counts = Counter()
    pair_counts = Counter()
    remaining = defaultdict(set)
    for row in candidates:
        p = row["peptide"]
        embargo = p in protected
        result = {
            **row,
            "protected_identity_or_study_overlap": embargo,
            "existing_train_sequence_overlap": p in in_train,
            "identity_clear": not embargo,
            "training_eligible": False,
        }
        for group, found in flagsets.items():
            for kind, values in found.items():
                result[group + ":" + kind] = p in values
        membership.append(result)
        if not embargo:
            sample_counts[row["sample_label"]] += 1
            condition_counts[row["condition_transduction"]] += 1
            pair_counts[row["mhc_allele_set"]] += 1
            remaining[(row["mhc_allele_set"], row["condition_transduction"])].add(p)
    pq.write_table(
        pa.Table.from_pylist(membership), out / "candidate-dispositions.parquet"
    )
    paired = []
    for pair in sorted({k[0] for k in remaining}):
        absent = remaining[(pair, "none")]
        present = remaining[(pair, "HLA-DM")]
        paired.append(
            {
                "mhc_pair": pair,
                "n_absent_peptides": len(absent),
                "n_present_peptides": len(present),
                "n_both_peptides": len(absent & present),
                "n_absent_only_peptides": len(absent - present),
                "n_present_only_peptides": len(present - absent),
            }
        )
    result = {
        "status": "passed",
        "n_candidate_rows": len(candidates),
        "n_candidate_peptides": len(peptides),
        "n_protected_overlap_rows": sum(
            r["protected_identity_or_study_overlap"] for r in membership
        ),
        "n_identity_clear_rows": sum(r["identity_clear"] for r in membership),
        "n_identity_clear_peptides": len(peptides - protected),
        "n_training_eligible_rows": 0,
        "sample_identity_clear_rows": dict(sorted(sample_counts.items())),
        "condition_identity_clear_rows": dict(condition_counts),
        "pair_identity_clear_rows": dict(pair_counts),
        "paired_detection_support": paired,
        "cohorts": summaries,
        "source_matches": dict(source_matches),
        "seconds": time.monotonic() - started,
        "outcome_access": False,
        "locked_confirmation_records_accessed": False,
        "note": "Identity qualification only. Training remains pending source release, a frozen comparison/background/split contract and model-feature coverage. Nondetection is not an affinity-negative label.",
    }
    (out / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
