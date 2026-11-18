"""Uncapped, protected presentation preparation shared by both MHC classes.

Generalized from the recorded broad presentation recipe; no experiment paths or
implicit parent checkpoints participate in the data contract.
"""

import csv
import gzip
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
from presto.data.modular import Observation
from presto.data.modular_hitlist import AdmissionError, HumanMHCCatalog
from presto.data.modular_hitlist_ms import (
    has_resolved_condition,
    ms_observation,
    sample_candidate_set,
    study_factor_vocabulary,
)
from presto.data.presentation import (
    RelatedSequences,
    background_candidate,
    map_peptides,
    protein_families,
    read_gencode_proteins,
    source_junction,
    stable_digest,
)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def rows(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f:
        for line in f:
            yield json.loads(line)


def dump(f, r):
    f.write(json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n")


def clean(r):
    return {k: None if isinstance(v, float) and math.isnan(v) else v for k, v in r.items()}


def raw_rows(path, columns=None):
    for b in pq.ParquetFile(path).iter_batches(batch_size=16384, columns=columns):
        for r in b.to_pylist():
            yield clean(r)


def _prepare(
    root,
    files,
    *,
    contract,
    fixed_sequences,
    sealed,
    parent_sequences,
    replay_sources,
    admission_only=False,
):
    start = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    out = root / "inputs"
    out.mkdir()
    report = {
        "status": "preparing",
        "test_accessed": False,
        "contract": contract,
        "source_policy": "Uncapped native MS; unknown APC and unspecified PTMs remain unknown",
        "counts": {},
        "source_files": {name: sha(path) for name, path in files.items()},
    }

    def save(stage):
        report["stage"] = stage
        report["seconds"] = time.monotonic() - start
        (root / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(stage, report["seconds"], flush=True)

    protection = RelatedSequences(fixed_sequences | sealed)
    observed = set(pq.read_table(files["observed"], columns=["peptide"])["peptide"].to_pylist())
    sources = [files["class_I"], files["class_II"]]
    metadata = {}
    fields = [
        "pmid",
        "condition_id",
        "condition_status",
        "condition_evidence",
        "condition_reference",
        "sample_attribution",
        "condition_knockout_genes",
        "condition_knockdown_genes",
        "condition_overexpression_genes",
        "condition_genetic_variants",
        "condition_transfection",
        "condition_transduction",
        "condition_cytokines",
        "condition_drugs",
        "condition_stimulation",
        "condition_background",
    ]
    for path in sources:
        for raw in (
            pq.read_table(path, columns=fields).to_pandas().drop_duplicates().to_dict("records")
        ):
            raw = clean(raw)
            metadata[json.dumps(raw, sort_keys=True)] = raw
    vocab = study_factor_vocabulary(metadata.values())
    samples = defaultdict(list)
    with files["samples"].open() as stream:
        for raw in csv.DictReader(stream):
            samples[(str(raw["pmid"]), raw.get("condition_id", ""))].append(raw)
    with files["catalog"].open() as stream:
        catalog = HumanMHCCatalog(csv.DictReader(stream))
    counts = Counter()
    by_study = Counter()
    peptides = set()
    identities = {}
    conflicts = set()
    admitted = root / "admitted.jsonl.gz"
    with gzip.open(admitted, "wt", compresslevel=1) as f:
        for path in sources:
            digest = sha(path)
            for ordinal, r in enumerate(raw_rows(path)):
                counts["source_rows"] += 1
                cls = r.get("mhc_class", "unknown")
                if not r.get("peptide"):
                    counts["missing_peptide"] += 1
                    continue
                reason = protection.reason(r["peptide"])
                if reason:
                    counts["protected_" + reason + ":" + cls] += 1
                    continue
                excluded = next(
                    (
                        rule
                        for rule in contract.get("source_exclusions", [])
                        if str(r.get("pmid")) == rule["pmid"]
                        and cls == rule["mhc_class"]
                        and rule.get("condition_id_contains", "") in (r.get("condition_id") or "")
                    ),
                    None,
                )
                if excluded:
                    counts["source_exclusion:" + excluded["reason"]] += 1
                    continue
                try:
                    if not r.get("mhc_allele_set") and has_resolved_condition(r):
                        matches = samples[(str(r["pmid"]), r["condition_id"])]
                        if len(matches) == 1:
                            r = sample_candidate_set(
                                r,
                                matches[0],
                                reference="pinned Hitlist samples.csv exact PMID+condition",
                            )
                    row = ms_observation(
                        r,
                        catalog,
                        source_sha256=digest,
                        ordinal=ordinal,
                        study_factors=vocab,
                        dra_reference=(
                            "HLA-DRA*01:01 reference for reported DR beta; not measured typing"
                        ),
                        legacy_source_reference=contract.get("legacy_source_reference"),
                        unspecified_modification_policy="sequence_only_unknown_ptm",
                    )
                except (AdmissionError, ValueError) as error:
                    counts["admission:" + getattr(error, "reason", str(error)) + ":" + cls] += 1
                    continue
                if not r.get("pmid"):
                    counts["missing_study"] += 1
                    continue
                signature = stable_digest(
                    json.dumps(
                        {
                            "peptide": row.peptide,
                            "mhc": [
                                m.to_dict()
                                if hasattr(m, "to_dict")
                                else (m.groove1, m.groove2, m.mhc_class)
                                for m in row.mhc
                            ],
                            "context": row.to_dict()["context"],
                            "value": row.value,
                        },
                        sort_keys=True,
                    )
                )
                if row.observation_id in identities:
                    if identities[row.observation_id] != signature:
                        conflicts.add(row.observation_id)
                        counts["conflicting_identity"] += 1
                    else:
                        counts["duplicate_identity"] += 1
                    continue
                identities[row.observation_id] = signature
                peptides.add(row.peptide)
                dump(f, row.to_dict())
                counts["admitted:" + cls] += 1
                by_study[(str(r["pmid"]), cls)] += 1
                if counts["source_rows"] % 200000 == 0:
                    print("admission", counts["source_rows"], dict(counts), flush=True)
    report["counts"] = dict(counts)
    report["admitted_studies"] = [
        {"pmid": p, "class": c, "rows": n} for (p, c), n in sorted(by_study.items())
    ]
    report["unique_admitted_peptides"] = len(peptides)
    report["conflicting_original_ids"] = len(conflicts)
    save("native_admission_complete")
    del identities, catalog
    if admission_only:
        return report
    proteins = read_gencode_proteins(files["proteins"])
    print(
        "mapping reference",
        len(proteins),
        "admitted peptides",
        len(peptides),
        "parent identities",
        len(parent_sequences),
        flush=True,
    )
    hits = map_peptides(peptides | parent_sequences | fixed_sequences, proteins)
    families = protein_families(hits)
    parent_families = {families[p] for seq in parent_sequences for p, _ in hits.get(seq, ())}
    fixed_families = {families[p] for peptide in fixed_sequences for p, _ in hits.get(peptide, ())}
    parent_protection = RelatedSequences(parent_sequences | fixed_sequences | sealed)
    newval_peptides = {
        p
        for p in peptides
        if hits.get(p)
        and families[hits[p][0][0]] not in parent_families | fixed_families
        and int(stable_digest(families[hits[p][0][0]])[:8], 16) % 5 == 0
        and not parent_protection.reason(p)
    }
    report["mapped_unique_admitted_peptides"] = sum(p in hits for p in peptides)
    report["reference_proteins"] = len(proteins)
    report["reference_families"] = len(set(families.values()))
    report["historical_train_families"] = len(parent_families)
    report["fixed_validation_families"] = len(fixed_families)
    report["new_validation_candidate_peptides"] = len(newval_peptides)
    save("reference_mapping_complete")
    # Generate new validation first, then protect every positive/background sequence.
    val_sequences = set()
    val_families = set()
    pair_cache = {}
    census = Counter()
    context_counts = defaultdict(Counter)

    def paired(row, split, guard):
        occurrences = hits.get(row.peptide)
        if not occurrences:
            counts["no_reference_occurrence:" + row.mhc[0].mhc_class] += 1
            return None
        family = families[occurrences[0][0]]
        context, policy = source_junction(row.peptide, occurrences, proteins)
        key = (split, row.peptide)
        if key not in pair_cache:
            pair_cache[key] = background_candidate(
                row.peptide,
                occurrences,
                proteins,
                observed=observed,
                protected=guard,
                key="shared-presentation:" + row.peptide,
            )
        candidate = pair_cache[key]
        if candidate is None:
            counts["no_qualified_background:" + row.mhc[0].mhc_class] += 1
            return None
        sequence, protein, position, negative_context = candidate
        if policy == "ambiguous_reference_junction":
            negative_context = context
        raw = row.provenance["raw_record"]
        study = str(raw["pmid"])
        pair = stable_digest(row.observation_id)[:32]
        provenance = {
            **row.provenance,
            "study_id": study,
            "pmid": study,
            "condition_id": raw.get("condition_id") or "",
            "pair_id": pair,
            "source_family": family,
            "source_protein": protein,
            "source_candidate_count": len(occurrences),
            "junction_policy": policy,
            "protein_reference_sha256": report["source_files"]["proteins"],
            "source_policy": (
                "All exact occurrences; possible-source background, not measured assignment"
            ),
        }
        positive = replace(
            row,
            split=split,
            group_id=family,
            context=replace(row.context, **context),
            provenance=provenance,
        )
        negid = "generated:shared-presentation:" + pair
        negative = replace(
            positive,
            observation_id=negid,
            peptide=sequence,
            value=0.0,
            context=replace(row.context, **negative_context),
            provenance={
                **provenance,
                "original_measurement_id": negid,
                "measurement_namespace": "synthetic_same_possible_source",
                "label_kind": "generated_background",
                "parent_positive_id": row.observation_id,
                "generated_position": position,
                "generation_policy": (
                    "Same-length source background with identity/related exclusion; not nonbinding"
                ),
            },
        )
        census[(split, study, row.mhc[0].mhc_class, raw.get("condition_id") or "")] += 1
        counts["junction:" + policy + ":" + split] += 1
        if split == "train":
            if row.context.cell_type:
                context_counts["cell_types"][row.context.cell_type] += 1
            for name in row.context.cytokines:
                context_counts["cytokines"][name] += 1
            for name in row.context.interventions:
                context_counts["interventions"][name] += 1
        return positive, negative

    valpath = out / "expanded-validation.jsonl"
    with valpath.open("w") as f:
        for r in rows(admitted):
            if r["observation_id"] in conflicts or r["peptide"] not in newval_peptides:
                continue
            result = paired(Observation.from_dict(r), "validation", parent_protection)
            if result:
                for row in result:
                    dump(f, row.to_dict())
                    val_sequences.add(row.peptide)
                    val_families.add(row.group_id)
    # The fixed embargo also applies to new validation backgrounds, checked separately.
    for r in rows(valpath):
        if protection.reason(r["peptide"]):
            raise ValueError("new validation collides with fixed/sealed identities")
    train_protection = RelatedSequences(fixed_sequences | sealed | val_sequences)
    trainpath = out / "expanded-train.jsonl"
    with trainpath.open("w") as f:
        for r in rows(admitted):
            p = r["peptide"]
            occurrences = hits.get(p)
            if r["observation_id"] in conflicts:
                continue
            if not occurrences:
                counts["unmapped_train_candidate"] += 1
                continue
            family = families[occurrences[0][0]]
            if family in val_families | fixed_families:
                counts["held_source_family"] += 1
                continue
            reason = train_protection.reason(p)
            if reason:
                counts["new_validation_related:" + reason] += 1
                continue
            result = paired(Observation.from_dict(r), "train", train_protection)
            if result:
                for row in result:
                    dump(f, row.to_dict())
    report["counts"] = dict(counts)
    report["census"] = [
        {"split": s, "study": p, "class": c, "condition": a, "positive_pairs": n}
        for (s, p, c, a), n in sorted(census.items())
    ]
    report["context_counts"] = {k: dict(v) for k, v in context_counts.items()}
    report["model_context"] = {k: sorted(v) for k, v in context_counts.items()}
    report["new_validation_sequences"] = len(val_sequences)
    report["new_validation_families"] = len(val_families)
    (out / "new-validation-identities.txt").write_text("\n".join(sorted(val_sequences)) + "\n")
    # Replay pairs are excluded together; supplied validation stays immutable.
    for number, source in enumerate(replay_sources):
        path = source["resolved_path"]
        name = f"replay-{number}-{source['split']}.jsonl"
        excluded = set()
        if source["split"] == "train":
            for raw in rows(path):
                if train_protection.reason(raw["peptide"]):
                    excluded.add(raw.get("provenance", {}).get("pair_id") or raw["observation_id"])
        with (out / name).open("w") as stream:
            for raw in rows(path):
                if raw["split"] != source["split"]:
                    raise ValueError("replay contents disagree with declared split")
                key = raw.get("provenance", {}).get("pair_id") or raw["observation_id"]
                if key not in excluded:
                    dump(stream, raw)
        report["counts"]["replay_excluded_pairs:" + name] = len(excluded)
    (root / "source-family-map.json").write_text(json.dumps(families, sort_keys=True) + "\n")
    report["files"] = {
        p.name: {
            "sha256": sha(p),
            "bytes": p.stat().st_size,
            "rows": sum(1 for _ in p.open()),
        }
        for p in out.glob("*.jsonl")
    }
    report["status"] = "prepared"
    save("complete")
    return {
        "status": report["status"],
        "files": report["files"],
        "seconds": report["seconds"],
        "model_context": report["model_context"],
    }


def _pinned_file(entry, base):
    if not isinstance(entry, dict) or not entry.get("sha256"):
        raise ValueError("every preparation input requires a path and SHA-256")
    path = (base / entry["path"]).resolve()
    if sha(path) != entry["sha256"]:
        raise ValueError(f"preparation input hash mismatch: {path}")
    return path


def prepare_from_manifest(manifest_path, output_dir, *, admission_only=False):
    """Verify source/identity pins before any population gets split membership."""
    from .modular import AA
    from .modular_sources import MANIFEST_VERSION

    path = Path(manifest_path).resolve()
    contract = json.loads(path.read_text())
    if contract.get("schema") != "presto.presentation-preparation.v1":
        raise ValueError("unsupported presentation preparation schema")
    required = {"class_I", "class_II", "samples", "catalog", "proteins", "observed"}
    if not required <= contract.get("files", {}).keys():
        raise ValueError("presentation preparation lacks required source assets")
    files = {name: _pinned_file(entry, path.parent) for name, entry in contract["files"].items()}
    contract["files"] = {
        name: {**contract["files"][name], "path": str(filename)} for name, filename in files.items()
    }
    if contract.get("identity_source_manifest"):
        from .presentation_identities import freeze_identities

        identity_manifest = _pinned_file(contract["identity_source_manifest"], path.parent)
        destination = Path(output_dir).with_name(Path(output_dir).name + "-identities")
        identity_receipt = freeze_identities(identity_manifest, destination)
        contract = {
            **contract,
            "identities": [*contract.get("identities", []), *identity_receipt["identities"]],
            "derived_identity_receipt_sha256": sha(destination / "receipt.json"),
        }
    if not contract.get("identities") or contract.get("ancestry_complete") is not True:
        raise ValueError("explicit protected identities and complete ancestry declaration required")
    identities = {name: set() for name in ("sealed", "fixed_validation", "ancestry")}
    for entry in contract["identities"]:
        role = entry["role"]
        if role not in identities:
            raise ValueError("identity role must be sealed, fixed_validation or ancestry")
        source = _pinned_file(entry, path.parent)
        entry["path"] = str(source)
        opener = gzip.open if source.suffix == ".gz" else open
        with opener(source, "rt") as stream:
            for line in stream:
                peptide = line.strip()
                if not peptide or set(peptide) - set(AA):
                    raise ValueError(
                        "identity protection accepts peptide-only files, never outcome tables"
                    )
                identities[role].add(peptide)
    if not identities["sealed"]:
        raise ValueError("sealed held-out identity protection must be nonempty")
    if contract.get("parent_checkpoint_sha256") and not identities["ancestry"]:
        raise ValueError("warm-start preparation requires parent-ancestry identities")
    replay = []
    for entry in contract.get("replay_sources", []):
        if entry.get("split") not in {"train", "validation"}:
            raise ValueError("replay source must declare train or validation before it is opened")
        replay.append({**entry, "resolved_path": _pinned_file(entry, path.parent)})
    root = Path(output_dir)
    report = _prepare(
        root,
        files,
        contract=contract,
        fixed_sequences=identities["fixed_validation"],
        sealed=identities["sealed"],
        parent_sequences=identities["ancestry"],
        replay_sources=replay,
        admission_only=admission_only,
    )
    if admission_only:
        return report
    sources = []
    for source in sorted((root / "inputs").glob("*.jsonl")):
        first = next(rows(source), None)
        if first is None:
            continue
        sources.append(
            {
                "path": str(source.relative_to(root)),
                "sha256": sha(source),
                "source": first["source"],
                "split": first["split"],
                "population": "source-admitted human presentation and declared replay",
                "label_policy": (
                    "Reported MS detections and generated backgrounds; not binding negatives"
                ),
                "split_policy": (
                    "Pinned ancestry/held-out identities; related sequences and source families"
                ),
            }
        )
    manifest = {
        "schema": MANIFEST_VERSION,
        "sources": sources,
        "preparation_manifest_sha256": sha(path),
        "model_context": report["model_context"],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return report


def audit_prepared(directory):
    """Recheck written pairs, reference origins and held-out protection independently."""
    root = Path(directory).resolve()
    report = json.loads((root / "preparation.json").read_text())
    if report.get("status") != "prepared":
        raise ValueError("preparation has not completed")
    contract = report["contract"]
    protected = {role: set() for role in ("sealed", "fixed_validation", "ancestry")}
    for entry in contract["identities"]:
        path = _pinned_file(entry, root)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            protected[entry["role"]].update(line.strip() for line in stream)
    proteins = read_gencode_proteins(_pinned_file(contract["files"]["proteins"], root))
    observed_path = _pinned_file(contract["files"]["observed"], root)
    observed = set(pq.read_table(observed_path, columns=["peptide"])["peptide"].to_pylist())
    validation_sequences, validation_families, seen_ids = set(), set(), set()
    counts, studies = Counter(), defaultdict(set)

    def state(row):
        return {
            key: value
            for key, value in row["context"].items()
            if key not in {"source_n", "source_c", "n_terminus", "c_terminus"}
        }

    for split in ("validation", "train"):
        guard = RelatedSequences(
            protected["sealed"]
            | protected["fixed_validation"]
            | (protected["ancestry"] if split == "validation" else validation_sequences)
        )
        path = root / "inputs" / f"expanded-{split}.jsonl"
        if sha(path) != report["files"][path.name]["sha256"]:
            raise ValueError("prepared observation hash mismatch")
        stream = iter(rows(path))
        n_rows = 0
        for positive in stream:
            negative = next(stream, None)
            if negative is None or positive["value"] != 1 or negative["value"] != 0:
                raise ValueError(
                    "preparation requires consecutive complete positive/background pairs"
                )
            pair = positive["provenance"]["pair_id"]
            if negative["provenance"]["pair_id"] != pair:
                raise ValueError("mismatched presentation pair")
            if positive["mhc"] != negative["mhc"] or positive["group_id"] != negative["group_id"]:
                raise ValueError("pair candidate or source-family mismatch")
            if state(positive) != state(negative):
                raise ValueError("pair APC context mismatch")
            for row in (positive, negative):
                Observation.from_dict(row)
                if row["split"] != split or row["observation_id"] in seen_ids:
                    raise ValueError("wrong partition or repeated observation identity")
                seen_ids.add(row["observation_id"])
                if guard.reason(row["peptide"]):
                    raise ValueError("prepared peptide collides with protected identities")
                if split == "validation":
                    validation_sequences.add(row["peptide"])
                    validation_families.add(row["group_id"])
                elif row["group_id"] in validation_families:
                    raise ValueError("source family crosses train/validation")
            provenance = negative["provenance"]
            sequence = proteins[provenance["source_protein"]].sequence
            position = provenance["generated_position"]
            length = len(positive["peptide"])
            if (
                positive["peptide"] not in sequence
                or len(negative["peptide"]) != length
                or sequence[position : position + length] != negative["peptide"]
                or negative["peptide"] in observed
            ):
                raise ValueError("background is not an unobserved same-source same-length window")
            cls = {mhc["mhc_class"] for mhc in positive["mhc"]}
            if len(cls) != 1:
                raise ValueError("pair must have one declared MHC class")
            key = split + ":" + next(iter(cls))
            counts[key] += 1
            studies[key].add(positive["provenance"]["study_id"])
            n_rows += 2
        if n_rows != report["files"][path.name]["rows"]:
            raise ValueError("prepared row count mismatch")
    return {
        "status": "passed",
        "preparation_sha256": sha(root / "preparation.json"),
        "pairs": dict(counts),
        "studies": {key: len(value) for key, value in studies.items()},
        "new_validation_has_both_classes": all(counts["validation:" + cls] for cls in ("I", "II")),
        "test_outcomes_accessed": False,
    }
