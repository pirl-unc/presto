"""Uncapped Hitlist cohort through the canonical admission/background machinery."""

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

NAME = "2026-11-17_1155_codex_shared-presentation-expansion"
PRIOR = Path("/volume/2026-11-16_2035_codex_joint-apc-capacity")
PARENT_DATA = Path("/volume/2026-11-07_1915_codex_unified-presentation/prepared-v2")


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
    return {
        k: None if isinstance(v, float) and math.isnan(v) else v for k, v in r.items()
    }


def raw_rows(path, columns=None):
    for b in pq.ParquetFile(path).iter_batches(batch_size=16384, columns=columns):
        for r in b.to_pylist():
            yield clean(r)


def prepare(root, inputs, commit=lambda: None, *, prior=PRIOR, admission_only=False):
    start = time.monotonic()
    root.mkdir(parents=True, exist_ok=True)
    out = root / "inputs"
    out.mkdir(exist_ok=True)
    report = {
        "status": "preparing",
        "test_accessed": False,
        "source_policy": "Uncapped native Hitlist MS, known modifications excluded; unspecified PTMs explicitly admitted as sequence-only, not asserted unmodified. Unknown APC stays unknown.",
        "counts": {},
        "source_files": {p.name: sha(p) for p in inputs.iterdir() if p.is_file()},
    }

    def save(stage):
        report["stage"] = stage
        report["seconds"] = time.monotonic() - start
        (root / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(stage, report["seconds"], flush=True)
        commit()

    fixed = []
    for name in (
        "affinity-validation.jsonl",
        "apc-validation.jsonl",
        "broad-validation.jsonl",
    ):
        fixed.extend(rows(prior / "inputs" / name))
    fixed_sequences = {r["peptide"] for r in fixed}
    for r in rows(prior / "inputs/matching-candidates.jsonl"):
        fixed_sequences.update(o["peptide"] for o in r["options"])
    with gzip.open(inputs / "sealed-identity-embargo.txt.gz", "rt") as f:
        sealed = {p.strip() for p in f if p.strip()}
    protection = RelatedSequences(fixed_sequences | sealed)
    observed = set(
        pq.read_table(inputs / "observed-identities.parquet")["peptide"].to_pylist()
    )
    sources = [inputs / "hitlist-I.parquet", inputs / "hitlist-II.parquet"]
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
        for r in (
            pq.read_table(path, columns=fields)
            .to_pandas()
            .drop_duplicates()
            .to_dict("records")
        ):
            r = clean(r)
            metadata[json.dumps(r, sort_keys=True)] = r
    vocab = study_factor_vocabulary(metadata.values())
    del metadata
    samples = defaultdict(list)
    with (inputs / "samples.csv").open() as f:
        for r in csv.DictReader(f):
            samples[(str(r["pmid"]), r.get("condition_id", ""))].append(r)
    with (inputs / "mhc-full-seqs.csv").open() as f:
        catalog = HumanMHCCatalog(csv.DictReader(f))
    counts = Counter()
    by_study = Counter()
    peptides = set()
    identities = {}
    conflicts = set()
    admitted = root / "admitted.jsonl.gz"
    with gzip.open(admitted, "wt", compresslevel=1) as f:
        for path in sources:
            digest = report["source_files"][path.name]
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
                if (
                    str(r.get("pmid")) == "33592498"
                    and cls == "II"
                    and "parental" in (r.get("condition_id") or "")
                ):
                    counts["parental_class_ii_background"] += 1
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
                        dra_reference="HLA-DRA*01:01 reference alpha for source-reported DR beta; not measured typing",
                        legacy_source_reference=f"Hitlist 2.1.9 public export SHA256 {digest}; study {r.get('pmid')}",
                        unspecified_modification_policy="sequence_only_unknown_ptm",
                    )
                except (AdmissionError, ValueError) as error:
                    counts[
                        "admission:" + getattr(error, "reason", str(error)) + ":" + cls
                    ] += 1
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
    # Historical TRAIN exposure is consulted only for excluding new validation.
    parent_sequences = set()
    for p in [
        PARENT_DATA / "presentation-train.jsonl",
        PARENT_DATA / "concentrations-s42-concentrations-s42-train.jsonl",
        prior / "inputs/apc-train.jsonl",
    ]:
        for r in rows(p):
            parent_sequences.add(r["peptide"])
    proteins = read_gencode_proteins(inputs / "proteins.fa.gz")
    print(
        "mapping reference",
        len(proteins),
        "admitted peptides",
        len(peptides),
        "parent identities",
        len(parent_sequences),
        flush=True,
    )
    hits = map_peptides(
        peptides | parent_sequences | {r["peptide"] for r in fixed}, proteins
    )
    families = protein_families(hits)
    parent_families = {
        families[p] for seq in parent_sequences for p, _ in hits.get(seq, ())
    }
    fixed_families = {families[p] for r in fixed for p, _ in hits.get(r["peptide"], ())}
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
            "protein_reference_sha256": report["source_files"]["proteins.fa.gz"],
            "source_policy": "All exact reference occurrences; same possible-source protein background, not a measured protein assignment",
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
                "generation_policy": "one same-length possible-source protein background; observed identity and protected related-sequence exclusion; not measured nonbinding",
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
    (out / "new-validation-identities.txt").write_text(
        "\n".join(sorted(val_sequences)) + "\n"
    )
    # Refilter replay TRAIN against the expanded validation; keep existing validation exact.
    for name in (
        "affinity-train.jsonl",
        "apc-train.jsonl",
        "broad-train.jsonl",
        "affinity-validation.jsonl",
        "apc-validation.jsonl",
        "broad-validation.jsonl",
    ):
        path = prior / "inputs" / name
        with (out / name).open("w") as f:
            excluded = set()
            if "train" in name:
                for r in rows(path):
                    if train_protection.reason(r["peptide"]):
                        excluded.add(
                            r.get("provenance", {}).get("pair_id")
                            or r["observation_id"]
                        )
            for r in rows(path):
                if (
                    r.get("provenance", {}).get("pair_id") or r["observation_id"]
                ) in excluded:
                    continue
                dump(f, r)
        report["counts"]["replay_excluded_pairs:" + name] = len(excluded)
    # Explicit transfer arm: expanded II with the original class-I replay only.
    for destination, sources, cls in [
        ("expanded-II-train.jsonl", [trainpath], "II"),
        ("original-apc-I-train.jsonl", [out / "apc-train.jsonl"], "I"),
        ("original-broad-I-train.jsonl", [out / "broad-train.jsonl"], "I"),
    ]:
        with (out / destination).open("w") as f:
            for source in sources:
                for r in rows(source):
                    if {
                        m.get("class_override") or m["mhc_class"] for m in r["mhc"]
                    } == {cls}:
                        dump(f, r)
    (root / "source-family-map.json").write_text(
        json.dumps(families, sort_keys=True) + "\n"
    )
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
