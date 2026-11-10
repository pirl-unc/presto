"""Native positive-MS condition admission; no fitting or split/outcome evaluation."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from presto.data.modular_hitlist import AdmissionError, HumanMHCCatalog
from presto.data.modular_hitlist_ms import (
    FACTOR_FIELDS,
    has_resolved_condition,
    ms_observation,
    source_condition,
    study_factor_vocabulary,
)

EXPORT_SHA = "4a739794ab7a21ceb633df6e4cde4c01054f3e53213a5a8dcdc4d78604dde58f"
CATALOG_SHA = "ddc3493c64f4328de174a6c2cda1f007ec2f93a1ef649add25d2ef9c1720add1"
META = [
    "pmid",
    "mhc_class",
    "sample_attribution",
    "condition_id",
    "condition_status",
    "condition_evidence",
    "condition_reference",
    "cell_type",
    "cell_line_name",
    "condition_control",
    "condition_control_for",
    "condition_combination",
    *FACTOR_FIELDS,
]
ARM = [
    "pmid",
    "mhc_class",
    "condition_id",
    "cell_type",
    "cell_line_name",
    "mhc_allele_set",
    "ip_antibody",
    "instrument",
    "acquisition_mode",
    "condition_material",
    "condition_background",
    "condition_culture",
    "condition_mhc_context",
]
EXTRA = [
    "peptide",
    "is_ms_observation",
    "assay_modality",
    "evidence_kind",
    "ms_outcome_status",
    "mhc_species",
    "has_ptm",
    "peptide_modifications",
    "is_non_peptide_ligand",
    "is_potential_contaminant",
    "provenance_status",
    "evidence_row_id",
    "evidence_source_id",
    "provenance_id",
    "restriction_evidence",
    "mhc_allele_provenance",
    "mhc_allele_set",
    "attributed_sample_label",
    "sample_label",
    "sample_mhc",
    "reference_iri",
    "source",
    "specimen_status",
    "donor_status",
    "acquisition_status",
    "lineage_status",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def factors(context):
    return {
        **{f"intervention:{k}": v.state for k, v in context.interventions.items()},
        **{f"cytokine:{k}": v.state for k, v in context.cytokines.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    pa.set_cpu_count(1)
    assert sha(args.source) == EXPORT_SHA
    assert sha(args.catalog) == CATALOG_SHA
    args.output.mkdir(parents=True, exist_ok=False)
    if args.observations.exists():
        raise ValueError("Never overwrite an existing admission")
    pf = pq.ParquetFile(args.source)
    assert pf.metadata.num_rows == 5_440_019
    columns = sorted(set(META + ARM + EXTRA))
    assert set(columns) <= set(pf.schema_arrow.names)
    write(
        args.output / "schema.json",
        {"columns_read": columns, "source_columns": pf.schema_arrow.names},
    )
    metadata = Counter()
    for batch in pf.iter_batches(columns=META, batch_size=65536, use_threads=False):
        table = pa.Table.from_batches([batch]).append_column(
            "n_source_rows", pa.array([1] * len(batch))
        )
        for record in (
            table.group_by(META, use_threads=False)
            .aggregate([("n_source_rows", "sum")])
            .to_pylist()
        ):
            metadata[tuple(record[k] for k in META)] += record["n_source_rows_sum"]
    records = [
        {**dict(zip(META, key)), "n_source_rows": n} for key, n in metadata.items()
    ]
    assert sum(r["n_source_rows"] for r in records) == pf.metadata.num_rows
    pq.write_table(
        pa.Table.from_pylist(records), args.output / "condition-metadata.parquet"
    )
    vocabulary = study_factor_vocabulary(records)
    write(args.output / "study-factor-vocabulary.json", vocabulary)
    relevant = [
        r
        for r in records
        if has_resolved_condition(r)
        and factors(source_condition(r, vocabulary.get(str(r["pmid"] or ""), {})))
    ]
    condition_ids = {r["condition_id"] for r in relevant}
    relevant_counts = Counter()
    for r in relevant:
        relevant_counts[r["mhc_class"]] += r["n_source_rows"]
    print(
        json.dumps(
            {
                "stage": "metadata",
                "n_signatures": len(records),
                "n_conditions": len(condition_ids),
                "n_rows_with_qualified_factors_by_class": dict(relevant_counts),
            }
        ),
        flush=True,
    )
    with args.catalog.open() as stream:
        catalog = HumanMHCCatalog(csv.DictReader(stream))
    counts = Counter()
    arms = {}
    seen = set()
    ordinal = 0
    with gzip.open(args.observations, "wt") as output:
        for batch in pf.iter_batches(
            columns=columns, batch_size=8192, use_threads=False
        ):
            table = pa.Table.from_batches([batch]).append_column(
                "source_ordinal", pa.array(range(ordinal, ordinal + len(batch)))
            )
            ordinal += len(batch)
            table = table.filter(
                pc.is_in(
                    table["condition_id"], value_set=pa.array(sorted(condition_ids))
                )
            )
            for raw in table.to_pylist():
                if not has_resolved_condition(raw):
                    continue
                context = source_condition(
                    raw, vocabulary.get(str(raw.get("pmid") or ""), {})
                )
                claims = factors(context)
                if not claims:
                    continue
                cls = raw["mhc_class"]
                counts[(cls, "attempted")] += 1
                source_ordinal = raw.pop("source_ordinal")
                try:
                    row = ms_observation(
                        raw,
                        catalog,
                        source_sha256=EXPORT_SHA,
                        ordinal=source_ordinal,
                        study_factors=vocabulary,
                        dra_reference="2026-11-07_1645_codex_unified-data-audit:human_dr_dra0101_v1",
                    )
                except AdmissionError as exc:
                    counts[(cls, exc.disposition + ":" + exc.reason)] += 1
                    continue
                if row.observation_id in seen:
                    raise ValueError(
                        "Repeated evidence row identity needs explicit conflict review"
                    )
                seen.add(row.observation_id)
                counts[(cls, "admitted")] += 1
                row.provenance["raw_record_projection"] = columns
                row.provenance["audit_only_not_train_eligible"] = True
                output.write(
                    json.dumps(row.to_dict(), separators=(",", ":"), allow_nan=False)
                    + "\n"
                )
                key = tuple(raw[k] for k in ARM) + (json.dumps(claims, sort_keys=True),)
                if key not in arms:
                    arms[key] = {
                        "context": {k: raw[k] for k in ARM},
                        "factors": claims,
                        "n_observation_rows": 0,
                        "peptides": set(),
                        "source_ids": set(),
                        "n_candidate_mhc_pairs": set(),
                        "attributions": Counter(),
                        "evidence": Counter(),
                        "references": set(),
                        "specimen_status": Counter(),
                        "first_source_ordinals": [],
                    }
                arm = arms[key]
                arm["n_observation_rows"] += 1
                arm["peptides"].add(row.peptide)
                arm["source_ids"].add(raw["evidence_source_id"])
                arm["n_candidate_mhc_pairs"].add(len(row.mhc))
                arm["attributions"][raw["sample_attribution"]] += 1
                arm["evidence"][raw["condition_evidence"]] += 1
                arm["references"].add(raw["condition_reference"] or "")
                arm["specimen_status"][raw["specimen_status"] or "unknown"] += 1
                if len(arm["first_source_ordinals"]) < 3:
                    arm["first_source_ordinals"].append(source_ordinal)
            if ordinal % (8192 * 64) == 0:
                print(
                    json.dumps(
                        {
                            "stage": "admission",
                            "n_scanned_rows": ordinal,
                            "n_admitted_rows": len(seen),
                        }
                    ),
                    flush=True,
                )
    assert ordinal == pf.metadata.num_rows
    for cls, n in relevant_counts.items():
        assert counts[(cls, "attempted")] == n
        assert (
            sum(
                v
                for (c, reason), v in counts.items()
                if c == cls and reason != "attempted"
            )
            == n
        )
    summaries = []
    factor_studies = defaultdict(
        lambda: defaultdict(lambda: {"n_observation_rows": 0, "condition_ids": set()})
    )
    for arm in arms.values():
        arm["n_distinct_peptides"] = len(arm.pop("peptides"))
        arm["n_distinct_source_ids"] = len(arm.pop("source_ids"))
        for k in ("n_candidate_mhc_pairs", "references"):
            arm[k] = sorted(arm[k])
        summaries.append(arm)
        for factor, state in arm["factors"].items():
            entry = factor_studies[
                (arm["context"]["pmid"], arm["context"]["mhc_class"], factor)
            ][state]
            entry["n_observation_rows"] += arm["n_observation_rows"]
            entry["condition_ids"].add(arm["context"]["condition_id"])
    contrasts = []
    for (pmid, cls, factor), states in sorted(factor_studies.items()):
        for value in states.values():
            value["condition_ids"] = sorted(value["condition_ids"])
        if "present" in states and "absent" in states:
            contrasts.append(
                {
                    "pmid": pmid,
                    "mhc_class": cls,
                    "factor": factor,
                    "states": states,
                    "qualification": "candidate only; molecule/cell/capture/specimen confounding not yet reviewed",
                }
            )
    write(args.output / "admitted-arms.json", summaries)
    write(args.output / "candidate-contrasts.json", contrasts)
    summary = {
        "status": "passed",
        "schema": "presto.ms_condition_contrast_audit.v1",
        "inputs": {
            "source": {"path": str(args.source), "sha256": EXPORT_SHA},
            "catalog": {"path": str(args.catalog), "sha256": CATALOG_SHA},
        },
        "n_source_rows": ordinal,
        "n_metadata_signatures": len(records),
        "n_source_rows_with_qualified_factors_by_class": dict(relevant_counts),
        "dispositions": [
            {"mhc_class": c, "disposition": reason, "n_rows": n}
            for (c, reason), n in sorted(counts.items())
        ],
        "n_arm_contexts": len(summaries),
        "n_candidate_factor_contrasts": len(contrasts),
        "observations": {
            "path": str(args.observations),
            "sha256": sha(args.observations),
            "bytes": args.observations.stat().st_size,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": {
            "python": platform.python_version(),
            "pyarrow": pa.__version__,
            "cpu_threads": 1,
            "gpu": None,
        },
        "access": {
            "native_positive_source": True,
            "designated_test": False,
            "confirmation": False,
            "predictions": False,
        },
        "training_eligibility": "unassigned; contributor, source overlap, split and contrast qualification required",
    }
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
