"""Audit native presenter processing inputs on the unchanged published cohort."""

import csv
import hashlib
import importlib.metadata
import json
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import presto
import pyarrow.parquet as pq
import torch
from datacache import materialize
from presto.data.allele_resolver import (
    PROCESSING_SPECIES_BUCKETS,
    PROCESSING_SPECIES_TO_IDX,
    normalize_processing_species_label,
)
from presto.data.collate import PrestoCollator, PrestoSample
from presto.data.hitlist_source import _lineage_fields
from presto.data.species_context import NATIVE_SPECIES_COLUMNS, species_lineage_fields
from presto.models.presto import _processing_species_idx_tensor
from presto.training.output_contract import build_output_contract
from presto.training.supervision import ROW_TASK_SPECS, resolve_row_targets
from presto.training.target_policy import AFFINITY_TARGET_POLICY

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
BASE = "https://github.com/pirl-unc/presto/releases/download/effective-target-fit-qualification-v1/"
INPUTS = {
    "prior-species-roles.tsv": {
        "url": "https://github.com/pirl-unc/presto/releases/download/molecular-species-lineage-audit-v1/species-role-observations.tsv",
        "sha256": "719becfaad3673f896785bf139715953368ec3a88cf57cf45500f685c35d049a",
        "size": 10067071,
    },
    "training.parquet": {
        "url": "https://github.com/pirl-unc/hitlist/releases/download/human-cohort-corrected-v1/training.parquet",
        "sha256": "f902341e90c1c57defc93f4bc540ffcbea1b41061afb4492ddd8aaa7642c65bd",
        "size": 188175209,
    },
    "membership.tsv": {
        "url": BASE + "membership.tsv",
        "sha256": "8959e60c0d9b8049aec25095b9c354d456b48f97ba8fb31d7b786e2861113e99",
        "size": 5855249,
    },
    "ic50-inspection.tsv": {
        "url": BASE + "ic50-inspection.tsv",
        "sha256": "ab0091db1284b72418a9eed16bb956194323f68154948fe04b5a20e993c8ab6c",
        "size": 1235284,
    },
    "source-species.tsv": {
        "url": BASE + "source-species.tsv",
        "sha256": "935cc99a385f06ea083ec23d67ab9a8c9ecc04fdc17beeef1642fbd0746c8f2e",
        "size": 3990280,
    },
}


def pin(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size": path.stat().st_size}


def text(value):
    return "" if value is None else str(value)


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def write_rows(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    assert presto.__version__ == "0.1.21"
    assert Path(presto.__file__).resolve() == ROOT / "__init__.py"
    code_paths = [
        "data/species_context.py",
        "data/vocab.py",
        "data/source_lineage.py",
        "models/presto.py",
        "data/collate.py",
        "data/hitlist_source.py",
        "data/loaders.py",
        "data/frozen_partitions.py",
        "training/supervision.py",
        "training/output_contract.py",
        "training/output_coverage.py",
    ]
    metadata = {
        "agent": "codex",
        "model": "GPT-6",
        "commit": commit,
        "dirty": False,
        "auditor_version": presto.__version__,
        "recipe": pin(__file__),
        "inputs": INPUTS,
        "initial_failed_recipe_commit": "fb39e8d302ef77a1b5a3160eb3fe442398418cf0",
        "initial_failure": "source snapshot copyfile arity; no generation published; tracked in #125",
        "code": {name: pin(ROOT / name) for name in code_paths},
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "datacache",
                "numpy",
                "pandas",
                "pyarrow",
                "torch",
                "mhcgnomes",
            )
        },
        "execution": "CPU data-only native presenter processing audit; no model or optimization",
    }

    def build(inputs, outputs):
        started = time.monotonic()
        membership = read_rows(inputs["membership.tsv"])
        selected = {row["evidence_row_id"]: row for row in membership}
        assert len(selected) == len(membership) == 9053
        prior_roles = {
            row["evidence_row_id"]: row
            for row in read_rows(inputs["prior-species-roles.tsv"])
        }
        assert prior_roles.keys() == selected.keys()
        old_species = {
            row["evidence_row_id"]: row
            for row in read_rows(inputs["source-species.tsv"])
        }
        ic50 = {
            row["evidence_row_id"]: row
            for row in read_rows(inputs["ic50-inspection.tsv"])
        }
        assert len(ic50) == 2614 and old_species.keys() == selected.keys()
        columns = list(
            dict.fromkeys(
                [
                    "evidence_row_id",
                    "peptide",
                    "pmid",
                    "reference_iri",
                    "reference_title",
                    "assay_iri",
                    "response_measured",
                    "measurement_units",
                    "quantitative_value",
                    "measurement_inequality",
                    "mhc_restriction",
                    "mhc_allele_set",
                    "mhc_class",
                    "cell_line_name",
                    "cell_type",
                    "source_tissue",
                    "condition_id",
                    "condition_status",
                    "condition_reference",
                    "assay_method",
                    *NATIVE_SPECIES_COLUMNS.values(),
                ]
            )
        )
        native = {}
        for batch in pq.ParquetFile(inputs["training.parquet"]).iter_batches(
            batch_size=5000, columns=columns
        ):
            values = batch.to_pydict()
            for index, identity in enumerate(values["evidence_row_id"]):
                if identity not in selected:
                    continue
                assert identity not in native
                native[identity] = {name: values[name][index] for name in columns}
        assert native.keys() == selected.keys()
        rows, changed, samples, effective, historical, ic50_rows = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for member in membership:
            identity = member["evidence_row_id"]
            source = native[identity]
            assert source["peptide"] == member["peptide"]
            for name, value in old_species[identity].items():
                if name in source:
                    assert text(source[name]) == value, (identity, name)
            kwargs = _lineage_fields(source)
            for key in ("source_bundle_id", "provenance_id", "provenance_status"):
                kwargs[key] = member[key]
            sample = PrestoSample(
                peptide=source["peptide"],
                mhc_class=source["mhc_class"],
                species=source["host"],
                sample_id=member["sample_id"],
                source_mhc_alleles=tuple(json.loads(member["source_mhc_alleles"])),
                resolved_mhc_alleles=tuple(json.loads(member["resolved_mhc_alleles"])),
                source_partition=member["source_partition"],
                **kwargs,
            )
            samples.append(sample)
            context = species_lineage_fields(sample)
            old_bucket = normalize_processing_species_label(source["host"]) or ""
            new_bucket = context["processing_species_bucket"]
            row = {
                **member,
                **{name: text(source[name]) for name in columns},
                "doi": member["doi"],
                **context,
                "historical_host_based_mhc_bucket": prior_roles[identity][
                    "historical_host_based_mhc_bucket"
                ],
                "corrected_molecular_mhc_bucket": "human",
                "mhc_label_changed": prior_roles[identity]["mhc_label_changed"],
                "historical_host_based_processing_bucket": old_bucket,
                "corrected_processing_bucket": new_bucket,
                "processing_context_changed": str(old_bucket != new_bucket),
            }
            for field, value in prior_roles[identity].items():
                assert row[field] == value, (identity, field)
            rows.append(row)
            if old_bucket != new_bucket:
                changed.append(row)
            if identity in ic50:
                prior = ic50[identity]
                assert sample.source_partition == prior["source_partition"]
                for current, original in (
                    ("source_endpoint", "source_endpoint"),
                    ("source_units", "source_units"),
                    ("source_inequality", "source_inequality"),
                ):
                    assert getattr(sample, current) == prior[original], (
                        identity,
                        current,
                    )
                raw = float(sample.source_value)
                assert raw == float(prior["source_value"])
                assert np.isclose(
                    np.log10(raw),
                    float(prior["raw_target_log10_nM"]),
                    rtol=0,
                    atol=1e-12,
                )
                effective.append(raw)
                historical.append(float(prior["effective_target_log10_nM"]))
                ic50_rows.append(
                    {
                        **row,
                        "historical_raw_target_log10_nM": prior["raw_target_log10_nM"],
                        "historical_effective_target_log10_nM": prior[
                            "effective_target_log10_nM"
                        ],
                    }
                )
        assert Counter(row["processing_species_bucket"] for row in rows) == {
            "human": 109,
            "murine": 15,
            "": 8929,
        }
        assert all(row["molecular_mhc_species_bucket"] == "human" for row in rows)
        assert len(changed) == 8936
        transformed = AFFINITY_TARGET_POLICY(
            torch.tensor(effective, dtype=torch.float32)
        ).numpy()
        assert len(transformed) == 2614 and np.array_equal(
            transformed, np.asarray(historical, dtype=np.float32)
        )
        for row, target in zip(ic50_rows, transformed):
            row["prospective_effective_target_log10_nM"] = str(float(target))
        spec = next(spec for spec in ROW_TASK_SPECS if spec.name == "mhc_species")
        n_canonical = 0
        for start in range(0, len(samples), 128):
            subset = samples[start : start + 128]
            batch = PrestoCollator()(subset).to("cpu")
            target = resolve_row_targets(batch, (spec,))["mhc_species"]
            assert target.mask.tolist() == [1] * len(subset)
            assert target.target.tolist() == [PROCESSING_SPECIES_TO_IDX["human"]] * len(
                subset
            )
            for index, sample in enumerate(subset):
                for field, value in species_lineage_fields(sample).items():
                    assert batch.source_lineage[field][index] == value
            expected_contexts = [
                row.species_context["presenting_species"] or None for row in subset
            ]
            assert batch.processing_species == expected_contexts
            actual_ids = _processing_species_idx_tensor(
                batch.model_inputs()["species"], len(subset), torch.device("cpu")
            )
            expected_ids = [
                PROCESSING_SPECIES_TO_IDX[
                    species_lineage_fields(row)["processing_species_bucket"]
                ]
                if species_lineage_fields(row)["processing_species_bucket"]
                else len(PROCESSING_SPECIES_BUCKETS)
                for row in subset
            ]
            assert actual_ids.tolist() == expected_ids
            n_canonical += len(subset)
        assert n_canonical == 9053
        contract = build_output_contract().to_dict()
        write_rows(outputs["presenter-context-observations.tsv"], rows)
        write_rows(outputs["changed-processing-contexts.tsv"], changed)
        write_rows(outputs["ic50-processing-contexts.tsv"], ic50_rows)
        write_json(
            outputs["output-policy.json"],
            {
                "schema_version": contract["schema_version"],
                "molecular_mhc_species_policy": contract[
                    "molecular_mhc_species_policy"
                ],
                "processing_species_policy": contract["processing_species_policy"],
                "output_contract": contract,
            },
        )
        write_json(
            outputs["audit.json"],
            {
                **metadata,
                "runtime_seconds": time.monotonic() - started,
                "runtime_scope": "builder; source acquisition and imports excluded",
                "process_peak_rss_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if sys.platform == "darwin" else 1024)
                ),
                "n_observations": len(rows),
                "n_changed_processing_contexts": len(changed),
                "n_ic50_by_processing_bucket": dict(
                    Counter(
                        row["processing_species_bucket"] or "unknown"
                        for row in ic50_rows
                    )
                ),
                "n_changed_by_partition": dict(
                    Counter(row["source_partition"] for row in changed)
                ),
                "n_unchanged_ic50_values_operators_partitions_and_effective_targets": len(
                    effective
                ),
                "n_canonical_collated_auxiliary_labels": n_canonical,
                "n_canonical_processing_model_inputs": n_canonical,
                "n_observations_by_axis": {
                    name: dict(Counter(row[name] or "unknown" for row in rows))
                    for name in (
                        "source_partition",
                        "host",
                        "mhc_species",
                        "presenting_species",
                        "in_vivo_host_species",
                        "source_organism",
                        "species_context_status",
                        "corrected_molecular_mhc_bucket",
                        "processing_species_bucket",
                        "processing_species_status",
                        "processing_species_basis",
                    )
                },
                "limits": [
                    "No model, MHC sequence reconstruction, training, new predictions or new full-input qualification.",
                    "Historical producer remains Presto 0.1.16 / Hitlist 1.78.0; historical contract and run unchanged.",
                    "Recorded host/antigen and reviewed presenter/transgene claims remain distinct; unknown attribution stays unknown.",
                    "Native processing inputs now use explicit presenters; source-antigen attribution, repertoire roles and legacy API/sampler semantics remain under #115/#46/#123.",
                    "Unknown donor/specimen/acquisition identities, shared studies and failed IC50 quality remain.",
                ],
            },
        )
        shutil.copyfile(__file__, outputs["audit_recipe.py"])
        shutil.copyfile(ROOT / "data/species_context.py", outputs["species_context.py"])

    names = [
        "presenter-context-observations.tsv",
        "changed-processing-contexts.tsv",
        "ic50-processing-contexts.tsv",
        "audit.json",
        "output-policy.json",
        "audit_recipe.py",
        "species_context.py",
    ]
    paths = materialize(
        Path.home() / ".hitlist/presto-presenter-context-audit" / commit,
        INPUTS,
        transform={"version": "native-presenter-context-audit-v2", "options": metadata},
        outputs={name: {} for name in names},
        builder=build,
    )
    (FAMILY / "results").mkdir(exist_ok=True)
    write_json(
        FAMILY / "results/audit-receipt.json",
        {
            "commit": commit,
            "files": {
                name: {"path": str(path), **pin(path)} for name, path in paths.items()
            },
        },
    )
    for name in ("audit.json", "output-policy.json"):
        shutil.copyfile(paths[name], FAMILY / "results" / name)
    print(
        json.dumps(
            {
                "commit": commit,
                "n_outputs": len(paths),
                "audit": json.loads(Path(paths["audit.json"]).read_text()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
