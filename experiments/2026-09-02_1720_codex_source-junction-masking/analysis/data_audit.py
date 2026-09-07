#!/usr/bin/env python
"""Freeze the exact capped data/category contract before remote launch."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from presto.data.hitlist_source import load_records_from_hitlist
from presto.data.loaders import PrestoDataset, peptide_grouped_three_way_split_indices


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(policy: str, seed: int):
    return load_records_from_hitlist(
        max_binding=None,
        max_kinetics=500,
        max_stability=1_000,
        max_elution=20_000,
        mhc_allele="HLA-A*02:01",
        include_flanks=True,
        source_mapping_policy=policy,
        sampling_seed=seed + 17,
    )


def _supervision_signature(records) -> list[dict]:
    """Everything except the policy-controlled flank input itself."""
    ignored = {
        "flank_n",
        "flank_c",
        "flank_n_is_terminus",
        "flank_c_is_terminus",
    }
    return [
        {key: value for key, value in asdict(record).items() if key not in ignored}
        for record in records
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hitlist-dir", type=Path, default=Path.home() / ".hitlist")
    args = parser.parse_args()

    per_seed = {}
    for seed in (42, 43, 44):
        masked = _load("mask_unresolved", seed)
        legacy = _load("legacy_global_canonical", seed)
        modality_names = ("binding", "kinetics", "stability", "elution")
        masked_modalities = (masked[0], masked[1], masked[2], masked[4])
        legacy_modalities = (legacy[0], legacy[1], legacy[2], legacy[4])
        parity = {
            name: _supervision_signature(masked_rows)
            == _supervision_signature(legacy_rows)
            for name, masked_rows, legacy_rows in zip(
                modality_names, masked_modalities, legacy_modalities
            )
        }
        if not all(parity.values()):
            raise RuntimeError(f"Policy supervision mismatch for seed {seed}: {parity}")

        binding, kinetics, stability, elution = masked_modalities
        stats = masked[7]
        dataset = PrestoDataset(
            binding_records=binding,
            kinetics_records=kinetics,
            stability_records=stability,
            elution_records=elution,
            strict_mhc_resolution=False,
        )
        train, val, test = peptide_grouped_three_way_split_indices(
            dataset, 0.1, 0.1, seed
        )
        partitions = {"train": train, "val": val, "test": test}
        split_audit = {}
        for name, indices in partitions.items():
            samples = [dataset[index] for index in indices]
            binding_samples = [
                sample for sample in samples if sample.bind_value is not None
            ]
            split_audit[name] = {
                "rows": len(samples),
                "peptides": len({sample.peptide for sample in samples}),
                "binding_rows": len(binding_samples),
                "binding_exact_rows": sum(
                    sample.bind_qual == 0 for sample in binding_samples
                ),
                "binding_categories": dict(
                    sorted(
                        Counter(
                            sample.source_mapping_category or "unmapped"
                            for sample in binding_samples
                        ).items()
                    )
                ),
            }
        per_seed[str(seed)] = {
            "sampling_seed": seed + 17,
            "policy_supervision_parity": parity,
            "loader_stats": stats,
            "record_counts": {
                "binding": len(binding),
                "kinetics": len(kinetics),
                "stability": len(stability),
                "elution": len(elution),
                "dataset": len(dataset),
            },
            "binding_categories": dict(
                sorted(
                    Counter(
                        record.source_mapping_category for record in binding
                    ).items()
                )
            ),
            "binding_exact_categories": dict(
                sorted(
                    Counter(
                        record.source_mapping_category
                        for record in binding
                        if record.qualifier == 0
                    ).items()
                )
            ),
            "split_audit": split_audit,
        }
        del masked, legacy, dataset, binding, kinetics, stability, elution
        gc.collect()

    artifact_names = (
        "observations.parquet",
        "binding.parquet",
        "peptide_mappings.parquet",
        "observations_meta.json",
        "peptide_mappings_meta.json",
    )
    payload = {
        "hitlist_version": __import__("hitlist").__version__,
        "artifact_sha256": {
            name: _sha256(args.hitlist_dir / name) for name in artifact_names
        },
        "per_seed": per_seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for seed, audit in per_seed.items():
        print(seed, json.dumps(audit["record_counts"], sort_keys=True))
        print(
            seed,
            "test",
            json.dumps(
                audit["split_audit"]["test"]["binding_categories"], sort_keys=True
            ),
        )


if __name__ == "__main__":
    main()
