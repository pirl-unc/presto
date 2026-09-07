#!/usr/bin/env python
"""Audit the capped real-source rows before training-time augmentation.

The mandatory production-contract preflight lives in ``code/launch.py`` and
executes the real trainer, including exclusions, synthetic decoys, MHC
resolution, split assignment, and collated target masks. This helper remains
an independent view of the source records and mapping strata only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from hitlist.downloads import data_dir as configured_hitlist_data_dir
from hitlist.downloads import set_data_dir
from presto.data.hitlist_source import load_records_from_hitlist
from presto.data.loaders import PrestoDataset, peptide_grouped_three_way_split_indices


DATA_SEED = 42


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(policy: str):
    return load_records_from_hitlist(
        max_binding=None,
        max_kinetics=500,
        max_stability=1_000,
        max_elution=20_000,
        mhc_allele="HLA-A*02:01",
        include_flanks=True,
        source_mapping_policy=policy,
        sampling_seed=DATA_SEED + 17,
    )


def _supervision_fingerprint(records) -> str:
    """Stream everything except the policy-controlled flank input."""
    ignored = {
        "flank_n",
        "flank_c",
        "flank_n_is_terminus",
        "flank_c_is_terminus",
    }
    digest = hashlib.sha256()
    for record in records:
        payload = {key: value for key, value in asdict(record).items() if key not in ignored}
        rendered = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        digest.update(len(rendered).to_bytes(8, "big"))
        digest.update(rendered)
    return digest.hexdigest()


def _configure_hitlist_snapshot(path: Path, artifact_names: tuple[str, ...]) -> Path:
    """Select and validate the one Hitlist snapshot used by this audit."""
    hitlist_dir = path.expanduser().resolve()
    missing_artifacts = [name for name in artifact_names if not (hitlist_dir / name).is_file()]
    if missing_artifacts:
        raise FileNotFoundError(
            f"Hitlist snapshot {hitlist_dir} is missing required artifacts: {missing_artifacts}"
        )
    set_data_dir(hitlist_dir)
    if configured_hitlist_data_dir().resolve() != hitlist_dir:
        raise RuntimeError(f"Hitlist data directory did not resolve to {hitlist_dir}")
    return hitlist_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hitlist-dir", type=Path, default=Path.home() / ".hitlist")
    args = parser.parse_args()

    artifact_names = (
        "observations.parquet",
        "binding.parquet",
        "peptide_mappings.parquet",
        "observations_meta.json",
        "peptide_mappings_meta.json",
    )
    hitlist_dir = _configure_hitlist_snapshot(args.hitlist_dir, artifact_names)

    masked = _load("mask_unresolved")
    legacy = _load("legacy_global_canonical")
    modality_names = ("binding", "kinetics", "stability", "elution")
    masked_modalities = (masked[0], masked[1], masked[2], masked[4])
    legacy_modalities = (legacy[0], legacy[1], legacy[2], legacy[4])
    parity = {
        name: _supervision_fingerprint(masked_rows) == _supervision_fingerprint(legacy_rows)
        for name, masked_rows, legacy_rows in zip(
            modality_names, masked_modalities, legacy_modalities
        )
    }
    if not all(parity.values()):
        raise RuntimeError(f"Policy supervision mismatch: {parity}")
    del legacy, legacy_modalities
    gc.collect()

    binding, kinetics, stability, elution = masked_modalities
    stats = masked[7]
    dataset = PrestoDataset(
        binding_records=binding,
        kinetics_records=kinetics,
        stability_records=stability,
        elution_records=elution,
        strict_mhc_resolution=False,
    )
    record_counts = {
        "binding": len(binding),
        "kinetics": len(kinetics),
        "stability": len(stability),
        "elution": len(elution),
        "dataset": len(dataset),
    }
    binding_categories = dict(
        sorted(Counter(record.source_mapping_category for record in binding).items())
    )
    binding_exact_categories = dict(
        sorted(
            Counter(
                record.source_mapping_category for record in binding if record.qualifier == 0
            ).items()
        )
    )

    per_seed = {}
    for seed in (42, 43, 44):
        train, val, test = peptide_grouped_three_way_split_indices(dataset, 0.1, 0.1, seed)
        partitions = {"train": train, "val": val, "test": test}
        split_audit = {}
        for name, indices in partitions.items():
            samples = [dataset[index] for index in indices]
            binding_samples = [sample for sample in samples if sample.bind_value is not None]
            split_audit[name] = {
                "rows": len(samples),
                "peptides": len({sample.peptide for sample in samples}),
                "binding_rows": len(binding_samples),
                "binding_exact_rows": sum(sample.bind_qual == 0 for sample in binding_samples),
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
            "data_seed_base": DATA_SEED,
            "sampling_seed": DATA_SEED + 17,
            "policy_supervision_parity": parity,
            "loader_stats": stats,
            "record_counts": record_counts,
            "binding_categories": binding_categories,
            "binding_exact_categories": binding_exact_categories,
            "split_audit": split_audit,
        }

    payload = {
        "scope": "pre-augmentation real-source audit; not the launch preflight",
        "hitlist_version": __import__("hitlist").__version__,
        "artifact_sha256": {name: _sha256(hitlist_dir / name) for name in artifact_names},
        "per_seed": per_seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for seed, audit in per_seed.items():
        print(seed, json.dumps(audit["record_counts"], sort_keys=True))
        print(
            seed,
            "test",
            json.dumps(audit["split_audit"]["test"]["binding_categories"], sort_keys=True),
        )


if __name__ == "__main__":
    main()
