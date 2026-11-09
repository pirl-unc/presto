"""Reference-label preparation for shared representation pretraining."""

import hashlib
import json
from pathlib import Path
import random

import numpy as np

from .modular import Observation
from .modular_sources import file_sha256


def weirdo_observations(
    directory,
    *,
    archive_sha256,
    metadata_sha256,
    reference_sha256,
    limits,
    seed,
):
    """Read Weirdo's actual family-split format with label-blind bounded selection.

    Labels mean membership in the pinned SwissProt reference, not one unique
    organism of origin. Family buckets remain source-declared grouping evidence;
    this adapter does not certify the upstream family construction or taxonomy.
    """
    directory = Path(directory)
    archive_path, metadata_path = directory / "splits.npz", directory / "splits.json"
    if file_sha256(archive_path) != archive_sha256 or file_sha256(metadata_path) != metadata_sha256:
        raise ValueError("Weirdo source hash mismatch")
    if len(reference_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in reference_sha256
    ):
        raise ValueError("pinned reference-sequence SHA256 required")
    metadata = json.loads(metadata_path.read_text())
    categories = metadata["categories"]
    if not categories or len(set(categories)) != len(categories) or not metadata.get("salt"):
        raise ValueError("Weirdo categories and family-split policy required")
    if set(limits) - {"train", "validation", "test"} or not limits:
        raise ValueError("explicit per-split selection limits required")
    if any(type(count) is not int or count < 0 for count in limits.values()):
        raise ValueError("selection limits must be nonnegative integers")
    with np.load(archive_path, allow_pickle=False) as archive:
        for offset, (source_split, split) in enumerate(
            (
                ("train", "train"),
                ("val", "validation"),
                ("test", "test"),
            )
        ):
            requested = limits.get(split, 0)
            if not requested:
                continue
            peptides, masks, groups = (
                archive[f"{source_split}_{suffix}"] for suffix in ("kmers", "masks", "groups")
            )
            if not (
                len(peptides) == len(masks) == len(groups) == metadata["sample_sizes"][source_split]
            ):
                raise ValueError("Weirdo split arrays disagree with the pinned metadata")
            if requested > len(peptides):
                raise ValueError("selection exceeds source split size")
            selected = sorted(random.Random(seed + offset).sample(range(len(peptides)), requested))
            for index in selected:
                peptide = str(peptides[index])
                mask = int(masks[index])
                if len(peptide) != metadata["k"] or not 0 < mask < (1 << len(categories)):
                    raise ValueError("invalid source k-mer length or category bitmask")
                labels = {
                    f"peptide_source:{category}": "present" if mask & (1 << position) else "absent"
                    for position, category in enumerate(categories)
                }
                measurement_id = f"k{metadata['k']}:{peptide}"
                row = Observation(
                    "weirdo:" + hashlib.sha256(measurement_id.encode()).hexdigest(),
                    "weirdo_swissprot_reference",
                    "identity",
                    peptide,
                    split=split,
                    group_id=f"weirdo:{archive_sha256}:{source_split}:bucket:{int(groups[index])}",
                    identity_labels=labels,
                    provenance={
                        "measurement_lineage": {
                            "namespace": f"swissprot:{reference_sha256}",
                            "measurement_id": measurement_id,
                        },
                        "source_archive_sha256": archive_sha256,
                        "source_metadata_sha256": metadata_sha256,
                        "source_reference_sha256": reference_sha256,
                        "source_split": source_split,
                        "source_row": index,
                        "source_family_bucket": int(groups[index]),
                        "family_split_salt": metadata["salt"],
                        "selection": {
                            "policy": "uniform_without_replacement",
                            "seed": seed + offset,
                            "requested": requested,
                            "population": len(peptides),
                        },
                        "label_definition": (
                            "membership in a pinned reference category; categories may overlap"
                        ),
                        "independence": (
                            "source-declared family split; adapter verifies bytes and schema only"
                        ),
                    },
                )
                row.validate()
                yield row
