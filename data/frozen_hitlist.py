"""Verified Hitlist training artifacts shared through datacache.

Only manifest identity and observation IDs travel with model records. The
immutable identity/contributor sidecars remain the authority for source and
reviewed biological lineage; none of these identifiers are model features.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MINIMUM_FROZEN_HITLIST_VERSION = (1, 74, 0)


def require_frozen_hitlist_version(raw: str | None) -> None:
    """Frozen admission requires the coherent-generation publication contract."""
    from .hitlist_source import _parse_version

    version = _parse_version(raw)
    if version is None or version < MINIMUM_FROZEN_HITLIST_VERSION:
        wanted = ".".join(map(str, MINIMUM_FROZEN_HITLIST_VERSION))
        raise RuntimeError(
            f"Frozen Hitlist inputs require Hitlist >={wanted}; found {raw!r}. "
            "Earlier versions do not guarantee coherent dataset generations."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenHitlistBundle:
    directory: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    source: str

    @property
    def bundle_id(self) -> str:
        return f"sha256:{self.manifest_sha256}"

    def receipt(self) -> dict[str, Any]:
        """Portable hashes plus the resolved shared location for human inspection."""
        return {
            "bundle_id": self.bundle_id,
            "manifest_sha256": self.manifest_sha256,
            "directory": str(self.directory),
            "source": self.source,
            "hitlist_version": self.manifest["hitlist_version"],
            "artifacts": self.manifest["artifacts"],
            "training_options": self.manifest["training_options"],
            "input_validation": self.manifest["input_validation"],
            "coverage": self.manifest["coverage"],
        }

    def identities(self):
        import pandas as pd

        return pd.read_parquet(self.directory / self.manifest["artifacts"]["identities"]["file"])

    def lineage_registry(self) -> dict[str, Any]:
        filename = self.manifest["artifacts"]["lineage"]["file"]
        return json.loads((self.directory / filename).read_text())


def load_frozen_hitlist_bundle(
    directory, *, expected_sha256: str | None = None, cache_dir=None
) -> FrozenHitlistBundle:
    """Verify and share an immutable local bundle, without consulting live indexes.

    The input can already be a datacache bundle directory. Its manifest hash
    defines identity, so identical bundles from different locations reuse the
    same shared copy. A damaged installed copy fails explicitly. Downloads can
    first be installed by Hitlist/datacache with their published hashes; this
    function never follows unpinned URLs or silently regenerates input data.
    """
    try:
        import hitlist
        from datacache import inspect_bundle, install_bundle
        from hitlist import verify_training_bundle
        from hitlist.downloads import data_dir
    except ImportError as exc:
        raise ImportError(
            "Frozen Hitlist inputs require presto[hitlist] with Hitlist >=1.74.0"
        ) from exc

    require_frozen_hitlist_version(getattr(hitlist, "__version__", None))
    source = Path(directory).resolve()
    manifest_path = source / "manifest.json"
    digest = _sha256(manifest_path)
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError("Hitlist bundle manifest does not match the pinned SHA-256")
    manifest = verify_training_bundle(source)
    validation = manifest.get("input_validation", {})
    if (
        validation.get("require_current_indexes") is not True
        or validation.get("observations") != "current"
        or validation.get("mappings") not in {"current", "not_used"}
    ):
        raise ValueError("Training requires a Hitlist bundle created from current verified indexes")
    # Metadata and payload checks must refer to the same manifest bytes.
    if _sha256(manifest_path) != digest:
        raise ValueError("Hitlist bundle manifest changed during verification")
    root = Path(cache_dir) if cache_dir is not None else data_dir() / "training-bundles"
    store = root / digest
    state = inspect_bundle(store)
    if state.status == "available":
        installed = Path(state.bundle)
    elif state.status == "missing":
        assets = {
            value["file"]: {
                "path": str((source / value["file"]).resolve()),
                "sha256": value["sha256"],
                "size": value["size_bytes"],
            }
            for value in manifest["artifacts"].values()
        }
        assets["manifest.json"] = {
            "path": str(manifest_path),
            "sha256": digest,
            "size": manifest_path.stat().st_size,
        }
        paths = install_bundle(store, assets, download_options={"show_progress": False})
        installed = Path(paths["manifest.json"]).parent
    else:
        raise ValueError(
            f"Invalid cached Hitlist bundle at {store}: {state.error}"
        ) from state.error
    if _sha256(installed / "manifest.json") != digest:
        raise ValueError("Cached Hitlist bundle has a different manifest identity")
    installed_manifest = verify_training_bundle(installed)
    return FrozenHitlistBundle(installed, digest, installed_manifest, str(source))


def training_frames(bundle: FrozenHitlistBundle, *, include_flanks: bool, **filters):
    """Project and filter frozen observations without consulting mutable indexes.

    Restriction-set matching follows Hitlist's recorded restriction, not donor
    typing or its larger candidate-allele set. ``species`` means MHC species.
    """
    import pandas as pd
    import pyarrow.parquet as pq
    from hitlist.curation import mhc_class_spellings, normalize_species, resolve_allele_identity

    from .hitlist_source import PROTEIN_MAPPING_COLUMNS, assert_columns_present, training_columns

    path = bundle.directory / bundle.manifest["artifacts"]["training"]["file"]
    available = set(pq.read_schema(path).names)
    required = {"evidence_kind", "evidence_row_id", "provenance_id", "provenance_status"}
    if filters.get("species") is not None:
        required.add("mhc_species")
    if not required <= available:
        raise ValueError(f"Frozen Hitlist table lacks columns: {sorted(required - available)}")
    columns = required | set(training_columns("binding", include_flanks=include_flanks))
    columns.update(training_columns("ms", include_flanks=include_flanks))
    frame = pd.read_parquet(path, columns=sorted(columns & available))
    # Check the schema before filtering: an empty selection must not hide a
    # missing required field in a family present in the frozen source.
    for family in ("binding", "ms"):
        if frame.evidence_kind.eq(family).any():
            assert_columns_present(frame, family, include_flanks=include_flanks)
    observations = frame.drop(
        columns=sorted(set(frame) & PROTEIN_MAPPING_COLUMNS)
    ).drop_duplicates()
    if observations.evidence_row_id.duplicated().any():
        raise ValueError("Frozen Hitlist observation has conflicting assay or source context")
    if not include_flanks:
        # Omitting mapping columns must also undo their multiplicity. Each
        # observation retains one vote even when its source peptide maps twice.
        frame = observations
    if filters.get("mhc_class") is not None:
        frame = frame[frame.mhc_class.isin(mhc_class_spellings(filters["mhc_class"]))]
    if filters.get("species") is not None:
        frame = frame[frame.mhc_species.eq(normalize_species(filters["species"]))]
    if filters.get("mhc_allele") is not None:
        values = filters["mhc_allele"]
        values = [values] if isinstance(values, str) else values
        wanted = {resolve_allele_identity(value) for value in values} - {""}
        if not wanted:
            raise ValueError("mhc_allele filter requires at least one usable allele")
        matching = {
            value
            for value in frame.mhc_restriction.dropna().unique()
            if wanted.intersection(resolve_allele_identity(value).split(";"))
            or resolve_allele_identity(value) in wanted
        }
        frame = frame[frame.mhc_restriction.isin(matching)]
    lengths = frame.peptide.astype("string").str.len()
    keep = pd.Series(True, index=frame.index)
    if filters.get("length_min") is not None:
        keep &= lengths.ge(filters["length_min"]).fillna(False)
    if filters.get("length_max") is not None:
        keep &= lengths.le(filters["length_max"]).fillna(False)
    frame = frame.loc[keep].copy()
    frame["source_bundle_id"] = bundle.bundle_id
    result = []
    for family in ("binding", "ms"):
        selected = frame.loc[frame.evidence_kind.eq(family)].copy()
        if selected.empty:
            # A missing evidence family is a supported empty modality. This
            # supplies an empty schema only, never values for observed rows.
            selected = selected.reindex(
                columns=sorted(
                    set(selected) | set(training_columns(family, include_flanks=include_flanks))
                )
            )
        result.append(selected)
    return tuple(result)
