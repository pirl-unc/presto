"""One-vote, search-qualified proteomics records from frozen Hitlist evidence.

These binary labels are proxies for detectability within one documented search.
They are neither biological replicate counts nor intrinsic detectability scores.
"""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import sqlite3
from tempfile import TemporaryDirectory

from .bulk_ms import BulkMSRecord, machinery_for_enzyme
from .frozen_hitlist import _sha256
from .vocab import is_encodable_sequence

MINIMUM_SCOPED_HITLIST_VERSION = (1, 75, 0)

EVIDENCE = {
    "By MS/MS": "scoped_msms",
    "By matching": "scoped_matching",
    "not observed in scope": "scoped_unobserved",
}
POLICY = {
    "version": 2,
    "enzyme_contract": "Trypsin/P routes to trypsin_p; ordinary Trypsin stays proline-blocked",
    "endpoint": "observed_in_selected_search_scope",
    "units": "binary",
    "censoring": "not_applicable",
    "evidence_role": "proxy",
    "vote": "one peptide per verified search scope; collapse before capping",
    "ambiguous_occurrences": "mask parent and flanks; all occurrences remain in source sidecar",
    "depth": "missing; no comparable depth series",
    "biological_replicates": "missing; one aggregate scope is not a biological replicate",
    "excision": "no labels or generated controls supplied by this adapter",
    "independence": "not established; one experiment, unresolved donors and specimens",
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_json(path):
    return json.loads(Path(path).read_text())


def _admit_bundle(directory, expected_sha256, cache_dir):
    import hitlist
    from datacache import install_bundle
    from hitlist.downloads import data_dir

    from .hitlist_source import _parse_version

    if (_parse_version(hitlist.__version__) or ()) < MINIMUM_SCOPED_HITLIST_VERSION:
        wanted = ".".join(map(str, MINIMUM_SCOPED_HITLIST_VERSION))
        raise RuntimeError(f"Scoped proteomics requires Hitlist >={wanted}")
    from hitlist import load_pxd013455_bundle, verify_pxd013455_bundle

    root = Path(cache_dir) if cache_dir is not None else data_dir() / "presto-scoped-proteomics"
    if directory is None:
        directory = load_pxd013455_bundle()
    elif not expected_sha256:
        raise ValueError("An explicit scoped bundle requires its manifest SHA-256")
    directory = Path(directory).resolve()
    digest = _sha256(directory / "manifest.json")
    manifest = verify_pxd013455_bundle(directory, expected_sha256=expected_sha256 or digest)
    # Local inputs acquire the same immutable, shared lifetime as downloaded ones.
    assets = {
        name: {"path": str(directory / name), "sha256": pin["sha256"], "size": pin["size_bytes"]}
        for name, pin in manifest["artifacts"].items()
    }
    assets["manifest.json"] = {
        "path": str(directory / "manifest.json"),
        "sha256": digest,
        "size": (directory / "manifest.json").stat().st_size,
    }
    paths = install_bundle(
        root / "inputs" / digest, assets, download_options={"show_progress": False}
    )
    frozen = Path(paths["manifest.json"]).parent
    verify_pxd013455_bundle(frozen, expected_sha256=digest)
    return frozen, digest, manifest, root


def _batches(path, columns=None):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=5000, columns=columns):
        yield batch.to_pylist()


def _projection(directory, digest, manifest, root):
    """Collapse globally in bounded scratch storage, then publish through datacache."""
    from datacache import materialize
    from presto import __version__

    registry = _read_json(directory / "source_registry.json")
    candidate_manifest = _read_json(directory / "candidates.manifest.json")
    source_policy = _read_json(directory / "label_policy.json")
    scope = registry["scope"]
    search_id = candidate_manifest["search_space_id"]
    reference_sha256 = manifest["artifacts"]["searched.fasta"]["sha256"]
    if (
        registry["dataset_id"] != manifest["dataset_id"]
        or candidate_manifest["scope"] != scope
        or candidate_manifest["artifact"]
        != {"name": "candidates.parquet", **manifest["artifacts"]["candidates.parquet"]}
        or candidate_manifest["search_reference"]["sha256"] != reference_sha256
        or registry["search_space"] != candidate_manifest["search_space"]
        or registry["search_space"]["fasta_sha256"] != reference_sha256
        or registry["search_space"]["enzyme"] != scope["digestion_enzyme"]
        or type(candidate_manifest["parameters"]["flank"]) is not int
        or candidate_manifest["parameters"]["flank"] < 1
        or source_policy["endpoint"] != POLICY["endpoint"]
        or source_policy["units"] != "binary"
        or source_policy["censoring"] != "not_applicable"
        or machinery_for_enzyme(scope["digestion_enzyme"]) == "unknown"
    ):
        raise ValueError("Scoped publication has inconsistent search, reference or label policy")

    def same_scope(row):
        return (
            all(row[key] == value for key, value in scope.items())
            and all(row[key] == value for key, value in registry["acquisition"].items())
            and row["search_space_id"] == search_id
        )

    options = {
        "manifest_sha256": digest,
        "policy": POLICY,
        "presto_version": __version__,
        "recipe_sha256": _sha256(Path(__file__)),
    }
    sources = {
        name: {"path": str(directory / name), "sha256": pin["sha256"], "size": pin["size_bytes"]}
        for name, pin in manifest["artifacts"].items()
    }

    def build(inputs, outputs):
        import pyarrow as pa
        import pyarrow.parquet as pq

        columns = (
            "peptide",
            "protein_id",
            "flank_n",
            "flank_c",
            "evidence_row_id",
            "scoped_peptide_id",
            "candidate_id",
            "candidate_lookup_key",
            "source_n_candidate_occurrences",
            "source_label_evidence",
        )
        schema = pa.schema(
            [(name, pa.string()) for name in columns]
            + [
                (name, pa.bool_())
                for name in ("observed", "flank_n_is_terminus", "flank_c_is_terminus")
            ]
        )
        counts = Counter(
            {
                name: 0
                for name in (
                    "n_candidate_occurrences",
                    "n_invalid_sequence_occurrences",
                    "n_ambiguous_peptides",
                    "n_peptides",
                    "n_source_observations",
                    "n_supported_parents",
                )
            }
        )
        counts_by_evidence = Counter(dict.fromkeys(EVIDENCE, 0))
        preview = []
        # The publication cache may live on a shared/network filesystem. Keep
        # disposable random-I/O SQLite state on local temporary storage; only
        # complete reusable outputs belong in the datacache generation.
        with TemporaryDirectory(prefix="presto-scoped-collapse-") as scratch:
            connection = sqlite3.connect(str(Path(scratch) / "occurrences.sqlite"))
            try:
                connection.execute("PRAGMA cache_size=-8192")
                connection.execute("PRAGMA temp_store=FILE")
                connection.execute(
                    "CREATE TABLE observations "
                    "(peptide TEXT PRIMARY KEY, identity TEXT, evidence TEXT)"
                )
                connection.execute("CREATE TABLE parents (identity TEXT PRIMARY KEY)")
                connection.execute("CREATE TABLE candidates (identity TEXT PRIMARY KEY)")
                connection.execute(
                    "CREATE TABLE peptides "
                    "(peptide TEXT PRIMARY KEY, payload TEXT, n_occurrences INTEGER)"
                )
                for batch in _batches(inputs["observations.parquet"]):
                    counts["n_source_observations"] += len(batch)
                    if any(
                        not same_scope(row)
                        or not row["evidence_row_id"]
                        or row["identification_type"] not in {"By MS/MS", "By matching"}
                        for row in batch
                    ):
                        raise ValueError("Observation has incompatible scope or evidence")
                    connection.executemany(
                        "INSERT INTO observations VALUES (?, ?, ?)",
                        [
                            (row["peptide"], row["evidence_row_id"], row["identification_type"])
                            for row in batch
                        ],
                    )
                for batch in _batches(inputs["proteins.parquet"]):
                    counts["n_supported_parents"] += len(batch)
                    if any(not same_scope(row) for row in batch):
                        raise ValueError("Parent has incompatible scope")
                    connection.executemany(
                        "INSERT INTO parents VALUES (?)", [(row["uniprot_acc"],) for row in batch]
                    )
                for batch in _batches(inputs["candidates.parquet"]):
                    for row in batch:
                        counts["n_candidate_occurrences"] += 1
                        if (
                            not same_scope(row)
                            or not row["protein_observed"]
                            or row["n_replicates_possible"] != 1
                            or row["first_seen_at_n_fractions"] is not None
                        ):
                            raise ValueError(
                                "Candidate has incompatible scope, parent support or depth contract"
                            )
                        parent = connection.execute(
                            "SELECT 1 FROM parents WHERE identity=?", (row["uniprot_acc"],)
                        ).fetchone()
                        if parent is None:
                            raise ValueError("Candidate parent lacks same-scope support")
                        identity = (
                            f'{manifest["dataset_id"]}:{row["uniprot_acc"]}:'
                            f'{row["start_position"]}:{row["end_position"]}'
                        )
                        if row["candidate_id"] != identity:
                            raise ValueError(
                                "Candidate occurrence identity disagrees with its coordinates"
                            )
                        connection.execute("INSERT INTO candidates VALUES (?)", (identity,))
                        observation = connection.execute(
                            "SELECT identity, evidence FROM observations WHERE peptide=?",
                            (row["peptide"],),
                        ).fetchone()
                        expected = observation if observation else (None, "not observed in scope")
                        if (
                            row["observed"] != bool(observation)
                            or (row["source_observation_id"], row["label_evidence"]) != expected
                            or row["label_evidence"] not in EVIDENCE
                        ):
                            raise ValueError(
                                "Candidate label disagrees with its source observation"
                            )
                        if not is_encodable_sequence(row["peptide"]):
                            counts["n_invalid_sequence_occurrences"] += 1
                            continue
                        payload = {
                            "peptide": row["peptide"],
                            "protein_id": row["uniprot_acc"],
                            "flank_n": row["n_flank"],
                            "flank_c": row["c_flank"],
                            "flank_n_is_terminus": row["start_position"] == 1,
                            "flank_c_is_terminus": row["c_flank"] == "",
                            "evidence_row_id": row["source_observation_id"] or "",
                            "scoped_peptide_id": (
                                f'{manifest["dataset_id"]}:peptide:{row["peptide"]}'
                            ),
                            "candidate_id": identity,
                            "candidate_lookup_key": row["peptide"],
                            "source_label_evidence": row["label_evidence"],
                            "observed": row["observed"],
                        }
                        connection.execute(
                            "INSERT INTO peptides VALUES (?, ?, 1) ON CONFLICT(peptide) "
                            "DO UPDATE SET n_occurrences=n_occurrences+1",
                            (row["peptide"], _json(payload)),
                        )
                    connection.commit()
                if counts["n_candidate_occurrences"] != candidate_manifest["n_candidates"]:
                    raise ValueError("Candidate census disagrees with its manifest")
                with pq.ParquetWriter(
                    outputs["records.parquet"], schema, compression="zstd"
                ) as writer:
                    cursor = connection.execute(
                        "SELECT payload, n_occurrences FROM peptides ORDER BY peptide"
                    )
                    while rows := cursor.fetchmany(5000):
                        records = []
                        for payload, n_occurrences in rows:
                            record = json.loads(payload)
                            record["source_n_candidate_occurrences"] = str(n_occurrences)
                            if n_occurrences > 1:
                                counts["n_ambiguous_peptides"] += 1
                                record["flank_n_is_terminus"] = False
                                record["flank_c_is_terminus"] = False
                                for key in ("protein_id", "flank_n", "flank_c", "candidate_id"):
                                    record[key] = ""
                            counts["n_peptides"] += 1
                            counts_by_evidence[record["source_label_evidence"]] += 1
                            if counts_by_evidence[record["source_label_evidence"]] <= 20:
                                preview.append(record.copy())
                            records.append(record)
                        writer.write_table(pa.Table.from_pylist(records, schema=schema))
            finally:
                connection.close()
        coverage = {**dict(counts), "n_peptides_by_evidence": dict(counts_by_evidence)}
        Path(outputs["coverage.json"]).write_text(
            json.dumps(coverage, sort_keys=True, indent=2) + "\n"
        )
        with Path(outputs["inspection.tsv"]).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=schema.names, delimiter="\t")
            writer.writeheader()
            writer.writerows(preview)
        Path(outputs["label_policy.json"]).write_text(
            json.dumps({"consumer": POLICY, "source": source_policy}, sort_keys=True, indent=2)
            + "\n"
        )
        Path(outputs["README.md"]).write_text(
            "# Presto scoped proteomics projection\n\n"
            f"Input manifest SHA-256: `{digest}`. One row per peptide in this search.\n\n"
            "`records.parquet` preserves scoped peptide and positive observation IDs. "
            "`candidate_lookup_key` joins the frozen source candidates by peptide; "
            "all parent occurrences remain there. Multiple occurrences mask model-facing "
            "parent/flanks and candidate_id; the count remains explicit.\n\n"
            "`inspection.tsv` shows the first 20 peptides per evidence category (a preview, "
            "not a representative sample). `coverage.json` counts full uncapped support; "
            "`label_policy.json` preserves "
            "source and consumer policies. Source registry holds citations, cells, conditions, "
            "acquisition, reference hashes and unresolved donor/specimen identities.\n\n"
            "MS/MS, matching and scoped absence remain distinct proxy evidence. No depth, "
            "biological replicate or intrinsic detectability claim. No partition assignment "
            "or independence qualification is supplied by this projection.\n"
        )

    paths = materialize(
        root / "projections" / hashlib.sha256(_json(options).encode()).hexdigest(),
        sources,
        transform={"version": "presto-scoped-bulk-v1", "options": options},
        outputs={
            name: {}
            for name in (
                "records.parquet",
                "coverage.json",
                "label_policy.json",
                "inspection.tsv",
                "README.md",
            )
        },
        builder=build,
        download_options={"show_progress": False},
    )
    return {name: Path(path) for name, path in paths.items()}, registry, candidate_manifest


def load_scoped_bulk_records(
    bundle_dir=None, *, bundle_sha256=None, max_records=None, seed=0, cache_dir=None
):
    """Admit the frozen PXD013455 scope; cap only after global occurrence collapse.

    Default input is Hitlist's public pinned download. Local inputs require an
    explicit manifest pin. Returned stats include immutable sidecar locations,
    hashes, full/admitted support and limitations. This API assigns no partitions.
    """
    if max_records is not None and (type(max_records) is not int or max_records < 1):
        raise ValueError("max_records must be a positive integer or None")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    directory, digest, manifest, root = _admit_bundle(bundle_dir, bundle_sha256, cache_dir)
    paths, registry, candidates = _projection(directory, digest, manifest, root)
    selected, rng = [], random.Random(seed)
    n_seen = 0
    for batch in _batches(paths["records.parquet"]):
        for row in batch:
            n_seen += 1
            if max_records is None or len(selected) < max_records:
                selected.append(row)
            else:
                index = rng.randrange(n_seen)
                if index < max_records:
                    selected[index] = row
    records = []
    for row in sorted(selected, key=lambda value: value["scoped_peptide_id"]):
        observed = row.pop("observed")
        records.append(
            BulkMSRecord(
                **row,
                observed=observed,
                detectability_label=float(observed),
                machinery=machinery_for_enzyme(registry["scope"]["digestion_enzyme"]),
                cell_line=registry["scope"]["cell_line_name"],
                n_fractions=registry["scope"]["n_fractions_in_run"],
                n_replicates_detected=None,
                source=manifest["dataset_id"],
                source_bundle_id=f"sha256:{digest}",
                source_search_space_id=candidates["search_space_id"],
                source_protocol_id=registry["scope"]["protocol_id"],
                source_reference_sha256=candidates["search_reference"]["sha256"],
                source_parent_assignment_basis=registry["acquisition"]["protein_observation_basis"],
                source_endpoint=POLICY["endpoint"],
                source_units="binary",
                reference_iri=registry["citations"]["reanalysis"]["url"],
                pmid=registry["citations"]["reanalysis"]["pmid"],
                doi=registry["citations"]["reanalysis"]["doi"],
                acquisition={**registry["acquisition"], **registry["scope"]},
            )
        )
    stats = {
        "n_records": len(records),
        "n_detectability_labels": len(records),
        "n_excision_labels": 0,
        "n_depth_labels": 0,
        "n_dropped_by_cap": n_seen - len(records),
        "sampling_seed": seed,
        "n_records_by_evidence": dict(Counter(record.source_label_evidence for record in records)),
        "full_coverage": _read_json(paths["coverage.json"]),
        "label_policy": POLICY,
        "frozen_bundle": {"directory": str(directory), "manifest_sha256": digest, **manifest},
        "projection": {
            "directory": str(paths["records.parquet"].parent),
            "artifacts": {
                name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
                for name, path in paths.items()
            },
        },
        "source_registry": registry,
    }
    return records, stats
