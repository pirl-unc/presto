"""Frozen peptide membership for an explicitly limited within-search evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path

import pandas as pd

from presto import __version__
from .frozen_hitlist import _sha256
from .frozen_partitions import PARTITIONS, _assign_components, _digest


@dataclass
class ScopedPartitions:
    membership: pd.DataFrame
    receipt: dict

    def finalize(self, dataset):
        """Require exact conversion of frozen records; never silently split again."""
        expected = self.membership.set_index("sample_id").to_dict("index")
        indices = {name: [] for name in PARTITIONS}
        seen = set()
        for index, sample in enumerate(dataset.samples):
            row = expected.get(sample.sample_id)
            if row is None or sample.sample_id in seen:
                raise ValueError("Scoped dataset added, duplicated or replaced frozen membership")
            seen.add(sample.sample_id)
            fields = (
                "peptide",
                "source_partition",
                "source_bundle_id",
                "scoped_peptide_id",
                "candidate_id",
                "candidate_lookup_key",
                "evidence_row_id",
                "source_label_evidence",
                "source_search_space_id",
                "source_protocol_id",
                "source_reference_sha256",
                "source_n_candidate_occurrences",
                "machinery",
                "flank_n_is_terminus",
                "flank_c_is_terminus",
            )
            if (
                any(getattr(sample, field) != row[field] for field in fields)
                or sample.ms_detectability_label != row["detectability_label"]
                or sample.excision_label is not None
                or sample.synthetic_kind
                or sample.assay_group != "bulk_ms"
                or (sample.source_protein or "") != row["protein_id"]
                or (sample.flank_n or "") != row["flank_n"]
                or (sample.flank_c or "") != row["flank_c"]
                or sample.enzymatic_digest != row["machinery"]
                or sample.peptide_source != "protein"
                or sample.mhc_a
                or sample.mhc_b
                or sample.mhc_class is not None
            ):
                raise ValueError("Scoped dataset changed frozen identity, partition or supervision")
            indices[row["source_partition"]].append(index)
        if seen != set(expected):
            raise ValueError("Scoped dataset omitted frozen membership")
        return indices, self.receipt


def prepare_scoped_partitions(
    records,
    stats,
    *,
    seed,
    val_frac,
    test_frac,
    assignment_file=None,
    assignment_sha256=None,
    cache_dir=None,
):
    """Publish membership and honest overlap audits; biological independence is unresolved."""
    import datacache
    from hitlist.downloads import data_dir

    source = stats["frozen_bundle"]
    source_dir = Path(source["directory"])
    datacache.validate_file(source_dir / "manifest.json", source["manifest_sha256"])
    candidate_manifest_path = source_dir / "candidates.manifest.json"
    datacache.validate_file(
        candidate_manifest_path, source["artifacts"]["candidates.manifest.json"]["sha256"]
    )
    candidate_manifest = json.loads(candidate_manifest_path.read_text())
    expected_scope = {
        "source_search_space_id": candidate_manifest["search_space_id"],
        "source_protocol_id": stats["source_registry"]["scope"]["protocol_id"],
        "source_reference_sha256": candidate_manifest["search_reference"]["sha256"],
    }
    source_bundle_id = "sha256:" + source["manifest_sha256"]
    columns = (
        "peptide",
        "scoped_peptide_id",
        "candidate_id",
        "candidate_lookup_key",
        "evidence_row_id",
        "source_label_evidence",
        "source_search_space_id",
        "source_protocol_id",
        "source_reference_sha256",
        "protein_id",
        "detectability_label",
        "source_bundle_id",
        "reference_iri",
        "pmid",
        "doi",
        "source_endpoint",
        "source_units",
        "cell_line",
        "n_fractions",
        "source_n_candidate_occurrences",
        "machinery",
        "flank_n",
        "flank_c",
        "flank_n_is_terminus",
        "flank_c_is_terminus",
    )
    rows = []
    for record in records:
        if (
            record.source_bundle_id != source_bundle_id
            or any(getattr(record, name) != value for name, value in expected_scope.items())
            or not all(
                (
                    record.scoped_peptide_id,
                    record.source_search_space_id,
                    record.source_protocol_id,
                    record.source_reference_sha256,
                )
            )
            or record.detectability_label not in (0.0, 1.0)
            or record.detectability_label != float(record.observed)
            or bool(record.evidence_row_id) != record.observed
            or record.excision_label is not None
            or record.generated_kind
            or record.source_label_evidence
            not in ("By MS/MS", "By matching", "not observed in scope")
            or (record.source_label_evidence == "not observed in scope") != (not record.observed)
        ):
            raise ValueError(
                "Scoped partitions require complete, consistent candidate/measurement lineage"
            )
        row = {name: getattr(record, name) for name in columns}
        row["source_censoring"] = stats["label_policy"]["censoring"]
        row["assay_type"] = "shotgun_mass_spectrometry"
        row["mhc_allele_set"] = "not_applicable"
        row["sample_id"] = f"scoped:{source_bundle_id}:{record.scoped_peptide_id}"
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty or frame.peptide.duplicated().any() or frame.sample_id.duplicated().any():
        raise ValueError("Scoped partitions require unique, nonempty peptide membership")
    if any(
        frame[name].nunique() != 1
        for name in ("source_search_space_id", "source_protocol_id", "source_reference_sha256")
    ):
        raise ValueError("Scoped partitions require one verified search/protocol/reference")
    frame = frame.sort_values("sample_id").reset_index(drop=True)
    projection_path = Path(stats["projection"]["directory"]) / "records.parquet"
    datacache.validate_file(
        projection_path, stats["projection"]["artifacts"]["records.parquet"]["sha256"]
    )
    selected_projection = pd.read_parquet(
        projection_path, filters=[("scoped_peptide_id", "in", frame.scoped_peptide_id.tolist())]
    ).set_index("scoped_peptide_id")
    if selected_projection.index.duplicated().any() or set(selected_projection.index) != set(
        frame.scoped_peptide_id
    ):
        raise ValueError(
            "Scoped partitions selected identities absent or duplicated in the verified projection"
        )
    for record in records:
        expected = selected_projection.loc[record.scoped_peptide_id]
        if any(getattr(record, name) != value for name, value in expected.items()):
            raise ValueError(
                "Scoped partitions changed a candidate, measured label or source junction"
            )
    assigned = _assign_components(
        [[identity] for identity in frame.sample_id],
        seed=seed,
        val_frac=val_frac,
        test_frac=test_frac,
    )
    if bool(assignment_file) != bool(assignment_sha256):
        raise ValueError("External scoped assignments require a file and its SHA-256")
    if assignment_file:
        datacache.validate_file(assignment_file, assignment_sha256)
        external = pd.read_parquet(assignment_file)
        required = {"sample_id", "source_bundle_id", "source_partition", "peptide"}
        if (
            not required <= set(external)
            or external.sample_id.duplicated().any()
            or set(external.sample_id) != set(frame.sample_id)
            or not external.source_bundle_id.eq(source_bundle_id).all()
            or not external.source_partition.isin(PARTITIONS).all()
        ):
            raise ValueError("External scoped assignments conflict with selected source membership")
        indexed = external.set_index("sample_id")
        if any(indexed.loc[row.sample_id, "peptide"] != row.peptide for row in frame.itertuples()):
            raise ValueError("External scoped assignments changed peptide identities")
        assigned = {identity: (row.source_partition, "") for identity, row in indexed.iterrows()}
    frame["source_partition"] = frame.sample_id.map(lambda identity: assigned[identity][0])
    wanted = {"train", "val", "test"} if test_frac else {"train", "val"}
    if set(frame.source_partition) != wanted:
        raise ValueError("Scoped assignments must populate every requested partition")
    assignments = frame[["sample_id", "source_bundle_id", "source_partition", "peptide"]]
    selected_rows = frame.drop(columns="source_partition").to_dict("records")
    contract = {
        "algorithm": "scoped-peptide-membership-v1",
        "presto_version": __version__,
        "source_bundle_id": source_bundle_id,
        "policy": "peptide_disjoint_within_search",
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "selection_sha256": _digest(selected_rows),
        "external_assignment_sha256": assignment_sha256,
        "projection_artifacts": stats["projection"]["artifacts"],
        "label_policy": stats["label_policy"],
        "recipe_sha256": _sha256(Path(__file__)),
        "generated_controls": "none; absences are scoped candidates without invented assay IDs",
    }
    partitions = {name: frame[frame.source_partition.eq(name)] for name in sorted(wanted)}
    pairwise = {}
    for left, right in combinations(partitions, 2):
        pairwise[f"{left}/{right}"] = {
            f"n_shared_{name}": len(
                (set(partitions[left][column]) - {""}) & (set(partitions[right][column]) - {""})
            )
            for name, column in (
                ("peptides", "peptide"),
                ("candidate_ids", "candidate_id"),
                ("measured_observation_ids", "evidence_row_id"),
                ("known_proteins", "protein_id"),
                ("search_scopes", "source_search_space_id"),
            )
        }
    if any(item["n_shared_measured_observation_ids"] for item in pairwise.values()):
        raise ValueError("Scoped partitions reuse a measured observation across partitions")
    audit = {
        "claim": "peptide-disjoint evaluation within one captured search",
        "peptide_disjoint": True,
        "pairwise": pairwise,
        "experimental_independence": "not established; all partitions reuse this experiment",
        "donor_independence": "inconclusive; unresolved source identity",
        "specimen_independence": "inconclusive; unresolved source identity",
        "protein_independence": "not required; known overlap reported, ambiguous parents masked",
    }
    coverage = {
        "n_selected_peptides": len(frame),
        "n_dropped_by_cap": stats["n_dropped_by_cap"],
        "full_projection": stats["full_coverage"],
        "partitions": {
            name: {
                "n_peptides": len(part),
                "n_measured_observations": int(part.evidence_row_id.ne("").sum()),
                "n_unknown_parents": int(part.protein_id.eq("").sum()),
                "n_peptides_by_evidence": dict(Counter(part.source_label_evidence)),
            }
            for name, part in partitions.items()
        },
    }
    root = Path(cache_dir) if cache_dir is not None else data_dir() / "presto-scoped-partitions"
    sources = {
        "source-manifest.json": {
            "path": str(Path(source["directory"]) / "manifest.json"),
            "sha256": source["manifest_sha256"],
        }
    }
    if assignment_file:
        sources["external.parquet"] = {
            "path": str(Path(assignment_file).resolve()),
            "sha256": assignment_sha256,
        }

    def builder(inputs, outputs):
        assignments.to_parquet(outputs["assignments.parquet"], index=False)
        frame.to_parquet(outputs["membership.parquet"], index=False)
        frame.to_csv(outputs["membership.tsv"], sep="\t", index=False)
        pd.DataFrame(columns=["generated_id", "parent_id", "parent_role"]).to_parquet(
            outputs["parent_graph.parquet"], index=False
        )
        for name, data in {
            "policy.json": contract,
            "coverage.json": coverage,
            "leakage_audit.json": audit,
            "source_registry.json": stats["source_registry"],
        }.items():
            Path(outputs[name]).write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
        Path(outputs["README.md"]).write_text(
            "# Frozen within-search membership\n\n"
            "membership.tsv/parquet retain the selected peptides, actual partition assignments, "
            "candidate identities, measured IDs where available and citations. policy.json pins "
            "the source, projection, selection and label policy. coverage.json separates MS/MS, "
            "matching and absence proxies. leakage_audit.json records known protein overlap and "
            "the shared experiment; donors/specimens remain unresolved. No MHC restriction "
            "applies to this shotgun search. No generated controls are present, so the parent "
            "graph is empty. This is not an independent-experiment or training-readiness claim.\n"
        )

    paths = datacache.materialize(
        root / _digest(contract),
        sources,
        transform={"version": "presto-scoped-partitions-v1", "options": contract},
        outputs={
            name: {}
            for name in (
                "assignments.parquet",
                "membership.parquet",
                "membership.tsv",
                "parent_graph.parquet",
                "policy.json",
                "coverage.json",
                "leakage_audit.json",
                "source_registry.json",
                "README.md",
            )
        },
        builder=builder,
    )
    receipt = {
        **contract,
        "directory": str(Path(paths["policy.json"]).parent),
        "coverage": coverage,
        "assignment_sha256": _sha256(Path(paths["assignments.parquet"])),
        "membership_sha256": _sha256(Path(paths["membership.parquet"])),
        "parent_graph_sha256": _sha256(Path(paths["parent_graph.parquet"])),
        "artifacts": {
            name: {"sha256": _sha256(Path(path)), "size_bytes": Path(path).stat().st_size}
            for name, path in paths.items()
        },
    }
    for record in records:
        record.source_partition = assigned[f"scoped:{source_bundle_id}:{record.scoped_peptide_id}"][
            0
        ]
    return ScopedPartitions(frame, receipt)
