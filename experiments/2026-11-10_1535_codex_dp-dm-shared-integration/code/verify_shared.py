"""Replay native source candidates through real shared Hitlist/Presto paths."""

import argparse
import csv
import gzip
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    expected_native = "daaa04962f690a184c553ac8ec55c8545f40bdfbecac61ce89a8997d94124beb"
    expected_catalog = (
        "ddc3493c64f4328de174a6c2cda1f007ec2f93a1ef649add25d2ef9c1720add1"
    )
    assert sha(args.native) == expected_native and sha(args.catalog) == expected_catalog
    args.output.mkdir(parents=True, exist_ok=False)
    index = args.output / "source-only-index"
    index.mkdir()
    os.environ["HITLIST_DATA_DIR"] = str(index)

    import hitlist
    from hitlist.builder import _OBSERVATIONS_ARTIFACT_VERSION
    from hitlist.export import generate_training_table
    from hitlist.provenance import ContributorCollector, file_digest, load_contributors
    from hitlist.supplement import load_supplementary_manifest, scan_supplementary
    from presto.data.modular_hitlist import HumanMHCCatalog
    from presto.data.modular_hitlist_ms import ms_observation, study_factor_vocabulary

    entries = [e for e in load_supplementary_manifest() if e["pmid"] == 33025005]
    assert len(entries) == 10
    with ContributorCollector(scratch_dir=args.output) as collector:
        frame = scan_supplementary(
            entries=entries, provenance=collector, allow_download=False
        )
        frame["source"] = "supplement"
        assert len(frame) == 21547 and frame.provenance_id.nunique() == len(frame)
        frame.to_parquet(index / "observations.parquet", index=False)
        for name in ("binding", "other_assays"):
            frame.iloc[:0].to_parquet(index / f"{name}.parquet", index=False)
        contract = collector.write([frame], index / "observation_contributors.parquet")
    contract["indexes"] = {
        name: file_digest(index / f"{name}.parquet")
        for name in ("observations", "binding", "other_assays")
    }
    write(
        index / "observations_meta.json",
        {
            "artifact_version": _OBSERVATIONS_ARTIFACT_VERSION,
            "provenance": contract,
            "scope": "Isolated PXD017154 source-only verification; not the user's full Hitlist dataset",
        },
    )
    exported = generate_training_table(include_evidence="ms")
    exported.to_parquet(args.output / "shared-export.parquet", index=False)
    assert len(exported) == len(frame)
    assert exported.evidence_row_id.nunique() == len(exported)
    assert exported.sample_attribution.eq("curated_sample_label").all()
    assert exported.provenance_status.eq("indexed").all()
    assert exported.experimental_origin_ids.eq('["origin:PXD017154"]').all()
    assert exported.specimen_status.eq("unknown").all()
    assert exported.donor_status.eq("unknown").all()
    assert (
        exported.quantitative_value.isna().all() and not exported.is_binding_assay.any()
    )

    with gzip.open(args.native, "rt") as stream:
        originals = {
            (r["sample"]["biological_sample"], r["peptide"]): r
            for r in map(json.loads, stream)
        }
    candidates = {k: v for k, v in originals.items() if v["source_qualified_candidate"]}
    links = load_contributors(exported.provenance_id.tolist())
    assert len(links) == len(candidates)
    original_fields = {
        row.provenance_id: json.loads(row.original_fields) for row in links.itertuples()
    }
    seen = set()
    for row in exported.itertuples():
        key = (row.sample_label, row.peptide)
        assert key in candidates and key not in seen
        seen.add(key)
        source = candidates[key]
        fields = original_fields[row.provenance_id]
        assert json.loads(fields["native_psm_support"]) == source["supporting_psms"]
        assert (
            json.loads(fields["native_protein_annotations"])
            == source["protein_annotations"]
        )
        assert json.loads(fields["source_acquisitions"]) == source["runs"]
        assert json.loads(fields["source_sample_metadata"]) == source["sample"]
        assert fields["native_confidence"] == "high"
        assert fields["native_target_fpr_high"] == "0.01"
        assert fields["observed_in_no_dp_control"] == "false"
        assert fields["has_ptm"] == "false" and not fields["peptide_modifications"]
        # Pandas' nullable boolean scalar is numpy.bool, not the Python singleton.
        # Comparison still rejects missing values rather than coercing them.
        assert row.has_ptm == False and not row.peptide_modifications
        assert row.condition_transduction == (
            "HLA-DM" if source["sample"]["dm_state"] == "present" else "none"
        )
    assert seen == set(candidates)

    with args.catalog.open() as stream:
        catalog = HumanMHCCatalog(csv.DictReader(stream))
    records = exported.astype(object).where(exported.notna(), None).to_dict("records")
    vocabulary = study_factor_vocabulary(records)
    source_sha = sha(args.output / "shared-export.parquet")
    states = Counter()
    pairs = Counter()
    output_observations = args.output / "presto-source-observations.jsonl.gz"
    with gzip.open(output_observations, "wt") as stream:
        for ordinal, raw in enumerate(records):
            obs = ms_observation(
                raw,
                catalog,
                source_sha256=source_sha,
                ordinal=ordinal,
                study_factors=vocabulary,
            )
            source = candidates[(raw["sample_label"], raw["peptide"])]
            actual_state = obs.context.interventions["transduction:HLA-DM"].state
            assert actual_state == source["sample"]["dm_state"]
            assert obs.split == "predict" and obs.group_id == "unpartitioned"
            assert obs.provenance["partition_status"].startswith("unassigned;")
            obs.provenance["training_eligible"] = False
            assert len(obs.mhc) == 1 and obs.value == 1.0
            assert not obs.context.components and not obs.context.processing_pathways
            states[actual_state] += 1
            pairs[raw["mhc_allele_set"]] += 1
            stream.write(
                json.dumps(obs.to_dict(), sort_keys=True, allow_nan=False) + "\n"
            )
    summary = {
        "status": "passed",
        "n_source_candidates": len(candidates),
        "n_shared_export_rows": len(exported),
        "n_independently_verified_contributor_links": len(links),
        "n_distinct_peptides": exported.peptide.nunique(),
        "n_training_eligible_rows": 0,
        "dm_context_counts": dict(states),
        "mhc_pair_counts": dict(pairs),
        "sample_counts": exported.groupby("sample_label").size().to_dict(),
        "hitlist_source": hitlist.__file__,
        "native_input_sha256": expected_native,
        "catalog_sha256": expected_catalog,
        "export": {
            "path": str(args.output / "shared-export.parquet"),
            "sha256": source_sha,
        },
        "observations": {
            "path": str(output_observations),
            "sha256": sha(output_observations),
        },
        "seconds": time.perf_counter() - started,
        "pending": [
            "prior-cohort source/peptide overlap",
            "prospective comparison and training",
        ],
    }
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
