#!/usr/bin/env python3
"""Verify real mhcseqs DQ/DP beta inputs through sample and batch lineage."""

import json

from presto.data.collate import PrestoCollator
from presto.data.loaders import BindingRecord, PrestoDataset
from presto.data.mhc_sequence_resolver import lookup_exact_mhc_input


def main() -> None:
    rows = []
    for allele in ("HLA-DQB1*06:02", "HLA-DPB1*02:01"):
        exact = lookup_exact_mhc_input(allele)
        if exact is None:
            raise RuntimeError(f"mhcseqs did not resolve {allele}")
        sample = PrestoDataset(
            binding_records=[
                BindingRecord(
                    peptide="PKYVKQNTLKLAT",
                    mhc_allele=allele,
                    value=75.0,
                    mhc_class="II",
                )
            ]
        )[0]
        batch = PrestoCollator()([sample])
        if exact.chain != "beta" or exact.groove1 or not exact.groove2:
            raise RuntimeError(f"Unexpected mhcseqs chain contract for {allele}: {exact}")
        if sample.mhc_a or sample.mhc_b != exact.groove2:
            raise RuntimeError(f"Model input does not preserve {allele} groove2")
        if sample.resolved_mhc_alleles != (allele,):
            raise RuntimeError(f"Sample lineage does not resolve {allele}")
        if batch.source_lineage["resolved_mhc_alleles"] != [allele]:
            raise RuntimeError(f"Batch lineage does not resolve {allele}")
        rows.append(
            {
                "query_allele": allele,
                "resolved_allele": exact.allele,
                "source": exact.source,
                "mhc_class": exact.mhc_class,
                "chain": exact.chain,
                "groove1_length": len(exact.groove1),
                "groove2_length": len(exact.groove2),
                "sample_mhc_a_length": len(sample.mhc_a),
                "sample_mhc_b_length": len(sample.mhc_b),
                "sample_resolved_mhc_alleles": list(sample.resolved_mhc_alleles),
                "batch_resolved_mhc_alleles": batch.source_lineage["resolved_mhc_alleles"],
            }
        )
    print(json.dumps({"schema_version": 1, "status": "passed", "alleles": rows}, indent=2))


if __name__ == "__main__":
    main()
