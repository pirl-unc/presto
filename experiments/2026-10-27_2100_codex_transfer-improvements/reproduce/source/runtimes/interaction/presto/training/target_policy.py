"""Executable affinity label policy and traceable preflight clipping evidence."""

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..models.affinity import (
    DEFAULT_MAX_AFFINITY_NM,
    DEFAULT_MIN_AFFINITY_NM,
    normalize_binding_target_log10,
)


@dataclass(frozen=True)
class AffinityTargetPolicy:
    """Describe and execute the historical nM-to-log10(nM) label transform."""

    minimum_nM: float = DEFAULT_MIN_AFFINITY_NM
    maximum_nM: float = DEFAULT_MAX_AFFINITY_NM

    def __post_init__(self):
        if not (
            math.isfinite(self.minimum_nM)
            and math.isfinite(self.maximum_nM)
            and 0 < self.minimum_nM <= self.maximum_nM
        ):
            raise ValueError("Affinity target bounds must be finite, positive and ordered")

    def __call__(self, value):
        return normalize_binding_target_log10(
            value,
            min_affinity_nM=self.minimum_nM,
            max_affinity_nM=self.maximum_nM,
            assume_log10=False,
        )

    def to_dict(self):
        return {
            "schema_version": 1,
            "operation": "clip_then_log10",
            "input_unit": "nM",
            "input_dtype": "float32",
            "minimum_nM": self.minimum_nM,
            "maximum_nM": self.maximum_nM,
            "target_unit": "log10(nM)",
            "qualifier_policy": "preserve_original_direction",
        }


AFFINITY_TARGET_POLICY = AffinityTargetPolicy()
COUNT_FIELDS = (
    "n_observations",
    "n_exact_observations",
    "n_left_censored_observations",
    "n_right_censored_observations",
    "n_clipped_observations",
    "n_below_minimum_observations",
    "n_above_maximum_observations",
    "n_clipped_exact_observations",
    "n_clipped_left_censored_observations",
    "n_clipped_right_censored_observations",
)
SOURCE_FIELDS = (
    "evidence_row_id",
    "source_endpoint",
    "source_value",
    "source_units",
    "source_inequality",
    "pmid",
    "doi",
    "reference_iri",
    "assay_iri",
    "source_mhc_alleles",
    "resolved_mhc_alleles",
)


def affinity_target_policies(contract):
    return {
        name: objective.target_policy
        for name, objective in contract.objectives.items()
        if objective.target_policy is not None
    }


class AffinityTargetAudit:
    """Keep bounded counters and store only clipped objective observations on disk."""

    def __init__(self, db, contract):
        self.db, self.contract = db, contract
        self.policies = affinity_target_policies(contract)
        self.counts = defaultdict(lambda: dict.fromkeys(COUNT_FIELDS, 0))
        self.db.execute("""CREATE TABLE affinity_target_changes (
            row_id INTEGER NOT NULL, objective_id TEXT NOT NULL, column_name TEXT NOT NULL,
            observation_key TEXT NOT NULL, raw_target REAL NOT NULL, effective_target REAL NOT NULL,
            qualifier INTEGER, direction TEXT NOT NULL, source_details TEXT NOT NULL,
            UNIQUE(row_id,objective_id,column_name,observation_key))""")
        self.db.execute(
            "CREATE INDEX affinity_changes_cell ON affinity_target_changes "
            "(objective_id,column_name,row_id)"
        )

    def add(
        self,
        *,
        split,
        row_id,
        objective_id,
        column,
        observation_key,
        sample,
        raw_target,
        effective_target,
        qualifier,
    ):
        policy = self.policies.get(objective_id)
        if policy is None:
            return
        below = raw_target < policy["minimum_nM"]
        above = raw_target > policy["maximum_nM"]
        clipped = below or above
        kind = (
            "left_censored" if qualifier == -1 else "right_censored" if qualifier == 1 else "exact"
        )
        for cell in ("", column) if column else ("",):
            counts = self.counts[split, objective_id, cell]
            counts["n_observations"] += 1
            counts[f"n_{kind}_observations"] += 1
            counts["n_clipped_observations"] += int(clipped)
            counts["n_below_minimum_observations"] += int(below)
            counts["n_above_maximum_observations"] += int(above)
            counts[f"n_clipped_{kind}_observations"] += int(clipped)
        if not clipped:
            return
        source = {}
        for field in SOURCE_FIELDS:
            value = getattr(sample, field, None)
            source[field] = (
                ""
                if value is None
                else (json.dumps(value) if isinstance(value, (list, tuple)) else str(value))
            )
        self.db.execute(
            "INSERT INTO affinity_target_changes VALUES (?,?,?,?,?,?,?,?,?)",
            (
                row_id,
                objective_id,
                column,
                observation_key,
                raw_target,
                effective_target,
                qualifier,
                "below_minimum" if below else "above_maximum",
                json.dumps(source, sort_keys=True),
            ),
        )

    def report(self, splits):
        rows = []
        for split in splits:
            for name in self.policies:
                objective = self.contract.objectives[name]
                columns = self.contract.outputs[objective.endpoint].columns
                for column in ("", *columns):
                    clause = " AND c.column_name=?" if column else ""
                    args = (split, name, column) if column else (split, name)
                    unique, untraceable = self.db.execute(
                        "SELECT COUNT(DISTINCT c.observation_key), "
                        "COALESCE(SUM(NOT s.traceable),0) FROM affinity_target_changes c "
                        "JOIN samples s ON s.id=c.row_id WHERE s.split=? AND c.objective_id=?"
                        + clause,
                        args,
                    ).fetchone()
                    rows.append(
                        {
                            "split": split,
                            "objective_id": name,
                            "endpoint": objective.endpoint,
                            "column": column,
                            **self.counts[split, name, column],
                            "n_clipped_unique_observations": unique,
                            "n_untraceable_clipped_observations": untraceable,
                        }
                    )
        return {
            "schema_version": 1,
            "scope": "affinity row and panel objectives",
            "policies": self.policies,
            "counts": rows,
            "interpretation": [
                "Counts are active objective exposures; aliases/panel axes are "
                "not independent evidence.",
                "Distinct clipped observations use source identity, raw collated value and "
                "qualifier; unknown identity remains explicit.",
                "Clipping is assessed on collated float32 values; "
                "original source text remains separate.",
                "Changed exact targets differ from their measurements; "
                "changed censor bounds change the constraint.",
                "The original qualifier is retained. No output performance or "
                "biological independence is established.",
            ],
        }

    def write_artifacts(self, output_dir, splits):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        report = output / "affinity_target_audit.json"
        report.write_text(json.dumps(self.report(splits), indent=2, sort_keys=True) + "\n")
        changes = output / "affinity_target_changes.tsv"
        fields = (
            "split",
            "objective_id",
            "endpoint",
            "column",
            "sample_id",
            "sample_source",
            "origin_id",
            "peptide",
            *SOURCE_FIELDS,
            "collated_raw_target_nM",
            "effective_target_log10_nM",
            "qualifier",
            "clipping_direction",
        )
        query = self.db.execute("""SELECT s.split,c.objective_id,c.column_name,s.sample_id,s.source,
            s.origin_id,s.peptide,c.raw_target,c.effective_target,c.qualifier,c.direction,c.source_details
            FROM affinity_target_changes c JOIN samples s ON s.id=c.row_id
            ORDER BY s.split,c.objective_id,c.column_name,s.source,s.origin_id,s.sample_id,
                     c.raw_target,c.qualifier,c.row_id""")
        with changes.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for (
                split,
                name,
                column,
                sid,
                source,
                origin,
                peptide,
                raw,
                effective,
                qual,
                direction,
                details,
            ) in query:
                writer.writerow(
                    {
                        "split": split,
                        "objective_id": name,
                        "endpoint": self.contract.objectives[name].endpoint,
                        "column": column,
                        "sample_id": sid,
                        "sample_source": source,
                        "origin_id": origin,
                        "peptide": peptide,
                        **json.loads(details),
                        "collated_raw_target_nM": raw,
                        "effective_target_log10_nM": effective,
                        "qualifier": qual,
                        "clipping_direction": direction,
                    }
                )
        return {"affinity_target_audit": report, "affinity_target_changes": changes}
