"""Exact, disk-backed counts of canonical output observations after splitting.

Rows and label observations are stored once; alias objective hits are counted
separately. SQLite performs exact distinct counting without retaining a Python
set of peptide/allele/source strings for every output column.
"""

import csv
import hashlib
import json
import math
import sqlite3
import tempfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from .mil import MIL_TASKS, get_mil_channel, resolve_mil_targets, mil_observation_slots
from .output_contract import OutputContract, label_evidence
from .supervision import resolve_row_targets


COUNT_FIELDS = (
    "observations",
    "training_rows",
    "source_rows",
    "unique_observations",
    "traceable_observations",
    "untraceable_rows",
    "distinct_peptides",
    "positive",
    "negative",
    "graded",
    "exact",
    "left_censored",
    "right_censored",
    "instances",
    "unique_positive",
    "unique_negative",
    "unique_exact",
)
CONTEXT_FIELDS = (
    "mhc_class",
    "species",
    "species_of_origin",
    "machinery",
    "peptide_source",
    "tcell_assay_method",
    "tcell_assay_readout",
    "tcell_apc_name",
    "tcell_effector_culture",
    "tcell_in_vitro_process",
    "apm_perturbation",
    "processing_stimulus",
)


class OutputCoverageCensus:
    """A context-managed census, optionally retained as a queryable artifact."""

    def __init__(self, contract: OutputContract, path: Path | str | None = None):
        self.contract = contract
        self._temporary = (
            tempfile.TemporaryDirectory(prefix="presto-output-coverage-") if path is None else None
        )
        self.path = (
            Path(path) if path is not None else Path(self._temporary.name) / "coverage.sqlite"
        )
        if self.path.exists():
            raise FileExistsError(f"Refusing to overwrite an existing census: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript("""
            PRAGMA temp_store=FILE;
            CREATE TABLE samples (
                id INTEGER PRIMARY KEY, split TEXT NOT NULL, sample_id TEXT NOT NULL,
                source TEXT NOT NULL, origin_id TEXT NOT NULL, traceable INTEGER NOT NULL,
                peptide TEXT NOT NULL
            );
            CREATE TABLE facets (row_id INTEGER NOT NULL, field TEXT NOT NULL, value TEXT NOT NULL,
                                 UNIQUE(row_id, field, value));
            CREATE INDEX facets_row ON facets(row_id);
            CREATE TABLE observations (
                id INTEGER PRIMARY KEY, row_id INTEGER NOT NULL, endpoint TEXT NOT NULL,
                column_name TEXT NOT NULL, role TEXT NOT NULL, family TEXT NOT NULL,
                origin TEXT NOT NULL, observation_key TEXT NOT NULL, target REAL NOT NULL,
                qualifier INTEGER, loss_type TEXT NOT NULL, instances INTEGER NOT NULL,
                scope TEXT NOT NULL, occurrence INTEGER NOT NULL,
                UNIQUE(row_id, endpoint, column_name, observation_key, scope, occurrence)
            );
            CREATE INDEX observations_endpoint ON observations(endpoint, column_name, row_id);
        """)
        self.splits: dict[str, int] = {}
        self.objective_hits = defaultdict(
            lambda: defaultdict(lambda: {"observations": 0, "mask_weight": 0.0})
        )
        self._hash_sum = 0
        self._hash_count = 0
        self._class_counts = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.db.close()
        if self._temporary is not None:
            self._temporary.cleanup()

    def declare_split(self, name: str) -> None:
        self.splits.setdefault(name, 0)

    def _fingerprint(self, value):
        rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
        self._hash_sum = (
            self._hash_sum + int.from_bytes(hashlib.sha256(rendered).digest(), "big")
        ) % (1 << 256)
        self._hash_count += 1

    @property
    def sha256(self):
        payload = (
            b"presto-output-evidence-v1\0"
            + self._hash_count.to_bytes(16, "big")
            + self._hash_sum.to_bytes(32, "big")
        )
        return hashlib.sha256(payload).hexdigest()

    def add_batch(self, split: str, samples, batch) -> None:
        """Consume the same collated batch used by the canonical support audit."""
        self._class_counts.clear()
        self.declare_split(split)
        row_ids = []
        origins = []
        for sample in samples:
            source = str(getattr(sample, "sample_source", "") or "unknown")
            sample_id = str(getattr(sample, "sample_id", "") or "")
            evidence = str(getattr(sample, "evidence_row_id", "") or "")
            # Missing source identity remains visible. A sample ID can dedupe
            # repeated objectives but cannot prove original assay-row identity.
            origin_id = evidence or sample_id or f"unidentified:{split}:{self.splits[split]}"
            traceable = bool(evidence and source != "unknown")
            peptide = str(getattr(sample, "peptide", "") or "")
            cursor = self.db.execute(
                "INSERT INTO samples(split,sample_id,source,origin_id,traceable,peptide) "
                "VALUES (?,?,?,?,?,?)",
                (split, sample_id, source, origin_id, int(traceable), peptide),
            )
            row_id = cursor.lastrowid
            row_ids.append(row_id)
            origins.append((source, origin_id))
            facets = [
                (row_id, name, str(getattr(sample, name, None) or "unknown"))
                for name in CONTEXT_FIELDS
            ]
            for attr, field in (
                ("source_mhc_alleles", "source_allele"),
                ("resolved_mhc_alleles", "resolved_allele"),
            ):
                facets.extend(
                    (row_id, field, str(allele)) for allele in getattr(sample, attr, ()) if allele
                )
            primary = getattr(sample, "primary_allele", None)
            if primary:
                facets.append((row_id, "primary_allele", str(primary)))
            self.db.executemany("INSERT OR IGNORE INTO facets VALUES (?,?,?)", facets)
            self._fingerprint(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "source": source,
                    "origin_id": origin_id,
                    "traceable": traceable,
                    "peptide": peptide,
                    "facets": sorted((name, value) for _, name, value in facets),
                    "provenance": getattr(sample, "target_provenance", {}),
                    "generated": getattr(sample, "synthetic_kind", None),
                }
            )
            self.splits[split] += 1

        def add(
            objective_id,
            sample_index,
            column,
            value,
            raw_value,
            qualifier,
            mask_weight,
            instances,
            occurrence=0,
        ):
            objective = self.contract.objectives[objective_id]
            if not math.isfinite(value) or not math.isfinite(raw_value):
                raise ValueError(f"{objective_id}: nonfinite active observation")
            if objective.loss_type == "bce" and not 0 <= value <= 1:
                raise ValueError(f"{objective_id}: binary response outside [0,1]")
            if qualifier is not None and qualifier not in {-1, 0, 1}:
                raise ValueError(f"{objective_id}: invalid censor qualifier {qualifier}")
            evidence = label_evidence(objective, samples[sample_index])
            # Different transformed KD objectives use the same binding source
            # label. Include source units/value and qualifier, not objective ID.
            identity = [
                *origins[sample_index],
                objective.source_target,
                column,
                raw_value,
                qualifier,
            ]
            key = json.dumps(identity, separators=(",", ":"))
            record = (
                row_ids[sample_index],
                objective.endpoint,
                column,
                evidence.role,
                evidence.family,
                evidence.origin,
                key,
                value,
                qualifier,
                objective.loss_type,
                instances,
                objective.channel,
                occurrence,
            )
            inserted = self.db.execute(
                """INSERT OR IGNORE INTO observations
                (row_id,endpoint,column_name,role,family,origin,observation_key,target,qualifier,
                 loss_type,instances,scope,occurrence)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                record,
            )
            if inserted.rowcount:
                self._fingerprint(
                    {
                        "split": split,
                        "endpoint": objective.endpoint,
                        "column": column,
                        "evidence": [evidence.role, evidence.family, evidence.origin],
                        "observation": identity,
                        "target": value,
                        "instances": instances,
                        "scope": objective.channel,
                        "occurrence": occurrence,
                    }
                )
            hits = self.objective_hits[split][objective_id]
            hits["observations"] += 1
            hits["mask_weight"] += mask_weight

        for name, target in resolve_row_targets(batch).items():
            spec = target.spec
            transformed = target.transformed
            for index in (target.mask > 0).nonzero().flatten().tolist():
                column = ""
                if target.selectors is not None:
                    column = spec.columns[int(target.selectors[index])]
                elif target.components is not None:
                    column = spec.component_names[int(target.components[index])]
                raw = float(target.raw_target[index])
                qualifier = int(target.qualifiers[index]) if target.qualifiers is not None else None
                add(
                    f"row:{name}",
                    int(target.source_rows[index]),
                    column,
                    float(transformed[index]),
                    raw,
                    qualifier,
                    float(target.mask[index]),
                    1,
                )
        for channel_name, specs in MIL_TASKS.items():
            channel = get_mil_channel(batch, channel_name)
            if channel is None:
                continue
            slots = mil_observation_slots(channel, source_rows=len(samples))
            for name, target in resolve_mil_targets(channel, specs).items():
                for index in target.mask.nonzero().flatten().tolist():
                    column = (
                        target.spec.columns[int(target.selectors[index])]
                        if target.selectors is not None
                        else ""
                    )
                    value = float(target.labels[index])
                    add(
                        f"{channel_name}:{name}",
                        int(channel["bag_sample_indices"][index]),
                        column,
                        value,
                        value,
                        None,
                        1.0,
                        int(target.instance_counts[index]),
                        occurrence=slots[index],
                    )
        self.db.commit()

    def _selection(self, split, endpoint, column, roles, families, sources):
        if endpoint not in self.contract.outputs:
            raise ValueError(f"Unknown canonical endpoint: {endpoint}")
        spec = self.contract.outputs[endpoint]
        if column and column not in spec.columns:
            raise ValueError(f"Unknown column {endpoint}:{column}")
        clauses, args = ["s.split=?", "o.endpoint=?"], [split, endpoint]
        if spec.shape != "classes" and column:
            clauses.append("o.column_name=?")
            args.append(column)
        for field, values in (("o.role", roles), ("o.family", families), ("s.source", sources)):
            if values is not None:
                if not values:
                    clauses.append("0")
                else:
                    clauses.append(f"{field} IN ({','.join('?' for _ in values)})")
                    args.extend(values)
        return " AND ".join(clauses), args

    def counts(
        self,
        split,
        endpoint,
        column="",
        *,
        roles=None,
        families=None,
        sources=None,
        include_facets=True,
    ):
        """Unique observations and response balance for one canonical claim cell.

        CE classes each receive the full vector objective. Their positive count
        is the class's labeled incidence, with remaining examples negative.
        Selected panels and independent vector components count selected cells.
        """
        where, args = self._selection(split, endpoint, column, roles, families, sources)
        spec = self.contract.outputs[endpoint]
        if spec.shape == "classes" and column:
            index = spec.columns.index(column)
            positive, negative = f"o.target={index}", f"o.target!={index}"
            result = self.counts(
                split,
                endpoint,
                roles=roles,
                families=families,
                sources=sources,
                include_facets=include_facets,
            )
            if result["observations"]:
                balance = self.db.execute(
                    f"""SELECT COALESCE(SUM({positive}),0), COALESCE(SUM({negative}),0),
                        COUNT(DISTINCT CASE WHEN {positive} THEN o.observation_key END),
                        COUNT(DISTINCT CASE WHEN {negative} THEN o.observation_key END)
                    FROM observations o JOIN samples s ON s.id=o.row_id WHERE {where}""",
                    args,
                ).fetchone()
                result.update(
                    zip(("positive", "negative", "unique_positive", "unique_negative"), balance)
                )
            return result

        # Categorical columns share every count except their response balance.
        cache_key = (where, tuple(args), include_facets)
        if spec.shape == "classes" and cache_key in self._class_counts:
            return deepcopy(self._class_counts[cache_key])
        positive, negative = (
            "o.loss_type='bce' AND o.target>0.5",
            "o.loss_type='bce' AND o.target<=0.5",
        )
        row = self.db.execute(
            f"""SELECT COUNT(*), COUNT(DISTINCT s.id),
                COUNT(DISTINCT CASE WHEN s.traceable
                    THEN LENGTH(s.source)||':'||s.source||s.origin_id END),
                COUNT(DISTINCT o.observation_key),
                COUNT(DISTINCT CASE WHEN s.traceable THEN o.observation_key END),
                COUNT(DISTINCT CASE WHEN NOT s.traceable THEN s.id END),
                COUNT(DISTINCT NULLIF(s.peptide,'')),
                COALESCE(SUM({positive}),0), COALESCE(SUM({negative}),0),
                COALESCE(SUM(o.loss_type='bce' AND o.target>0 AND o.target<1),0),
                COALESCE(SUM(o.qualifier=0 OR (o.qualifier IS NULL AND o.loss_type='mse')),0),
                COALESCE(SUM(o.qualifier<0),0), COALESCE(SUM(o.qualifier>0),0),
                COALESCE(SUM(o.instances),0),
                COUNT(DISTINCT CASE WHEN {positive} THEN o.observation_key END),
                COUNT(DISTINCT CASE WHEN {negative} THEN o.observation_key END),
                COUNT(DISTINCT CASE WHEN o.qualifier=0
                    OR (o.qualifier IS NULL AND o.loss_type='mse') THEN o.observation_key END)
            FROM observations o JOIN samples s ON s.id=o.row_id WHERE {where}""",
            args,
        ).fetchone()
        result = dict(zip(COUNT_FIELDS, row))
        if include_facets:
            result["facets"] = {}
            for field, value, count in self.db.execute(
                f"""SELECT f.field,f.value,COUNT(DISTINCT s.id)
                FROM observations o JOIN samples s ON s.id=o.row_id JOIN facets f ON f.row_id=s.id
                WHERE {where}
                    AND f.field NOT IN ('source_allele','resolved_allele','primary_allele')
                GROUP BY f.field,f.value ORDER BY f.field,f.value""",
                args,
            ):
                result["facets"].setdefault(field, {})[value] = count
            for field in ("source_allele", "resolved_allele", "primary_allele"):
                result[f"distinct_{field}s"] = 0
            for field, count in self.db.execute(
                f"""SELECT f.field,COUNT(DISTINCT f.value)
                FROM observations o JOIN samples s ON s.id=o.row_id JOIN facets f ON f.row_id=s.id
                WHERE {where}
                    AND f.field IN ('source_allele','resolved_allele','primary_allele')
                GROUP BY f.field""",
                args,
            ):
                result[f"distinct_{field}s"] = count
        if spec.shape == "classes":
            self._class_counts[cache_key] = deepcopy(result)
        return result

    def report(self) -> dict:
        """Declare zero-support quantities too; diagnostic existence is not a label."""
        report = {
            "schema_version": 1,
            "evidence_sha256": self.sha256,
            "contract": self.contract.to_dict(),
            "splits": {},
            "counting_notes": [
                "Objective hits include repeated losses; canonical counts deduplicate aliases.",
                "Unique observations without evidence_row_id use a sample identity "
                "and are separately marked untraceable.",
                "Bag labels do not establish individual candidate-allele outcomes.",
                "CE columns share vector observations; positives count labeled class incidence.",
                "Positive is response >0.5, negative <=0.5; graded counts targets between 0 and 1.",
            ],
        }
        for split, rows in self.splits.items():
            entries = {}
            for endpoint, spec in self.contract.outputs.items():
                entry = self.counts(split, endpoint)
                entry["columns"] = {
                    column: self.counts(split, endpoint, column) for column in spec.columns
                }
                groups = self.db.execute(
                    """SELECT DISTINCT o.role,o.family,s.source FROM observations o
                    JOIN samples s ON s.id=o.row_id WHERE s.split=? AND o.endpoint=?
                    ORDER BY o.role,o.family,s.source""",
                    (split, endpoint),
                ).fetchall()
                evidence = []
                for role, family, source in groups:
                    filters = dict(roles=[role], families=[family], sources=[source])
                    if len(groups) == 1:
                        # This group is the entire endpoint population, including its columns.
                        group_counts = deepcopy(entry)
                    else:
                        group_counts = self.counts(split, endpoint, **filters)
                        group_counts["columns"] = {
                            column: self.counts(split, endpoint, column, **filters)
                            for column in spec.columns
                        }
                    evidence.append(
                        {
                            "role": role,
                            "family": family,
                            "source": source,
                            **group_counts,
                        }
                    )
                entry["evidence"] = evidence
                entries[endpoint] = entry
            report["splits"][split] = {
                "rows": rows,
                "outputs": entries,
                "objectives": {
                    identity: self.objective_hits[split].get(
                        identity, {"observations": 0, "mask_weight": 0.0}
                    )
                    for identity in self.contract.objectives
                },
            }
        return report


def write_output_coverage_artifacts(output_dir: Path | str, report: dict) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "output_coverage.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    csv_path = output / "output_coverage.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("split", "endpoint", "column", *COUNT_FIELDS))
        writer.writeheader()
        for split, payload in report["splits"].items():
            for endpoint, entry in payload["outputs"].items():
                for column, counts in [("", entry), *entry["columns"].items()]:
                    writer.writerow(
                        {
                            "split": split,
                            "endpoint": endpoint,
                            "column": column,
                            **{key: counts[key] for key in COUNT_FIELDS},
                        }
                    )
    return {"output_coverage": json_path, "output_coverage_csv": csv_path}
