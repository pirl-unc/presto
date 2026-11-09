"""Disk-backed preparation with explicit dispositions and original-readout deduplication."""

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3

from .modular import Observation
from .modular_lineage import measurement_key, semantic_fingerprint
from .modular_sources import file_sha256, normalized_observation


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def prepare_observations(rows, output_dir, *, population, label_policy, split_policy):
    """Prepare pinned rows; frozen assignments are checked, never silently reshuffled.

    Exact copies of an original outcome get one vote. Conflicting exports of that
    same outcome are deferred together. Independent measurements retain separate
    votes, even when their same-input outcomes disagree. Source-specific adapters
    supply original identity and grouping; absent evidence gets a disposition.
    """
    if not all((population, label_policy, split_policy)):
        raise ValueError("population, evidence and frozen split policies are required")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    database = sqlite3.connect(output_dir / "preparation.sqlite3")
    try:
        database.execute("""CREATE TABLE rows (
            ordinal INTEGER PRIMARY KEY, export_id TEXT, source TEXT, lineage TEXT,
            semantic TEXT, inputs TEXT, split TEXT, group_id TEXT, payload TEXT,
            disposition TEXT, reason TEXT)""")
        for ordinal, raw in enumerate(rows):
            raw = raw.to_dict() if isinstance(raw, Observation) else raw
            row = None
            disposition, reason = "pending", ""
            lineage = semantic = inputs = None
            try:
                if not isinstance(raw, dict):
                    raise TypeError("observation must be a mapping")
                row = normalized_observation(raw, evidence_policy=label_policy)
                if row.split == "predict" or not row.group_id:
                    raise ValueError("training preparation requires a frozen split and group")
                key = measurement_key(row)
                if key is None:
                    disposition, reason = "deferred", "missing original measured-outcome lineage"
                elif (row.endpoint == "identity" and not row.identity_labels) or (
                    row.endpoint != "identity" and row.value is None
                ):
                    disposition, reason = "deferred", "missing measured label"
                else:
                    lineage = _json(key)
                    semantic = semantic_fingerprint(row)
                    inputs = semantic_fingerprint(row, include_target=False)
            except (ValueError, TypeError, KeyError) as error:
                disposition, reason = "excluded", str(error)
            try:
                payload = _json(row.to_dict() if row else raw)
            except (TypeError, ValueError):
                # Preserve the rejected raw value without writing non-JSON NaNs.
                payload = _json({"unserializable_raw": repr(raw)})
            metadata = raw if isinstance(raw, dict) else {}
            database.execute(
                "INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ordinal,
                    row.observation_id if row else str(metadata.get("observation_id", "")),
                    row.source if row else str(metadata.get("source", "")),
                    lineage,
                    semantic,
                    inputs,
                    row.split if row else None,
                    row.group_id if row else None,
                    payload,
                    disposition,
                    reason,
                ),
            )
        database.execute("CREATE INDEX lineage_idx ON rows(lineage)")
        database.execute("CREATE INDEX inputs_idx ON rows(inputs)")
        database.commit()
        # A disagreement about one original readout must not be resolved by input order.
        groups = database.execute("""SELECT lineage, MIN(ordinal), COUNT(DISTINCT semantic),
            COUNT(DISTINCT split) FROM rows WHERE disposition='pending' GROUP BY lineage""")
        for lineage, first, outcomes, splits in groups:
            if outcomes > 1 or splits > 1:
                reason = (
                    "conflicting exports of one measured outcome"
                    if outcomes > 1
                    else ("one original outcome spans frozen splits")
                )
                database.execute(
                    "UPDATE rows SET disposition='deferred', reason=? WHERE lineage=?",
                    (reason, lineage),
                )
            else:
                database.execute(
                    "UPDATE rows SET disposition='duplicate', reason=? WHERE lineage=?",
                    (f"copy of ordinal {first}", lineage),
                )
                database.execute(
                    "UPDATE rows SET disposition='admitted', reason='' WHERE ordinal=?", (first,)
                )
        database.commit()

        contradictions = 0
        with (output_dir / "contradictions.jsonl").open("x") as stream:
            for inputs, count in database.execute("""SELECT inputs, COUNT(*) FROM rows
                WHERE disposition='admitted' GROUP BY inputs HAVING COUNT(DISTINCT semantic)>1"""):
                members = [
                    {"export_id": row_id, "source": source, "lineage": json.loads(lineage)}
                    for row_id, source, lineage in database.execute(
                        "SELECT export_id,source,lineage FROM rows "
                        "WHERE inputs=? AND disposition='admitted'",
                        (inputs,),
                    )
                ]
                stream.write(
                    _json(
                        {
                            "inputs_sha256": inputs,
                            "measurements": count,
                            "members": members,
                            "action": "retained independent measurements",
                        }
                    )
                    + "\n"
                )
                contradictions += 1

        streams, paths, counts, census = {}, {}, Counter(), Counter()
        try:
            with (output_dir / "dispositions.jsonl").open("x") as dispositions:
                for (
                    ordinal,
                    export_id,
                    source,
                    lineage,
                    payload,
                    disposition,
                    reason,
                ) in database.execute(
                    "SELECT ordinal,export_id,source,lineage,payload,disposition,reason "
                    "FROM rows ORDER BY ordinal"
                ):
                    counts[disposition] += 1
                    raw = json.loads(payload)
                    if not isinstance(raw, dict):
                        raw = {"invalid_raw": raw}
                    census[
                        (
                            disposition,
                            source,
                            raw.get("endpoint", "unknown"),
                            raw.get("split", "unknown"),
                            raw.get("qualifier", "unknown"),
                            str(raw.get("unit")),
                            ",".join(
                                sorted(
                                    {x.get("mhc_class") or "unknown" for x in raw.get("mhc", [])}
                                )
                            ),
                        )
                    ] += 1
                    dispositions.write(
                        _json(
                            {
                                "ordinal": ordinal,
                                "export_id": export_id,
                                "source": source,
                                "lineage": json.loads(lineage) if lineage else None,
                                "disposition": disposition,
                                "reason": reason,
                                "raw": raw,
                            }
                        )
                        + "\n"
                    )
                    if disposition != "admitted":
                        continue
                    row = Observation.from_dict(raw)
                    aliases = [
                        {"source": other_source, "export_id": other_id, "group_id": group}
                        for other_source, other_id, group in database.execute(
                            "SELECT source,export_id,group_id FROM rows "
                            "WHERE lineage=? ORDER BY ordinal",
                            (lineage,),
                        )
                    ]
                    provenance = {
                        **row.provenance,
                        "export_aliases": aliases,
                        "group_aliases": sorted({value["group_id"] for value in aliases}),
                    }
                    canonical_id = "measurement:" + hashlib.sha256(lineage.encode()).hexdigest()
                    row = replace(row, observation_id=canonical_id, provenance=provenance)
                    if source not in streams:
                        path = output_dir / (
                            "observations-"
                            + hashlib.sha256(source.encode()).hexdigest()[:16]
                            + ".jsonl"
                        )
                        paths[source] = path
                        streams[source] = path.open("x")
                    streams[source].write(_json(row.to_dict()) + "\n")
        finally:
            for stream in streams.values():
                stream.close()
        report = {
            "schema": "presto.modular.preparation.v1",
            "population": population,
            "label_policy": label_policy,
            "split_policy": split_policy,
            "dispositions": dict(counts),
            "independent_input_contradictions": contradictions,
            "census": [
                dict(
                    zip(
                        (
                            "disposition",
                            "source",
                            "endpoint",
                            "split",
                            "qualifier",
                            "unit",
                            "classes",
                        ),
                        key,
                    ),
                    count=value,
                )
                for key, value in sorted(census.items())
            ],
            "sources": [
                {
                    "source": source,
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                    "population": population,
                    "label_policy": label_policy,
                    "split_policy": split_policy,
                }
                for source, path in sorted(paths.items())
            ],
            "dispositions_sha256": file_sha256(output_dir / "dispositions.jsonl"),
        }
        (output_dir / "preparation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        return report
    finally:
        database.close()
