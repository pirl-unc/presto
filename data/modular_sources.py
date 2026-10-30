"""Explicit source adapters and a hash-pinned streaming observation index.

Adapters require resolved sequence pairs and frozen split/group assignments.
They do not guess the TCRvdb authors' file schema or qualify source evidence.
"""

from __future__ import annotations

from collections import Counter
from array import array
from dataclasses import asdict, dataclass, replace
from itertools import groupby
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .bulk_ms import BulkMSRecord
from .groove import prepare_mhc_input
from .loaders import BindingRecord, KineticsRecord, StabilityRecord
from .modular import Context, MHCInput, Observation, sequence, endpoint_spec, supervised_targets
from .modular_lineage import measurement_key

MANIFEST_VERSION = "presto.modular.manifest.v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_pair(mhc_a: str, mhc_b: str | None, mhc_class: str, **metadata) -> MHCInput:
    """Resolve groove segments with mhcseqs, never infer a missing II partner."""
    sequence(mhc_a, max_length=4096)
    if mhc_b is not None:
        sequence(mhc_b, max_length=4096)
    result = prepare_mhc_input(
        mhc_a=mhc_a, mhc_b=mhc_b, mhc_class=mhc_class, allow_fallback_truncation=False
    )
    pair = MHCInput(result.groove_half_1, result.groove_half_2, result.mhc_class, **metadata)
    pair.validate()
    return pair


def affinity_observations(
    record: BindingRecord | KineticsRecord | StabilityRecord,
    mhc: MHCInput,
    *,
    split: str,
    group_id: str,
) -> list[Observation]:
    """Use raw record values before legacy clipping, preserving separate proxies."""
    if not record.evidence_row_id or not group_id:
        raise ValueError("source observation ID and frozen group assignment required")
    if record.generation_kind:
        raise ValueError("generated affinity controls require a separate declared policy")
    if isinstance(record, BindingRecord):
        names = {
            "IC50": "ic50",
            "KD": "kd",
            "EC50": "ec50",
            "KD (IC50)": "kd_via_ic50",
            "KD (EC50)": "kd_via_ec50",
            "KD via IC50": "kd_via_ic50",
            "KD via EC50": "kd_via_ec50",
            "half maximal inhibitory concentration (IC50)": "ic50",
            "half maximal effective concentration (EC50)": "ec50",
            "dissociation constant KD": "kd",
            "dissociation constant KD (~IC50)": "kd_via_ic50",
            "dissociation constant KD (~EC50)": "kd_via_ec50",
        }
        if record.measurement_type not in names:
            raise ValueError(f"unsupported measured endpoint: {record.measurement_type}")
        outcomes = [(names[record.measurement_type], record.value, record.unit, record.qualifier)]
    elif isinstance(record, KineticsRecord):
        outcomes = [("koff", record.koff, "1/s", record.koff_qualifier)]
    elif isinstance(record, StabilityRecord):
        outcomes = [
            ("half_life", record.t_half, "h", record.t_half_qualifier),
            ("tm", record.tm, "C", record.tm_qualifier),
        ]
    else:
        raise TypeError("expected a raw binding, kinetics or stability record")
    result = []
    for endpoint, value, unit, qualifier in outcomes:
        if value is None:
            continue
        if qualifier not in (-1, 0, 1):
            raise ValueError("unsupported raw qualifier")
        observation = Observation(
            f"{record.evidence_row_id}:{endpoint}",
            record.source,
            endpoint,
            record.peptide,
            (mhc,),
            value=float(value),
            unit=unit,
            qualifier={-1: "<", 0: "=", 1: ">"}[qualifier],
            split=split,
            group_id=group_id,
            provenance={"raw_record": asdict(record), "target_policy": "physical_unclipped_v1"},
        )
        observation.validate()
        result.append(observation)
    return result


def scoped_bulk_observation(record: BulkMSRecord, *, split: str, group_id: str) -> Observation:
    """Only a verified search-scoped vote, never a raw-positive/generated label."""
    if (
        record.detectability_label is None
        or record.generated_kind
        or record.source_endpoint != "observed_in_selected_search_scope"
        or not record.source_search_space_id
        or not record.source_reference_sha256
        or not record.evidence_row_id
        or not group_id
    ):
        raise ValueError(
            "bulk observation requires verified search scope, lineage and measured vote"
        )
    try:
        unique_parent = int(record.source_n_candidate_occurrences) == 1
    except (TypeError, ValueError):
        unique_parent = False
    context = Context(
        enzyme=record.machinery,
        source_n=record.flank_n if unique_parent else None,
        source_c=record.flank_c if unique_parent else None,
        n_terminus=record.flank_n_is_terminus if unique_parent else None,
        c_terminus=record.flank_c_is_terminus if unique_parent else None,
        # Raw acquisition metadata is not automatically a prospective numeric feature.
        provenance={"raw_acquisition": record.acquisition},
    )
    observation = Observation(
        record.evidence_row_id,
        record.source,
        "bulk_observation",
        record.peptide,
        value=record.detectability_label,
        unit="binary",
        split=split,
        group_id=group_id,
        context=context,
        provenance={"raw_record": asdict(record), "interpretation": "within_search_observability"},
    )
    observation.validate()
    return observation


def normalized_observation(row: dict, *, evidence_policy: str) -> Observation:
    """Common admission for already curated cellular, identity and paired-TCR rows.

    The caller pins the source-specific evidence policy. Mere database membership
    does not establish a measured functional response or a biological negative.
    """
    if not evidence_policy:
        raise ValueError("explicit source evidence policy required")
    observation = Observation.from_dict(row)
    if observation.endpoint != "identity" and endpoint_spec(observation.endpoint).quantitative:
        if observation.provenance.get("measured_outcome") is not True:
            raise ValueError("quantity supervision requires an attested measured outcome")
        parts = endpoint_spec(observation.endpoint).parts
        if len(parts) == 1:
            evidence = observation.provenance.get("component_measurement", {})
            if (
                not isinstance(evidence, dict)
                or evidence.get("component") != parts[0]
                or evidence.get("basis") != "independent_component_measurement"
                or not isinstance(evidence.get("reference"), str)
                or not evidence["reference"]
            ):
                raise ValueError("component anchor requires independent measurement evidence")
    if observation.endpoint in ("tcr_binding", "tcr_response"):
        if observation.provenance.get("tcr_sequence_scope") != "full_paired":
            raise ValueError("paired specificity requires source-attested full TCR chains")
        if observation.provenance.get("measured_outcome") is not True:
            raise ValueError("database association alone is not a measured TCR outcome")
    for task in ("mhc_pairing", "tcr_pairing"):
        if task not in observation.identity_labels:
            continue
        pair_evidence = observation.provenance.get("pairing_evidence", {})
        if not isinstance(pair_evidence, dict) or not isinstance(pair_evidence.get(task, {}), dict):
            raise ValueError("pairing evidence must be a task-specific mapping")
        evidence = pair_evidence.get(task, {})
        basis = evidence.get("basis")
        if (
            basis
            not in (
                "observed_pair",
                "measured_pairing",
                "generated_pairing_control",
            )
            or not isinstance(evidence.get("reference"), str)
            or not evidence["reference"]
        ):
            raise ValueError(f"{task} requires explicit source pairing evidence")
        if basis == "observed_pair" and observation.identity_labels[task] != "matched":
            raise ValueError("an observed pair does not supply a measured mismatch label")
        if basis == "generated_pairing_control" and (
            not isinstance(evidence.get("policy"), str)
            or not evidence["policy"]
            or not isinstance(evidence.get("parent_ids"), list)
            or not evidence["parent_ids"]
            or any(not isinstance(value, str) or not value for value in evidence["parent_ids"])
        ):
            raise ValueError(
                "generated pairing controls require a declared policy and parent lineage"
            )
    return replace(
        observation, provenance={**observation.provenance, "evidence_policy": evidence_policy}
    )


def write_observations(path: str | Path, observations: Iterable[Observation]) -> dict:
    """Preparation primitive; callers retain excluded/deferred rows in their census."""
    path = Path(path)
    counts = Counter()
    with path.open("x") as stream:
        for observation in observations:
            observation.validate()
            stream.write(json.dumps(observation.to_dict(), sort_keys=True, allow_nan=False) + "\n")
            counts[observation.endpoint] += 1
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "counts": dict(counts)}


@dataclass(frozen=True, slots=True)
class ObservationMetadata:
    """Interned admission projection; no sequences, labels or raw provenance."""

    endpoint: str
    split: str
    mhc_classes: frozenset[str | None]
    targets: frozenset[str]

    @classmethod
    def from_observation(cls, row: Observation):
        return cls(
            row.endpoint,
            row.split,
            frozenset(mhc.class_override or mhc.mhc_class for mhc in row.mhc),
            supervised_targets(row),
        )


class ObservationIndex:
    """Byte offsets keep sequences/registers out of corpus-sized in-memory arrays."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("schema") != MANIFEST_VERSION:
            raise ValueError("unsupported source manifest schema")
        if not self.manifest.get("sources"):
            raise ValueError("empty source manifest")
        self.offsets: list[tuple[Path, int]] = []
        self.partitions: dict[str, list[int]] = {}
        self.census = Counter()
        ids, groups, peptides, identities, measurements = set(), {}, {}, {}, {}
        identity_chains = {}
        self.source_paths = []
        self.file_states = {}
        self.label_policies = {}
        self._metadata_codes = array("I")
        self._metadata_values = []
        metadata_codes = {}
        for source in self.manifest["sources"]:
            required = ("path", "sha256", "source", "population", "label_policy", "split_policy")
            if any(not source.get(key) for key in required):
                raise ValueError("source manifest lacks hashes or population/label/split policy")
            path = (self.manifest_path.parent / source["path"]).resolve()
            if path in self.source_paths:
                raise ValueError("duplicate source path")
            self.source_paths.append(path)
            if file_sha256(path) != source["sha256"]:
                raise ValueError(f"source hash mismatch: {path}")
            self.file_states[path] = self._file_state(path)
            self.label_policies[path] = source["label_policy"]
            with path.open("rb") as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    try:
                        observation = normalized_observation(
                            json.loads(line), evidence_policy=source["label_policy"]
                        )
                        if observation.source != source["source"]:
                            raise ValueError("row source disagrees with source manifest")
                        if observation.observation_id in ids:
                            raise ValueError("duplicate observation ID across admitted sources")
                        measurement = measurement_key(observation)
                        if measurement is None:
                            raise ValueError(
                                "missing original measured-outcome lineage; "
                                "prepare row dispositions first"
                            )
                        if measurement in measurements:
                            raise ValueError(
                                f"duplicate measured outcome: {observation.observation_id} "
                                f"and {measurements[measurement]}"
                            )
                        if not observation.group_id:
                            raise ValueError("missing frozen grouping identity")
                        if observation.endpoint != "identity" and observation.value is None:
                            raise ValueError("unlabelled observations belong in deferred census")
                        if observation.endpoint == "identity" and not observation.identity_labels:
                            raise ValueError("identity row has no measured identity labels")
                        if observation.split == "predict":
                            raise ValueError("prediction requests cannot enter a training manifest")
                        self._disjoint(groups, observation.group_id, observation.split, "group")
                        for alias in observation.provenance.get("group_aliases", []):
                            if not isinstance(alias, str) or not alias:
                                raise ValueError("invalid original group alias")
                            self._disjoint(groups, alias, observation.split, "group alias")
                        if observation.peptide:
                            self._disjoint(
                                peptides, observation.peptide, observation.split, "peptide"
                            )
                        if observation.endpoint == "identity":
                            # Changing a partner must not hide an identical chain's
                            # exposure across auxiliary annotation partitions.
                            chains = [
                                ("mhc", value)
                                for mhc in observation.mhc
                                for value in (mhc.groove1, mhc.groove2)
                            ] + [("tcr", observation.tcr_a), ("tcr", observation.tcr_b)]
                            for family, value in chains:
                                if value:
                                    self._disjoint(
                                        identity_chains,
                                        (family, value),
                                        observation.split,
                                        "identity chain",
                                    )
                            key = tuple((x.groove1, x.groove2) for x in observation.mhc) + (
                                observation.tcr_a,
                                observation.tcr_b,
                            )
                            if not observation.mhc and not (observation.tcr_a or observation.tcr_b):
                                key = ("peptide", observation.peptide)
                            self._disjoint(identities, key, observation.split, "identity molecule")
                    except (ValueError, TypeError, KeyError) as error:
                        raise ValueError(f"{path} byte {offset}: {error}") from error
                    ids.add(observation.observation_id)
                    measurements[measurement] = observation.observation_id
                    index = len(self.offsets)
                    self.offsets.append((path, offset))
                    metadata = ObservationMetadata.from_observation(observation)
                    code = metadata_codes.get(metadata)
                    if code is None:
                        code = len(self._metadata_values)
                        metadata_codes[metadata] = code
                        self._metadata_values.append(metadata)
                    self._metadata_codes.append(code)
                    family = "identity" if observation.endpoint == "identity" else "assay"
                    key = f"{observation.split}:{family}"
                    self.partitions.setdefault(key, []).append(index)
                    self.census[
                        f"{observation.split}:{observation.endpoint}:{observation.qualifier}"
                    ] += 1
        self.manifest_sha256 = file_sha256(self.manifest_path)

    @staticmethod
    def _disjoint(assignments, key, split, label):
        if key in assignments and assignments[key] != split:
            raise ValueError(f"{label} leaks across splits")
        assignments[key] = split

    @staticmethod
    def _file_state(path):
        state = path.stat()
        return state.st_ino, state.st_size, state.st_mtime_ns

    def __len__(self):
        return len(self.offsets)

    def metadata(self, index: int) -> ObservationMetadata:
        """Read admitted metadata with the same source-integrity check as a row."""
        path, _ = self.offsets[index]
        if self._file_state(path) != self.file_states[path]:
            raise ValueError("pinned input changed after admission")
        return self._metadata_values[self._metadata_codes[index]]

    def __getitem__(self, index: int) -> Observation:
        path, offset = self.offsets[index]
        if self._file_state(path) != self.file_states[path]:
            raise ValueError("pinned input changed after admission")
        with path.open("rb") as stream:
            stream.seek(offset)
            return normalized_observation(
                json.loads(stream.readline()), evidence_policy=self.label_policies[path]
            )

    def __iter__(self):
        # Keep sequential scans sequential without caching rows or relaxing the
        # per-observation source-mutation check used by random access.
        for path, offsets in groupby(self.offsets, key=lambda item: item[0]):
            with path.open("rb") as stream:
                position = None
                for _, offset in offsets:
                    if self._file_state(path) != self.file_states[path]:
                        raise ValueError("pinned input changed after admission")
                    if offset != position:
                        stream.seek(offset)
                    line = stream.readline()
                    position = offset + len(line)
                    yield normalized_observation(
                        json.loads(line), evidence_policy=self.label_policies[path]
                    )
