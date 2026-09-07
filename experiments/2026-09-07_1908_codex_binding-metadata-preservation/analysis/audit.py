"""Trace actual pre/post merged loader constructors before memory-bounding caps."""

import collections
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from presto.data.collate import PrestoCollator, PrestoSample
from presto.data.loaders import BindingRecord, KineticsRecord, PrestoDataset, StabilityRecord
from presto.data import vocab
from presto.scripts import train_iedb

EXP = Path(__file__).resolve().parents[1]
DESCRIPTORS = {"assay_type", "assay_method", "effector_culture_condition", "apc_culture_condition"}
CLASSES = {"binding": BindingRecord, "kinetics": KineticsRecord, "stability": StabilityRecord}
AXES = {
    "assay_type": vocab.BINDING_ASSAY_TYPES,
    "assay_method": vocab.BINDING_ASSAY_METHODS,
    "assay_prep": vocab.BINDING_ASSAY_PREP,
    "assay_geometry": vocab.BINDING_ASSAY_GEOMETRY,
    "assay_readout": vocab.BINDING_ASSAY_READOUT,
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def run_condition(condition, data):
    started = time.perf_counter()
    counts = collections.Counter()
    signatures = collections.defaultdict(collections.Counter)
    missing = collections.defaultdict(collections.Counter)
    fingerprints = {group: hashlib.sha256() for group in CLASSES}
    example = {}
    namespace = vars(train_iedb).copy()

    def constructor(group, cls):
        def capture(**kwargs):
            record = cls(**kwargs)
            fields = dataclasses.asdict(record)
            counts[group] += 1
            numeric_fields = {key: value for key, value in fields.items() if key not in DESCRIPTORS}
            fingerprints[group].update(
                (json.dumps(numeric_fields, sort_keys=True, separators=(",", ":")) + "\n").encode()
            )
            for key in DESCRIPTORS & fields.keys():
                missing[group][key] += int(not fields[key])
            signatures[group][(
                record.assay_type, record.assay_method,
                getattr(record, "measurement_type", None),
            )] += 1
            if (
                group == "binding" and record.peptide == "EVMPVSMAK"
                and record.mhc_allele == "HLA-A*03:01" and record.value == 473.0
                and not example
            ):
                example.update(fields)
            return record
        return capture

    for group, cls in CLASSES.items():
        namespace[cls.__name__] = constructor(group, cls)
    frozen = EXP / f"reproduce/source/load_{condition}.py"
    exec(compile(frozen.read_text(), str(frozen), "exec"), namespace)
    result = namespace["load_records_from_merged_tsv"](
        data, max_binding=1, max_kinetics=1, max_stability=1, max_processing=1,
        max_elution=1, max_tcell=1, max_vdjdb=1, cap_sampling="head", sampling_seed=42,
    )
    collator = PrestoCollator()
    selectors = {}
    for group, signatures_for_group in signatures.items():
        axis_counts = {axis: collections.Counter() for axis in AXES}
        for (assay_type, method, measurement), count in signatures_for_group.items():
            fields = (
                {"binding_assay_type": assay_type, "binding_assay_method": method,
                 "bind_measurement_type": measurement} if group == "binding" else
                {f"{group}_assay_type": assay_type, f"{group}_assay_method": method}
            )
            sample = PrestoSample(peptide="SIINFEKL", **fields)
            context = collator._collate_binding_context([sample])
            for axis, vocabulary in AXES.items():
                axis_counts[axis][vocabulary[context[f"{axis}_idx"].item()]] += count
        selectors[group] = {axis: dict(values) for axis, values in axis_counts.items()}
    summary = {
        "condition": condition, "runtime_seconds": time.perf_counter() - started,
        "routed_quantitative_rows_before_caps": dict(counts),
        "non_descriptor_record_sha256": {key: value.hexdigest() for key, value in fingerprints.items()},
        "missing_descriptor_counts": {key: dict(value) for key, value in missing.items()},
        "selector_counts_before_caps": selectors, "loader_stats": result[-1],
        "retained_records": {group: len(result[i]) for i, group in enumerate(CLASSES)},
        "example_record": example,
    }
    target = EXP / f"results/{condition}.json"
    target.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"condition": condition, "rows": dict(counts),
                      "runtime_seconds": summary["runtime_seconds"]}), flush=True)
    return summary


def main():
    started = time.perf_counter()
    launch = json.loads((EXP / "reproduce/launch.json").read_text())
    for name, expected in launch["production_sha256"].items():
        assert digest(ROOT / name) == expected, f"Use recorded production revision for {name}"
    data = ROOT / launch["data_path"]
    assert digest(data) == launch["data_sha256"], "Source file changed"
    (EXP / "results").mkdir(exist_ok=True)
    before = run_condition("before", data)
    after = run_condition("after", data)
    assert before["routed_quantitative_rows_before_caps"] == after["routed_quantitative_rows_before_caps"]
    assert before["non_descriptor_record_sha256"] == after["non_descriptor_record_sha256"]
    assert before["loader_stats"] == after["loader_stats"]
    changes = {}
    for group, axes in after["selector_counts_before_caps"].items():
        changes[group] = {}
        for axis, new in axes.items():
            old = before["selector_counts_before_caps"][group][axis]
            changes[group][axis] = {
                label: {"before": old.get(label, 0), "after": new.get(label, 0),
                        "delta": new.get(label, 0) - old.get(label, 0)}
                for label in sorted(set(old) | set(new))
            }
    record = BindingRecord(**after["example_record"])
    dataset = PrestoDataset(binding_records=[record])
    assert len(dataset) == 1
    sample = dataset[0]
    batch = PrestoCollator()([sample])
    sample_context = {key: value.tolist() for key, value in batch.binding_context.items()}
    (EXP / "results/real_example_sample.json").write_text(
        json.dumps({"record": dataclasses.asdict(record), "sample": dataclasses.asdict(sample),
                    "binding_context": sample_context}, indent=2) + "\n"
    )
    summary = {
        "agent_model": "Codex / GPT-6", "kind": "Full-source adapter/selector audit; no training",
        "data_path": launch["data_path"], "data_sha256": launch["data_sha256"],
        "runtime_seconds": time.perf_counter() - started,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "unchanged_row_counts": True, "unchanged_non_descriptor_fingerprints": True,
        "unchanged_loader_stats": True, "selector_changes": changes,
        "real_example_context": sample_context,
    }
    (EXP / "results/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "selector_changes"}, indent=2))


if __name__ == "__main__":
    main()
