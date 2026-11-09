"""Portable observation-manifest benchmark worker; all optimization uses canonical fit_index."""

from pathlib import Path
import copy
from dataclasses import replace
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, rows):
    with Path(path).open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())


def manifest_for(scope, seed, output, pins):
    from presto.data.modular_sources import file_sha256

    sources = []
    inputs = output / "inputs"
    inputs.mkdir(exist_ok=True)
    for split in ("train", "validation", "test"):
        source = Path("/input") / f"joint-{split}.jsonl"
        expected = pins["joint_input_hashes"][split]
        assert sha(source) == expected
        destination = inputs / f"{scope}-{split}.jsonl"
        if not destination.exists():
            with source.open() as src, destination.open("x") as dst:
                for line in src:
                    if scope == "joint" or json.loads(line)["mhc"][0]["mhc_class"] == scope:
                        dst.write(line)
        assert file_sha256(destination) == pins["filtered_input_hashes"][f"{scope}-{split}"]
        sources.append(
            dict(
                path=str(destination),
                sha256=sha(destination),
                source="hitlist-v4-iedb",
                population="lossless class subset of admitted human direct IC50 cohort",
                label_policy="physical unclipped exact IC50 training; exact and original bounds in holdouts",
                split_policy="original exact-peptide/original-assay components; historically inspected; shared studies",
            )
        )
    template = copy.deepcopy(pins["manifest_templates"][f"shared-{scope}-s{seed}"])
    template["sources"] = sources
    path = output / f"{scope}-manifest.json"
    atomic(path, template)
    return path


def expanded_view(source):
    import torch

    model = type(source)(replace(source.config, mhc_classes=("I", "II"))).to(source.device)
    model.load_state_dict(source.state_dict(), strict=True)
    for name, tensor in source.state_dict().items():
        assert torch.equal(tensor, model.state_dict()[name])
    for name in (
        "support",
        "trained_parameters",
        "trained_features",
        "exposed_features",
        "source_label_support",
        "frozen_targets",
        "frozen_parameters",
    ):
        setattr(model, name, copy.deepcopy(getattr(source, name)))
    return model.eval()


def mean_records(rows, mean, scale):
    from presto.training.modular_evaluation import summarize

    result = []
    for row in rows:
        value = dict(row, prediction=mean)
        residual = mean - row["target"]
        if row["qualifier"] in ("<", "<="):
            residual = max(0.0, residual)
        elif row["qualifier"] in (">", ">="):
            residual = min(0.0, residual)
        value["loss"] = (residual / scale) ** 2
        result.append(value)
    return summarize(result), result


def run(condition, output):
    pins = read("/bundle/source-pins.json")
    spec = pins["conditions"][condition]
    runtime = Path("/runtimes") / spec["runtime"] / "presto"
    sys.path.insert(0, str(runtime.parent))
    import torch
    import psutil
    from presto.data.modular_sources import ObservationIndex
    from presto.training import modular_run as runner
    from presto.training.modular import load_checkpoint
    from presto.training.modular_evaluation import evaluate
    from presto.models.modular import MODEL_VERSION

    for relative, expected in pins["runtimes"][spec["runtime"]]["files"].items():
        assert sha(runtime / relative) == expected, relative
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    assert torch.cuda.is_available()
    gpu = torch.cuda.get_device_properties(0)
    assert "H100" in gpu.name, gpu.name
    torch.cuda.reset_peak_memory_stats()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "status.json").exists():
        raise ValueError("condition already attempted; no automatic retry")
    hardware = dict(
        requested_gpu="H100!",
        actual_gpu=gpu.name,
        total_gpu_bytes=gpu.total_memory,
        python=sys.version,
        torch=str(torch.__version__),
        numpy=__import__("numpy").__version__,
        cuda=torch.version.cuda,
        platform=platform.platform(),
        threads=4,
        interop_threads=1,
        tf32=False,
        amp=False,
        compile=False,
        model_version=MODEL_VERSION,
        source=pins["runtimes"][spec["runtime"]],
        nvidia_smi=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
    )
    atomic(output / "hardware.json", hardware)
    atomic(
        output / "status.json", dict(status="admitting", condition=condition, hardware=hardware)
    )
    scope = "joint" if len(spec["classes"]) == 2 else spec["classes"][0]
    manifest = manifest_for(scope, spec["seed"], output, pins)
    index = ObservationIndex(manifest)
    original = runner.train_step
    started = time.monotonic()
    process = psutil.Process()
    state = dict(steps=0, examples=0, optimizer_seconds=0.0, peak_rss=process.memory_info().rss)

    def observed(model, optimizer, rows, scales, **kwargs):
        torch.cuda.synchronize()
        before = time.monotonic()
        result = original(model, optimizer, rows, scales, **kwargs)
        torch.cuda.synchronize()
        state["optimizer_seconds"] += time.monotonic() - before
        state["steps"] += 1
        state["examples"] += len(rows)
        state["peak_rss"] = max(state["peak_rss"], process.memory_info().rss)
        if state["steps"] == 1 or state["steps"] % 10 == 0:
            point = {
                **state,
                "latest_training_loss": result["loss"],
                "elapsed_seconds": time.monotonic() - started,
                "rss": process.memory_info().rss,
                "updated_unix": time.time(),
                "gpu_allocated": torch.cuda.max_memory_allocated(),
                "gpu_reserved": torch.cuda.max_memory_reserved(),
            }
            atomic(output / "progress.json", point)
            with (output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(point) + "\n")
        return result

    runner.train_step = observed
    atomic(
        output / "status.json",
        dict(status="training", condition=condition, started_unix=time.time()),
    )
    raw = output / "raw"
    try:
        summary = runner.fit_index(index, raw, device="cuda")
        assert summary["steps"] == 1000
        atomic(output / "status.json", dict(status="evaluating_transfer", condition=condition))
        model, scales, payload = load_checkpoint(raw / "selected.pt", device="cuda")
        model.eval()
        checkpoint_hash = sha(raw / "selected.pt")
        report = dict(
            condition=condition,
            training_classes=spec["classes"],
            runtime=hardware,
            summary=summary,
            checkpoint=str(raw / "selected.pt"),
            checkpoint_sha256=checkpoint_hash,
            parameters=sum(p.numel() for p in model.parameters()),
            transfer={},
            mean_baselines={},
        )
        for split in ("train", "validation", "test"):
            with (raw / f"{split}-predictions.jsonl").open() as stream:
                rows = [json.loads(line) for line in stream]
            dump(output / f"{split}-predictions.jsonl.gz", rows)
            if split != "train":
                metrics, means = mean_records(rows, scales["ic50"].mean, scales["ic50"].scale)
                report["mean_baselines"][split] = metrics
                dump(output / f"mean-{split}-predictions.jsonl.gz", means)
            del rows
        if len(spec["classes"]) == 1:
            target = "II" if scope == "I" else "I"
            all_manifest = manifest_for("joint", spec["seed"], output, pins)
            all_index = ObservationIndex(all_manifest)
            view = expanded_view(model)
            protocol = "identical_weights_expanded_routing"
            transfer = dict(
                target_class=target,
                contract=dict(
                    protocol=protocol,
                    all_weights_preserved=True,
                    new_parameters=[],
                    target_class_labels_used=0,
                    capability_status="unsupported_class_transfer_diagnostic",
                ),
                metrics={},
            )
            for split in ("validation", "test"):
                indices = [
                    i
                    for i in all_index.partitions[f"{split}:assay"]
                    if all_index[i].mhc[0].mhc_class == target
                ]
                metrics, rows = evaluate(view, all_index, indices, scales, batch_size=64)
                for row in rows:
                    row.update(
                        evaluation_scope="zero_shot",
                        training_classes=spec["classes"],
                        capability_status="unsupported_class_transfer_diagnostic",
                    )
                dump(output / f"{protocol}-{split}-predictions.jsonl.gz", rows)
                transfer["metrics"][split] = metrics
                means, values = mean_records(rows, scales["ic50"].mean, scales["ic50"].scale)
                report["mean_baselines"][f"{split}:transfer"] = means
                dump(output / f"mean-transfer-{split}-predictions.jsonl.gz", values)
            report["transfer"][protocol] = transfer
        assert sha(raw / "selected.pt") == checkpoint_hash
        _, replay = evaluate(
            model, index, list(index.partitions["train:assay"])[:64], scales, batch_size=64
        )
        dump(output / "training-replay.jsonl.gz", replay)
        report["resources"] = dict(
            wall_seconds=time.monotonic() - started,
            peak_rss=max(state["peak_rss"], process.memory_info().rss),
            peak_gpu_allocated=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved=torch.cuda.max_memory_reserved(),
        )
        atomic(output / "report.json", report)
        atomic(
            output / "status.json",
            dict(status="completed", condition=condition, finished_unix=time.time()),
        )
        print(
            json.dumps(
                dict(condition=condition, status="completed", test=summary["metrics"]["test"])
            ),
            flush=True,
        )
    except BaseException as exc:
        atomic(
            output / "status.json", dict(status="failed", condition=condition, error=repr(exc))
        )
        raise
    finally:
        runner.train_step = original


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
