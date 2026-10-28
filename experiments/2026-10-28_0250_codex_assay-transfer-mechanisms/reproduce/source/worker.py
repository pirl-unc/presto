"""Scaled comparisons through the single canonical fit_index trainer."""

import copy
import gzip
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


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
    with Path(path).open("rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def dump(path, rows):
    with Path(path).open("xb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as stream:
            for r in rows:
                stream.write(
                    (json.dumps(r, sort_keys=True, allow_nan=False) + "\n").encode()
                )


def common_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        if name.startswith("heads.") and not name.startswith("heads.ic50."):
            continue
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run(condition, output):
    pins = read("/bundle/source-pins.json")
    spec = pins["conditions"][condition]
    runtime = Path("/runtimes/shared/presto")
    sys.path.insert(0, str(runtime.parent))
    import psutil
    import torch
    from presto.data.modular_sources import ObservationIndex
    from presto.models.modular import MODEL_VERSION
    from presto.models import Presto, PrestoConfig
    from presto.training import modular_run as runner
    from presto.training.modular import load_checkpoint
    from presto.training.modular_evaluation import evaluate, summarize

    for rel, expected in pins["runtimes"]["shared"]["files"].items():
        assert sha(runtime / rel) == expected, rel
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    assert torch.cuda.is_available()
    gpu = torch.cuda.get_device_properties(0)
    assert "H100" in gpu.name
    torch.cuda.reset_peak_memory_stats()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "status.json").exists():
        raise ValueError("already attempted; no automatic retry")
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
        source=pins["runtimes"]["shared"],
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
    atomic(output / "status.json", dict(status="admitting", condition=condition))
    document = copy.deepcopy(pins["manifests"][condition])
    for source in document["sources"]:
        source["path"] = str(Path("/input") / source["path"])
    manifest = output / "manifest.json"
    atomic(manifest, document)
    started = time.monotonic()
    index = ObservationIndex(manifest)
    torch.manual_seed(document["training"]["seed"])
    initial = Presto(PrestoConfig.from_dict(document["model"]))
    initial_common_hash = common_hash(initial.state_dict())
    torch.manual_seed(document["training"]["seed"])
    comparator = Presto(
        PrestoConfig.from_dict({**document["model"], "endpoints": ["ic50"]})
    )
    assert common_hash(comparator.state_dict()) == initial_common_hash
    del initial, comparator
    atomic(
        output / "initialization.json",
        dict(
            initial_common_sha256=initial_common_hash,
            same_width_ic50_only_initialization_identical=True,
        ),
    )
    original = runner.train_step
    process = psutil.Process()
    state = dict(
        steps=0,
        examples=0,
        optimizer_seconds=0.0,
        class_examples={"I": 0, "II": 0},
        endpoint_examples={},
        endpoint_class_examples={},
        peak_rss=process.memory_info().rss,
    )

    def observed(model, optimizer, rows, scales, **kwargs):
        torch.cuda.synchronize()
        before = time.monotonic()
        result = original(model, optimizer, rows, scales, **kwargs)
        torch.cuda.synchronize()
        state["optimizer_seconds"] += time.monotonic() - before
        state["steps"] += 1
        state["examples"] += len(rows)
        for row in rows:
            state["class_examples"][row.mhc[0].mhc_class] += 1
            state["endpoint_examples"][row.endpoint] = (
                state["endpoint_examples"].get(row.endpoint, 0) + 1
            )
            key = row.endpoint + ":" + row.mhc[0].mhc_class
            state["endpoint_class_examples"][key] = (
                state["endpoint_class_examples"].get(key, 0) + 1
            )
        state["peak_rss"] = max(state["peak_rss"], process.memory_info().rss)
        if state["steps"] == 1 or state["steps"] % 25 == 0:
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
            with (output / "progress.jsonl").open("a") as f:
                f.write(json.dumps(point) + "\n")
        return result

    runner.train_step = observed
    atomic(
        output / "status.json",
        dict(status="training", condition=condition, started_unix=time.time()),
    )
    raw = output / "raw"
    try:
        summary = runner.fit_index(index, raw, device="cuda")
        assert summary["steps"] == spec["max_steps"]
        initial_payload = torch.load(
            raw / "initial.pt", map_location="cpu", weights_only=True
        )
        assert common_hash(initial_payload["state_dict"]) == initial_common_hash
        del initial_payload
        assert (
            state["endpoint_class_examples"] == pins["projected_exposures"][condition]
        )
        atomic(
            output / "status.json",
            dict(status="packaging_evaluation", condition=condition),
        )
        model, scales, payload = load_checkpoint(raw / "selected.pt", device="cuda")
        model.eval()
        checkpoint_hash = sha(raw / "selected.pt")
        report = dict(
            condition=condition,
            training_classes=["I", "II"],
            runtime=hardware,
            summary=summary,
            checkpoint=str(raw / "selected.pt"),
            checkpoint_sha256=checkpoint_hash,
            parameters=sum(p.numel() for p in model.parameters()),
            common_ic50={},
            mean_baselines={},
            selected_steps=payload["metadata"]["steps"],
            transfer={},
            initial_common_sha256=initial_common_hash,
            projected_exposures_match_actual=True,
            synthetic_auxiliary_labels=spec["synthetic_auxiliary_labels"],
        )
        for split in ("train", "validation", "test"):
            # The common trainer retains the full row-level dump; compressed copies are canonical artifacts.
            with (raw / f"{split}-predictions.jsonl").open() as f:
                rows = [json.loads(line) for line in f]
            dump(output / f"{split}-predictions.jsonl.gz", rows)
            if split != "train":
                original_ids = {
                    json.loads(line)["observation_id"]
                    for relative in pins["primary_ic50_files"][split]
                    for line in (Path("/input") / relative).open()
                }
                common = [r for r in rows if r["observation_id"] in original_ids]
                assert len(common) == len(original_ids)
                report["common_ic50"][split] = summarize(common)
                dump(output / f"common-ic50-{split}-predictions.jsonl.gz", common)
                means = []
                for row in rows:
                    scale = scales[row["endpoint"]]
                    value = dict(row, prediction=scale.mean)
                    residual = scale.mean - row["target"]
                    if row["qualifier"] in ("<", "<="):
                        residual = max(0.0, residual)
                    elif row["qualifier"] in (">", ">="):
                        residual = min(0.0, residual)
                    value["loss"] = (residual / scale.scale) ** 2
                    means.append(value)
                report["mean_baselines"][split] = summarize(means)
                dump(output / f"mean-{split}-predictions.jsonl.gz", means)
            del rows
        assert sha(raw / "selected.pt") == checkpoint_hash
        chosen = []
        counts = {}
        for i in index.partitions["train:assay"]:
            r = index[i]
            key = (r.endpoint, r.mhc[0].mhc_class)
            if counts.get(key, 0) < 4:
                chosen.append(i)
                counts[key] = counts.get(key, 0) + 1
        for i in index.partitions["train:assay"]:
            if len(chosen) >= 64:
                break
            if i not in chosen:
                chosen.append(i)
        _, replay = evaluate(model, index, chosen, scales, batch_size=64)
        dump(output / "training-replay.jsonl.gz", replay)
        report["class_examples"] = state["class_examples"]
        report["endpoint_examples"] = state["endpoint_examples"]
        report["endpoint_class_examples"] = state["endpoint_class_examples"]
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
                dict(
                    condition=condition,
                    status="completed",
                    validation=report["common_ic50"]["validation"],
                )
            ),
            flush=True,
        )
    except BaseException as exc:
        atomic(
            output / "status.json",
            dict(status="failed", condition=condition, error=repr(exc)),
        )
        raise
    finally:
        runner.train_step = original


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
