"""Serial fixed-budget comparison using frozen common trainers; live receipts are read-only."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

from support import (
    CONDITIONS,
    FAMILY,
    RAW,
    ROOT,
    RUNTIMES,
    atomic,
    diagnostic_view,
    guarded,
    instrument_steps,
    prediction_dump,
    read,
    sha,
)

SOURCE = Path(
    "/Users/iskander/code/presto/.worktrees/presto-modular-model/data/modular_cohorts/2026-10-21-full-ic50/manifest-exact.json"
)
SOURCE_SHA = "914e0930283a9e5f7a41024576976f3e92e280f0719feda261ffcf48bf01d7be"
COMMITS = dict(
    full="f24f7dfd6404c6eb450545daf72769269f55a5bf",
    three_aa="9dd71348124faa49a4c6a60872385c86895d50b0",
)
STEPS = 1000


def configure():
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    return torch


def runtime_receipt(label):
    import presto
    import torch
    from presto.models.modular import MODEL_VERSION
    from presto.training.modular_resume import SOURCE_FILES

    runtime = RUNTIMES[label]
    assert Path(presto.__file__).resolve().parent == runtime
    assert (
        MODEL_VERSION
        == {
            "shared": "presto.shared.v4",
            "full": "presto.shared.v3",
            "three_aa": "presto.modular.v2",
        }[label]
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=runtime, text=True
    ).strip()
    if label != "shared":
        assert commit == COMMITS[label]
    return dict(
        runtime=str(runtime),
        commit=commit,
        version=MODEL_VERSION,
        dirty=bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=runtime, text=True
            ).strip()
        ),
        source_hashes={p: sha(runtime / p) for p in SOURCE_FILES},
        python=sys.version,
        torch=str(torch.__version__),
        platform=platform.platform(),
        requested_modal_gpu=None,
        device="cpu",
        threads=4,
        interop_threads=1,
        environment={
            k: os.environ.get(k)
            for k in (
                "PYTHONPATH",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTORCH_ENABLE_MPS_FALLBACK",
            )
        },
    )


def frozen_check(label):
    frozen = read(FAMILY / "reproduce" / "runtime-pins.json")[label]
    current = runtime_receipt(label)
    assert current["source_hashes"] == frozen["source_hashes"], "runtime changed"
    for relative, expected in read(FAMILY / "reproduce" / "code-pins.json").items():
        assert sha(FAMILY / relative) == expected, relative
    admission = read(FAMILY / "results" / "admission.json")
    for path, expected in admission["derived_files"].items():
        assert sha(path) == expected, path
    return current


def prepare():
    from presto.data.modular import Observation
    from presto.data.modular_lineage import measurement_key
    from presto.data.modular_sources import ObservationIndex
    from presto.training.modular_run import preflight

    if (FAMILY / "results" / "admission.json").exists():
        raise ValueError("admission already exists")
    assert sha(SOURCE) == SOURCE_SHA
    original = read(SOURCE)
    inputs = RAW / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    groups = {
        (scope, split): []
        for scope in ("I", "II", "joint")
        for split in ("train", "validation", "test")
    }
    peptides, measurements = {}, {}
    counts, length_counts = Counter(), Counter()
    originals = {}
    for source in original["sources"]:
        path = Path(source["path"])
        if not path.is_absolute():
            path = (SOURCE.parent / path).resolve()
        assert sha(path) == source["sha256"], str(path)
        originals[str(path)] = source["sha256"]
        with path.open() as stream:
            for line in stream:
                data = json.loads(line)
                row = Observation.from_dict(data)
                classes = {m.mhc_class for m in row.mhc}
                assert len(classes) == 1 and classes <= {"I", "II"}
                cls = next(iter(classes))
                assert row.endpoint == "ic50" and row.unit == "nM"
                assert row.split != "train" or row.qualifier == "="
                old = peptides.setdefault(row.peptide, row.split)
                assert old == row.split, "cross-class peptide leakage"
                key = measurement_key(row)
                assert key, "missing original measurement identity"
                old = measurements.setdefault(key, row.split)
                assert old == row.split, "cross-class measurement leakage"
                groups[cls, row.split].append(line)
                groups["joint", row.split].append(line)
                counts[f"{row.split}:{cls}"] += 1
                length_counts[f"{row.split}:{cls}:{len(row.peptide)}"] += 1
    assert dict(counts) == {
        "train:I": 14375,
        "train:II": 7261,
        "validation:I": 2051,
        "validation:II": 874,
        "test:I": 1845,
        "test:II": 958,
    }
    derived, manifests, admitted = {}, {}, {}
    for scope in ("I", "II", "joint"):
        sources = []
        for split in ("train", "validation", "test"):
            path = inputs / f"{scope}-{split}.jsonl"
            with path.open("x") as stream:
                stream.writelines(groups[scope, split])
            derived[str(path)] = sha(path)
            sources.append(
                dict(
                    path=str(path),
                    sha256=sha(path),
                    source="hitlist-v4-iedb",
                    population="lossless class subset of admitted human direct IC50 cohort",
                    label_policy="physical unclipped exact IC50 training; exact and original bounds in holdouts",
                    split_policy="original exact-peptide/original-assay components; historically inspected; shared studies",
                )
            )
        for condition, spec in CONDITIONS.items():
            wanted = "joint" if len(spec["classes"]) == 2 else spec["classes"][0]
            if wanted != scope:
                continue
            manifest = dict(
                schema=original["schema"],
                sources=sources,
                model=dict(
                    endpoints=["ic50"],
                    identity_tasks=[],
                    mhc_classes=spec["classes"],
                    species=["human"],
                    dropout=0.1,
                ),
                training=dict(
                    seed=spec["seed"],
                    batch_size=64,
                    max_epochs=100,
                    max_steps=STEPS,
                    min_epochs=100,
                    patience=100,
                    learning_rate=3e-4,
                    weight_decay=1e-4,
                    identity_weight=0,
                    checkpoint_every_steps=50,
                ),
            )
            path = FAMILY / "reproduce" / f"{condition}-manifest.json"
            atomic(path, manifest)
            index = ObservationIndex(path)
            _, _, _, audit = preflight(index)
            admitted[condition] = audit["training_labels"]
            manifests[condition] = dict(path=str(path), sha256=sha(path), **spec)
            derived[str(path)] = sha(path)
    # A full-view manifest is used only as an observation index for held-out diagnostics.
    atomic(
        FAMILY / "manifest.json",
        dict(
            conditions=manifests,
            source_manifest=str(SOURCE),
            source_sha256=SOURCE_SHA,
            steps=STEPS,
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            evaluation_scope="fixed-budget, previously inspected development holdouts",
        ),
    )
    atomic(
        FAMILY / "results" / "admission.json",
        dict(
            status="passed",
            counts=dict(counts),
            lengths=dict(length_counts),
            original_files=originals,
            derived_files=derived,
            exact_peptide_cross_split_overlap=0,
            measurement_root_cross_split_overlap=0,
            training_labels=admitted,
        ),
    )
    print(json.dumps(dict(status="admitted", counts=dict(counts)), sort_keys=True), flush=True)


def profile(label):
    import statistics
    import torch
    from presto.data.modular_sources import ObservationIndex
    from presto.training.modular import train_step
    from presto.training.modular_run import preflight, initialized_model

    receipt = frozen_check(label)
    condition = {
        "shared": "shared-joint-s42",
        "full": "full-joint-s42",
        "three_aa": "three-aa-joint-s42",
    }[label]
    path = read(FAMILY / "manifest.json")["conditions"][condition]["path"]
    index = ObservationIndex(path)
    config, training, scales, _ = preflight(index)
    torch.manual_seed(42)
    model, _ = initialized_model(config, index)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    order = list(index.partitions["train:assay"])
    random.Random(42).shuffle(order)
    times, ids = [], []
    for i in range(8):
        rows = [index[j] for j in order[i * 64 : (i + 1) * 64]]
        ids.append([r.observation_id for r in rows])
        before = time.perf_counter()
        train_step(model, optimizer, rows, scales, identity_weight=0)
        duration = time.perf_counter() - before
        if i >= 3:
            times.append(duration)
    result = dict(
        runtime=receipt,
        condition=condition,
        warmup_steps=3,
        timed_steps=5,
        batch_ids=ids,
        seconds=times,
        median_step_seconds=statistics.median(times),
        examples_per_second=64 / statistics.median(times),
        parameters=sum(p.numel() for p in model.parameters()),
        target_split="train",
    )
    atomic(FAMILY / "results" / f"profile-{label}.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("runtime", "batch_ids")},
            sort_keys=True,
        ),
        flush=True,
    )


def summarize_mean(records, mean, scale):
    from presto.training.modular_evaluation import summarize

    values = []
    for row in records:
        r = dict(row, prediction=mean)
        residual = mean - r["target"]
        if r["qualifier"] in ("<", "<="):
            residual = max(0, residual)
        if r["qualifier"] in (">", ">="):
            residual = min(0, residual)
        r["loss"] = (residual / scale) ** 2
        values.append(r)
    return summarize(values), values


def fit(condition):
    import torch
    from presto.data.modular_sources import ObservationIndex
    from presto.training import modular_run as runner
    from presto.training.modular import load_checkpoint
    from presto.training.modular_evaluation import evaluate

    spec = CONDITIONS[condition]
    receipt = frozen_check(spec["runtime"])
    manifest = read(FAMILY / "manifest.json")["conditions"][condition]
    assert sha(manifest["path"]) == manifest["sha256"]
    index = ObservationIndex(manifest["path"])
    output = RAW / condition
    status = FAMILY / "results" / f"{condition}-status.json"
    if status.exists():
        raise ValueError("condition already attempted; no automatic retry")
    atomic(
        status,
        dict(status="training", condition=condition, started_unix=time.time(), runtime=receipt),
    )
    original = instrument_steps(runner, FAMILY / "results" / f"{condition}-progress.json")
    try:
        summary = runner.fit_index(index, output, device="cpu")
        assert summary["steps"] == STEPS
        atomic(status, dict(status="evaluating_transfer", condition=condition, runtime=receipt))
        source, scales, payload = load_checkpoint(output / "selected.pt")
        checkpoint_sha = sha(output / "selected.pt")
        report = dict(
            condition=condition,
            training_classes=spec["classes"],
            runtime=receipt,
            summary=summary,
            checkpoint=str(output / "selected.pt"),
            checkpoint_sha256=checkpoint_sha,
            parameters=sum(p.numel() for p in source.parameters()),
            transfer={},
            mean_baselines={},
        )
        destination = FAMILY / "results" / condition
        destination.mkdir(exist_ok=True)
        for name in (
            "summary.json",
            "history.json",
            "resolved-config.json",
            "coverage.json",
            "admission.json",
        ):
            shutil.copy2(output / name, destination / name)
        for split in ("train", "validation", "test"):
            with (output / f"{split}-predictions.jsonl").open() as stream:
                records = [json.loads(line) for line in stream]
            prediction_dump(destination / f"{split}-predictions.jsonl.gz", records)
            if split != "train":
                metrics, means = summarize_mean(
                    records, scales["ic50"].mean, scales["ic50"].scale
                )
                report["mean_baselines"][split] = metrics
                prediction_dump(destination / f"mean-{split}-predictions.jsonl.gz", means)
            del records
        if len(spec["classes"]) == 1:
            target = "II" if spec["classes"] == ["I"] else "I"
            full_index = ObservationIndex(
                read(FAMILY / "manifest.json")["conditions"]["full-joint-s42"]["path"]
            )
            view, contract = diagnostic_view(source)
            key = contract["protocol"]
            report["transfer"][key] = dict(contract=contract, target_class=target, metrics={})
            for split in ("validation", "test"):
                indices = [
                    i
                    for i in full_index.partitions[f"{split}:assay"]
                    if full_index[i].mhc[0].mhc_class == target
                ]
                metrics, records = evaluate(view, full_index, indices, scales, batch_size=64)
                for row in records:
                    row.update(
                        evaluation_scope="zero_shot_" + key,
                        training_classes=spec["classes"],
                        capability_status="unsupported_class_transfer_diagnostic",
                    )
                prediction_dump(destination / f"{key}-{split}-predictions.jsonl.gz", records)
                report["transfer"][key]["metrics"][split] = metrics
                means, mean_records = summarize_mean(
                    records, scales["ic50"].mean, scales["ic50"].scale
                )
                report["mean_baselines"][f"{split}:transfer"] = means
                prediction_dump(
                    destination / f"mean-transfer-{split}-predictions.jsonl.gz", mean_records
                )
                del records, mean_records
            del view
        assert sha(output / "selected.pt") == checkpoint_sha
        # Checkpoint replay reference uses TRAIN rows only; held-outs will not be rescored for verification.
        first = list(index.partitions["train:assay"])[:64]
        _, replay = evaluate(source, index, first, scales, batch_size=64)
        prediction_dump(destination / "training-replay.jsonl.gz", replay)
        atomic(destination / "report.json", report)
        atomic(
            status,
            dict(
                status="completed",
                condition=condition,
                runtime=receipt,
                finished_unix=time.time(),
                report=str(destination / "report.json"),
            ),
        )
        print(
            json.dumps(
                dict(
                    condition=condition,
                    status="completed",
                    steps=summary["steps"],
                    metrics=summary["metrics"]["test"],
                ),
                sort_keys=True,
            ),
            flush=True,
        )
    except BaseException as exc:
        atomic(
            status, dict(status="failed", condition=condition, error=repr(exc), runtime=receipt)
        )
        raise
    finally:
        runner.train_step = original


def child(action, label, condition=None):
    runtime = RUNTIMES[label]
    env = {
        **os.environ,
        "PYTHONPATH": str(runtime.parent),
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    }
    command = [sys.executable, str(Path(__file__).resolve()), action, "--runtime", label]
    if condition:
        command += ["--condition", condition]
    name = condition or f"{action}-{label}"
    log = FAMILY / "launch_logs" / f"{name}.log"
    with log.open("x") as stream:
        result = subprocess.run(
            command, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise RuntimeError(f"{name} failed with code {result.returncode}; see {log}")


def run_all():
    state = FAMILY / "reproduce" / "controller.json"
    if state.exists():
        raise ValueError("controller already attempted")
    atomic(state, dict(status="running", completed=[], started_unix=time.time()))
    done = []
    try:
        for label in ("three_aa", "full", "shared"):
            child("profile", label)
        for condition, spec in CONDITIONS.items():
            atomic(state, dict(status="running", active=condition, completed=done))
            child("fit", spec["runtime"], condition)
            done.append(condition)
        atomic(state, dict(status="fits_completed", completed=done, finished_unix=time.time()))
    except BaseException as exc:
        atomic(state, dict(status="failed", completed=done, error=repr(exc)))
        raise


def freeze():
    # Run pin helpers in each immutable runtime, without model fitting.
    pins = {}
    for label in ("shared", "full", "three_aa"):
        runtime = RUNTIMES[label]
        env = {**os.environ, "PYTHONPATH": str(runtime.parent), "PYTHONDONTWRITEBYTECODE": "1"}
        command = [sys.executable, str(Path(__file__).resolve()), "pins", "--runtime", label]
        pins[label] = json.loads(subprocess.check_output(command, env=env, cwd=ROOT, text=True))
        for relative in pins[label]["source_hashes"]:
            dest = FAMILY / "reproduce" / "source" / label / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(runtime / relative, dest)
    atomic(FAMILY / "reproduce" / "runtime-pins.json", pins)
    files = [p for p in (FAMILY / "code").glob("*") if p.suffix in (".py", ".html")]
    code_pins = {str(p.relative_to(FAMILY)): sha(p) for p in files}
    atomic(FAMILY / "reproduce" / "code-pins.json", code_pins)
    for p in files:
        target = FAMILY / "reproduce" / "source" / p.name
        shutil.copy2(p, target)
    invocation = dict(
        agent_model="Codex / GPT-6",
        utc=datetime.now(timezone.utc).isoformat(),
        argv=[sys.executable, str(Path(__file__).resolve()), "all"],
        cwd=str(ROOT),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        git_dirty=bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        runtime_commits={label: value["commit"] for label, value in pins.items()},
        source_pins=code_pins,
        conditions=CONDITIONS,
        raw_artifacts=str(RAW),
        environment=dict(
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            PYTHONDONTWRITEBYTECODE="1",
            PYTORCH_ENABLE_MPS_FALLBACK="0",
        ),
    )
    atomic(FAMILY / "reproduce" / "launch.json", invocation)
    (FAMILY / "reproduce" / "launch.sh").write_text(
        "#!/bin/sh\nset -eu\n"
        + f'cd "{ROOT}"\nexec "{sys.executable}" "{Path(__file__).resolve()}" all\n'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "freeze", "pins", "profile", "fit", "all")
    )
    parser.add_argument("--runtime", choices=("shared", "full", "three_aa"), default="shared")
    parser.add_argument("--condition", choices=tuple(CONDITIONS))
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.action == "all":
        run_all()
        return
    if args.action == "freeze":
        freeze()
        return
    configure()
    if args.action == "pins":
        print(json.dumps(runtime_receipt(args.runtime)))
        return
    if args.attempt != 1 and args.action != "prepare":
        raise ValueError("only a manually reviewed admission attempt may have a new receipt")
    name = args.condition or f"{args.action}-{args.runtime}"
    if args.attempt != 1:
        name += f"-attempt{args.attempt}"
    with guarded(name):
        if args.action == "prepare":
            prepare()
        elif args.action == "profile":
            profile(args.runtime)
        elif args.action == "fit":
            if not args.condition:
                raise ValueError("fit requires condition")
            fit(args.condition)
    gc.collect()


if __name__ == "__main__":
    main()
