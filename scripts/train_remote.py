#!/usr/bin/env python
"""Launch Presto unified training on a remote GPU via runplz.

Modeled on mhcflurry's `scripts/training/launch_pan_allele_training_remote.py`,
which is the working reference for this pattern in the sibling repo.

Invoke through the runplz CLI, never directly:

    runplz brev  --outputs-dir ./brev_runs/<name> --instance <box> scripts/train_remote.py
    runplz modal --outputs-dir ./modal_runs/<name>                 scripts/train_remote.py
    runplz local --outputs-dir ./out/<name>                        scripts/train_remote.py

Backends are interchangeable; `runplz modal` keeps the historical Modal path
available without a second launcher to maintain.

Data
----
This launcher runs **hitlist-sourced** training, which needs no merged TSV. That
is deliberate: `data/merged_deduped.tsv` is a gitignored 1 GB file, and runplz
stages the repo git-aware, so shipping it to every worker would be both slow and
a reproducibility hazard. hitlist-only mode covers binding, stability, kinetics,
elution and the non-MHC shotgun corpus; T-cell, TCR and IEDB processing rows are
absent, which is fine for the Stage 4 factorial (its metric is elution AUPRC).

hitlist caches its built indexes under `~/.hitlist/`. On a persistent Brev box
that cache survives between runs, so only the first run pays the build cost.
Set `PRESTO_HITLIST_BUILD=1` to build on the worker when the cache is cold;
expect it to take a while the first time.

Corpus size
-----------
``PRESTO_MAX_*`` caps are per-modality and ``0`` means unlimited. Capping
elution and binding does not bound the epoch, because ``--bulk-ms`` pulls the
shotgun corpus and ``PRESTO_MAX_BULK_MS`` defaults to 0. Set every cap you care
about, and check the printed ``Train batches:`` line before walking away.

Throughput
----------
``--num-workers`` defaults to 0 in the trainer, which starves the GPU: a
1.01 s/batch run sat at 33% GPU with load 1.68 on a 48-core box. Set it.

Batch size
----------
Peak memory is not linear in ``PRESTO_BATCH_SIZE``. Core-window scoring expands
the batch by the number of candidate registers (``peptide_len - core_window +
1``) and repeats the full MHC hidden states for each, so the effective batch is
``batch_size * candidates``. On a 40 GB A100, ``PRESTO_BATCH_SIZE=256`` with
class I data OOMs at ~36 GB; 64 is comfortable. Scale it down before scaling
``--max-mil-instances`` down -- the MIL cap is a global per-batch instance cap
and is rarely what is filling the card.

Hardware defaults to **one cheap GPU**, chosen by resource floors rather than a
pinned model name. Presto training is single-GPU; the Stage 4 factorial is 15
independent runs, so a multi-GPU box would idle most of what it bills for. Set
``RUNPLZ_GPU`` to pin a specific part when a run genuinely needs one.

Everything is configured by environment variable so the same file serves every
experiment arm without edits — freeze the env in the experiment's
`reproduce/launch.sh`, per AGENTS.md.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys

try:
    from runplz import App, BrevConfig, Image
except ImportError as exc:  # pragma: no cover - exercised via install docs
    raise SystemExit(
        "runplz is required for remote launches. Install it with "
        "`pip install runplz`, or run `python -m presto train unified` locally."
    ) from exc


APP_NAME = os.environ.get("RUNPLZ_APP_NAME", "presto-train")
BASE_IMAGE = os.environ.get(
    "RUNPLZ_IMAGE", "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"
)
# Default to one cheap GPU, not a multi-GPU box. Presto training is a single-GPU
# job: the Stage 4 factorial is 15 independent runs (5 arms x 3 seeds), so a
# 4xA100 instance would idle three GPUs per run and bill for them. Expressed as
# a memory floor rather than a model name so the selector picks the cheapest
# match rather than a fixed (expensive) part.
GPU_TYPE = os.environ.get("RUNPLZ_GPU", "")  # empty -> selector chooses by the floors below
MIN_GPUS = int(os.environ.get("RUNPLZ_MIN_GPUS", "1"))
MIN_GPU_MEMORY = int(os.environ.get("RUNPLZ_MIN_GPU_MEMORY", "16"))
MIN_CPU = int(os.environ.get("RUNPLZ_MIN_CPU", "4"))
MIN_MEMORY = int(os.environ.get("RUNPLZ_MIN_MEMORY", "32"))
MIN_DISK = int(os.environ.get("RUNPLZ_MIN_DISK", "100"))
TIMEOUT_SECONDS = int(os.environ.get("RUNPLZ_TIMEOUT_SECONDS", str(12 * 60 * 60)))

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    token = raw.strip().lower()
    if token in TRUE_VALUES:
        return True
    if token in FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} must be one of {sorted(TRUE_VALUES)} or {sorted(FALSE_VALUES)}; "
        f"got {raw!r}"
    )


def brev_config_from_env() -> BrevConfig:
    """Brev settings, mirroring mhcflurry's env contract so the two agree."""
    return BrevConfig(
        auto_create_instances=env_bool("RUNPLZ_BREV_AUTO_CREATE", default=False),
        instance_type=os.environ.get("RUNPLZ_BREV_INSTANCE_TYPE") or None,
        mode=os.environ.get("RUNPLZ_BREV_MODE", "container"),
        on_finish=os.environ.get("RUNPLZ_BREV_ON_FINISH", "leave"),
        ssh_ready_wait_seconds=int(
            os.environ.get("RUNPLZ_BREV_SSH_READY_WAIT_SECONDS", "2400")
        ),
    )


app = App(APP_NAME, brev_config=brev_config_from_env())

image = (
    Image.from_registry(BASE_IMAGE)
    .apt_install("rsync", "git")
    .pip_install("hitlist>=1.41", "pandas>=2.0", "pyarrow")
    .pip_install_local_dir(".", editable=True)
)


def training_argv() -> list[str]:
    """Build the `presto train unified` argv from the environment.

    Kept as an explicit list rather than a free-form string so a typo produces
    an argparse error on the worker instead of a silently different run.
    """
    out_dir = os.environ.get("RUNPLZ_OUT", "/out")
    argv = [
        sys.executable,
        "-u",
        "-m",
        "presto",
        "train",
        "unified",
        "--data-source",
        "hitlist",
        "--run-dir",
        out_dir,
        "--checkpoint",
        f"{out_dir}/model.pt",
        "--epochs",
        os.environ.get("PRESTO_EPOCHS", "50"),
        "--batch_size",
        os.environ.get("PRESTO_BATCH_SIZE", "256"),
        "--d_model",
        os.environ.get("PRESTO_D_MODEL", "128"),
        "--n_layers",
        os.environ.get("PRESTO_N_LAYERS", "2"),
        "--n_heads",
        os.environ.get("PRESTO_N_HEADS", "4"),
        "--lr",
        os.environ.get("PRESTO_LR", "3e-4"),
        "--seed",
        os.environ.get("PRESTO_SEED", "42"),
        "--latent-topology",
        os.environ.get("PRESTO_LATENT_TOPOLOGY", "expanded"),
    ]

    mhc_class = os.environ.get("PRESTO_HITLIST_MHC_CLASS")
    if mhc_class:
        argv += ["--hitlist-mhc-class", mhc_class]
    allele = os.environ.get("PRESTO_HITLIST_ALLELE")
    if allele:
        argv += ["--hitlist-allele", allele]

    if env_bool("PRESTO_BULK_MS", default=True):
        argv.append("--bulk-ms")
        cell_line = os.environ.get("PRESTO_BULK_CELL_LINE")
        if cell_line:
            argv += ["--bulk-cell-line", cell_line]
        argv += [
            # NB: 0 means *unlimited*, not "none" -- `int(max_bulk_ms) or None`
            # downstream. The shotgun corpus is by far the largest modality, so
            # leaving this at 0 while capping elution and binding still yields
            # a multi-million-row epoch: one run took 3.19M samples and 28h for
            # two epochs before this was noticed. Set it deliberately.
            "--max-bulk-ms",
            os.environ.get("PRESTO_MAX_BULK_MS", "0"),
            "--bulk-excision-negative-ratio",
            os.environ.get("PRESTO_EXCISION_NEGATIVE_RATIO", "1.0"),
        ]

    for env_name, flag in (
        ("PRESTO_MAX_BINDING", "--max-binding"),
        ("PRESTO_MAX_ELUTION", "--max-elution"),
        ("PRESTO_MAX_STABILITY", "--max-stability"),
        ("PRESTO_MAX_KINETICS", "--max-kinetics"),
    ):
        value = os.environ.get(env_name)
        if value:
            argv += [flag, value]

    extra = os.environ.get("PRESTO_EXTRA_ARGS", "").strip()
    if extra:
        argv += shlex.split(extra)
    return argv


#: Config forwarded to the worker.
#:
#: `training_argv()` runs remotely, so it reads the *worker's* environment, not
#: the driver's. Without forwarding, every PRESTO_* setting silently reverts to
#: its default on the box and the run is not the one that was launched.
#:
#: CUDA_VISIBLE_DEVICES is pinned to a single GPU. The org's Brev instances are
#: 4xA100, but presto training is single-GPU work: taking one device leaves the
#: other three free rather than idling them under this job.
def worker_env() -> dict:
    forwarded = {
        name: value
        for name, value in os.environ.items()
        if name.startswith("PRESTO_")
    }
    forwarded.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("PRESTO_CUDA_DEVICE", "0"))
    # MIL bags make peak memory scale with batch_size * max_mil_instances, not
    # batch_size, so allocation is bursty and fragments badly. Expandable
    # segments let the allocator grow a block instead of failing next to
    # several GB of reserved-but-unallocated memory.
    forwarded.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
    )
    # Deliberately NOT forwarding RUNPLZ_OUT: runplz sets it on the worker to
    # the run directory it later collects from. Overriding it to /out sends
    # every artifact somewhere runplz does not rsync back, so the run looks
    # successful and returns nothing.
    return forwarded


_FUNCTION_KWARGS = dict(
    image=image,
    env=worker_env(),
    min_gpus=MIN_GPUS,
    min_gpu_memory=MIN_GPU_MEMORY,
    min_cpu=MIN_CPU,
    min_memory=MIN_MEMORY,
    min_disk=MIN_DISK,
    timeout=TIMEOUT_SECONDS,
)
if GPU_TYPE:
    # Only pin a model when explicitly asked; otherwise the selector picks the
    # cheapest part meeting the floors.
    _FUNCTION_KWARGS["gpu"] = GPU_TYPE


@app.function(**_FUNCTION_KWARGS)
def train() -> None:
    import json
    from pathlib import Path

    out_dir = Path(os.environ.get("RUNPLZ_OUT", "/out"))
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch

        print(f"cuda available: {torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(index)
                total = torch.cuda.get_device_properties(index).total_memory / 1e9
                print(f"  gpu{index}: {name} ({total:.1f} GB)", flush=True)
    except ImportError:  # pragma: no cover - torch is in the base image
        print("torch unavailable", flush=True)

    # Fail loudly rather than training on nothing: a shadowed or unbuilt
    # hitlist would otherwise surface as an empty corpus much later.
    import hitlist

    if getattr(hitlist, "__file__", None) is None:
        raise SystemExit("hitlist resolved to an empty namespace package")

    if env_bool("PRESTO_HITLIST_BUILD", default=False):
        print("Building hitlist indexes (cold cache)...", flush=True)
        subprocess.run(["hitlist", "build", "observations"], check=True)

    argv = training_argv()
    print("Command: " + " ".join(shlex.quote(part) for part in argv), flush=True)
    (out_dir / "launch_argv.json").write_text(json.dumps(argv, indent=2))

    completed = subprocess.run(argv, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"presto train unified failed rc={completed.returncode}")

    produced = sorted(path.name for path in out_dir.iterdir())
    print(f"Outputs in {out_dir}: {produced}", flush=True)
    # The experiment contract wants held-out metrics, not just a checkpoint.
    for required in ("summary.json", "val_metrics.csv"):
        if required not in produced:
            raise SystemExit(
                f"expected {required} in the run directory; the held-out pass "
                "did not complete"
            )


@app.local_entrypoint()
def main() -> None:
    train.remote()
