#!/usr/bin/env python
"""Paired Modal comparison of legacy source selection and unknown masking."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

CODE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = CODE_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

GPU = "H100!"
HITLIST_VERSION = "1.55.8"
MHCSEQS_VERSION = "2.6.10"
MHCGNOMES_VERSION = "3.41.0"
MHC_INDEX_PATH = "/opt/presto/data/mhc_index.csv"
MHC_INDEX_SHA256 = "497938937f01394aeb18a3db15314f04ac1be162efe2844a1f018bcaff121063"
BASE_IMAGE = "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"
APP_NAME = "presto-groove-corrected-baseline"
CHECKPOINT_VOLUME_NAME = "presto-checkpoints"
DATA_VOLUME_NAME = "presto-data"
PREFIX = "presto-groove-corrected-20260904a"
POLICIES = ("legacy_global_canonical", "mask_unresolved")
SEEDS = (42, 43, 44)
EPOCHS = 10
BATCH_SIZE = 256
DATA_SEED = 42
REQUIRED_RUN_FILES = (
    "config.json",
    "run.log",
    "metrics.csv",
    "model.pt",
    "mhc_sequence_coverage.json",
    "mhc_sequence_coverage.csv",
    "val_summary.json",
    "test_summary.json",
    "val_metrics.csv",
    "test_metrics.csv",
    "val_predictions.csv",
    "test_predictions.csv",
    "condition_result.json",
    "data_contract.json",
    "hardware.json",
    "split_support.json",
    "split_support.csv",
    "data_funnel.json",
    "data_funnel.csv",
)
DATA_HASHES = {
    "observations.parquet": "f51440ab229fd187d2548b4dddcd1fc04580d97d45fb4d5b8e0222aa8080f928",
    "binding.parquet": "fbcae6f762f43edb4eb87b1a7c7f3849757859204d37f3ce82d1f83c23cece9f",
    "peptide_mappings.parquet": (
        "45580c16649daf75b11b51497d9d96dfaa987cdcf1197d6449deaa9261f6ec5c"
    ),
    "observations_meta.json": ("ac459e184fc6c54c73f7f1b4fc7dff424b2360b13f55f12bb68a3bbb743de118"),
    "peptide_mappings_meta.json": (
        "9a93de21753029ac08f1ba05e1a667ee6d7fd1b63a885bdc8dade1e626c7cbbb"
    ),
}


def _build_image() -> modal.Image:
    # Add only the installable package surface. Uploading the worktree root
    # also walks multi-GB data, artifact, experiment, and sibling caches; an
    # earlier launch spent long enough indexing those that the client heartbeat
    # expired before the image build began.
    image = modal.Image.from_registry(BASE_IMAGE).run_commands(
        "python -m pip install --upgrade pip",
        (
            # 2.6.10, not 2.5.12: the older release starts the mature chain
            # one or two residues early on ~1.8% of alleles (class II DOA/DPA
            # especially), which is the same off-by-one this family exists to
            # remove from presto's own parser. Measured across 600 index
            # alleles, the two versions disagree on 11.
            f"python -m pip install mhcseqs=={MHCSEQS_VERSION} "
            f"mhcgnomes=={MHCGNOMES_VERSION} "
            "numpy pyyaml tqdm matplotlib pandas pyarrow"
        ),
        f"python -m pip install hitlist=={HITLIST_VERSION}",
    )
    for directory in ("cli", "data", "inference", "models", "scripts", "training"):
        image = image.add_local_dir(
            str(REPO_ROOT / directory),
            remote_path=f"/opt/presto/{directory}",
            # Source is mounted when each container starts.  Baking every
            # directory as a separate image layer caused a chain of redundant
            # image builds; Presto is imported directly through PYTHONPATH.
            copy=False,
            # ``data/`` contains multi-GB downloaded corpora next to the
            # loader modules.  The experiment reads its frozen inputs from
            # the Modal volume, so only package source belongs in the image.
            ignore=~modal.FilePatternMatcher("**/*.py"),
        )
    for filename in (
        "__init__.py",
        "__main__.py",
        "pyproject.toml",
        "README.md",
        "LICENSE",
    ):
        image = image.add_local_file(
            str(REPO_ROOT / filename),
            remote_path=f"/opt/presto/{filename}",
            copy=False,
        )
    image = image.add_local_file(
        str(REPO_ROOT / "data" / "b2m_sequences.csv"),
        remote_path="/opt/presto/data/b2m_sequences.csv",
        copy=False,
    )
    image = image.add_local_file(
        str(REPO_ROOT / "data" / "mhc_index.csv"),
        remote_path=MHC_INDEX_PATH,
        copy=False,
    )
    return image


image = _build_image()
app = modal.App(APP_NAME, image=image)
checkpoints_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_package_versions() -> dict[str, str]:
    expected = {
        "hitlist": HITLIST_VERSION,
        "mhcseqs": MHCSEQS_VERSION,
        "mhcgnomes": MHCGNOMES_VERSION,
    }
    observed = {package: importlib.metadata.version(package) for package in expected}
    if observed != expected:
        raise RuntimeError(
            f"Package versions differ from the frozen contract: expected={expected}, "
            f"observed={observed}"
        )
    return observed


def _run_id(policy: str, seed: int) -> str:
    token = "legacy" if policy == "legacy_global_canonical" else "masked"
    return f"{PREFIX}-{token}-e{EPOCHS:03d}-s{seed}"


def _command(
    policy: str,
    seed: int,
    run_dir: Path,
    *,
    expected_support_hash: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "presto",
        "train",
        "unified",
        "--data-dir",
        "/tmp/presto-hitlist-only",
        "--data-source",
        "hitlist",
        "--source-mapping-policy",
        policy,
        "--hitlist-allele",
        "HLA-A*02:01",
        "--index-csv",
        MHC_INDEX_PATH,
        "--max-elution",
        "20000",
        "--max-stability",
        "1000",
        "--max-kinetics",
        "500",
        "--cap-sampling",
        "reservoir",
        "--data-seed",
        str(DATA_SEED),
        "--synthetic-pmhc-negative-ratio",
        "0",
        "--synthetic-elution-negative-ratio",
        "1",
        "--synthetic-cascade-elution-negative-ratio",
        "0",
        "--synthetic-cascade-tcell-negative-ratio",
        "0",
        "--synthetic-class-i-no-mhc-beta-negative-ratio",
        "0",
        "--synthetic-processing-negative-ratio",
        "0",
        "--exclude-target",
        "kon",
        "--exclude-target",
        "koff",
        "--exclude-target",
        "tm",
        "--mhc-augmentation-samples",
        "0",
        "--uniprot-negative-ratio",
        "0",
        "--val-frac",
        "0.1",
        "--test-frac",
        "0.1",
        "--split-mode",
        "peptide_group",
        "--epochs",
        str(EPOCHS),
        "--batch_size",
        str(BATCH_SIZE),
        "--d_model",
        "128",
        "--n_layers",
        "2",
        "--n_heads",
        "4",
        "--latent-topology",
        "expanded",
        "--lr",
        "2.8e-4",
        "--weight_decay",
        "0.01",
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--num-workers",
        "4",
        "--no-balanced-batches",
        "--no-compile",
        "--no-track-probe-affinity",
        "--no-track-probe-motif-scan",
        "--no-track-pmhc-flow",
        "--no-track-output-latent-stats",
        "--require-all-active-target-support",
        "--require-binary-balance-target",
        "elution",
        "--require-all-active-binary-balance",
        "--require-traceable-lineage",
        "--forbid-fake-null-sequences",
        "--run-dir",
        str(run_dir),
        "--checkpoint",
        str(run_dir / "model.pt"),
    ]
    if expected_support_hash:
        command.extend(["--expected-split-support-sha256", expected_support_hash])
    return command


def _replace_flag_value(command: list[str], flag: str, value: str) -> None:
    command[command.index(flag) + 1] = value


@app.function(
    cpu=4,
    memory=16384,
    timeout=60 * 60,
    volumes={"/data": data_volume},
)
def preflight_condition(policy: str, seed: int) -> dict[str, Any]:
    """Build and gate one exact production dataset without starting training."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    package_versions = _validated_package_versions()
    os.environ["HITLIST_DATA_DIR"] = "/data/hitlist"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONPATH"] = "/opt"
    Path("/tmp/presto-hitlist-only").mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"presto-groove-preflight-{policy}-{seed}-"))

    observed_hashes = {name: _sha256(Path("/data/hitlist") / name) for name in DATA_HASHES}
    if observed_hashes != DATA_HASHES:
        raise RuntimeError(f"Preflight data hash mismatch: {observed_hashes}")
    observed_mhc_index_hash = _sha256(Path(MHC_INDEX_PATH))
    if observed_mhc_index_hash != MHC_INDEX_SHA256:
        raise RuntimeError(f"Preflight MHC index hash mismatch: {observed_mhc_index_hash}")

    command = _command(policy, seed, run_dir)
    for flag, value in {
        "--device": "cpu",
        "--num-workers": "0",
    }.items():
        _replace_flag_value(command, flag, value)
    command.append("--data-preflight-only")
    process = subprocess.run(
        command,
        cwd="/opt/presto",
        env=os.environ.copy(),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(process.stdout, end="", flush=True)
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
    required = (
        "config.json",
        "split_support.json",
        "split_support.csv",
        "data_funnel.json",
        "data_funnel.csv",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Preflight is missing required artifacts: {missing}")
    support = json.loads((run_dir / "split_support.json").read_text())
    return {
        "status": "complete",
        "policy": policy,
        "seed": seed,
        "artifact_sha256": observed_hashes,
        "mhc_index_sha256": observed_mhc_index_hash,
        "package_versions": package_versions,
        "required_outputs": list(required),
        "split_support_sha256": support["sha256"],
        "dataset_contract_sha256": support["dataset_contract_sha256"],
        "dataset_supervision_contract_sha256": support["dataset_supervision_contract_sha256"],
        "split_supervision_contract_sha256": {
            name: split["supervision_contract_sha256"] for name, split in support["splits"].items()
        },
        "support": support,
    }


@app.function(
    gpu=GPU,
    timeout=6 * 60 * 60,
    volumes={"/checkpoints": checkpoints_volume, "/data": data_volume},
)
def train_condition(policy: str, seed: int, expected_support_hash: str) -> dict[str, Any]:
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    package_versions = _validated_package_versions()
    os.environ["HITLIST_DATA_DIR"] = "/data/hitlist"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONPATH"] = "/opt"
    Path("/tmp/presto-hitlist-only").mkdir(parents=True, exist_ok=True)

    run_id = _run_id(policy, seed)
    run_dir = Path("/checkpoints") / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            f"Refusing to reuse non-empty run directory {run_dir}; choose a new PREFIX "
            "or explicitly archive the prior attempt."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    started_at = datetime.now(UTC).isoformat()

    observed_hashes = {name: _sha256(Path("/data/hitlist") / name) for name in DATA_HASHES}
    if observed_hashes != DATA_HASHES:
        raise RuntimeError(
            "Hitlist artifact hashes do not match the frozen experiment contract: "
            f"{observed_hashes}"
        )
    observed_mhc_index_hash = _sha256(Path(MHC_INDEX_PATH))
    if observed_mhc_index_hash != MHC_INDEX_SHA256:
        raise RuntimeError(
            f"MHC index does not match the frozen experiment contract: {observed_mhc_index_hash}"
        )

    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    hardware = {
        "requested_gpu": GPU,
        "nvidia_smi": gpu_query,
        "package_versions": package_versions,
    }
    (run_dir / "hardware.json").write_text(
        json.dumps(hardware, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "data_contract.json").write_text(
        json.dumps(
            {
                "artifact_sha256": observed_hashes,
                "mhc_index_sha256": observed_mhc_index_hash,
                "hitlist_version": HITLIST_VERSION,
                "mhcseqs_version": MHCSEQS_VERSION,
                "mhcgnomes_version": MHCGNOMES_VERSION,
                "allele_filter": "HLA-A*02:01",
                "elution_allele_semantics": "retain complete co-expressed source allele set",
                "policy": policy,
                "seed": seed,
                "data_seed": DATA_SEED,
                "preflight_split_support_sha256": expected_support_hash,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    cmd = _command(
        policy,
        seed,
        run_dir,
        expected_support_hash=expected_support_hash,
    )
    try:
        with (run_dir / "run.log").open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                cmd,
                cwd="/opt/presto",
                env=os.environ.copy(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
                log_handle.flush()
            returncode = process.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)

        required = tuple(name for name in REQUIRED_RUN_FILES if name != "condition_result.json")
        missing = [name for name in required if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Completed process is missing required artifacts: {missing}")

        result = {
            "run_id": run_id,
            "policy": policy,
            "seed": seed,
            "status": "complete",
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "runtime_seconds": time.time() - started,
            "hardware": hardware,
            "remote_run_dir": str(run_dir),
            "required_files": list(REQUIRED_RUN_FILES),
        }
        (run_dir / "condition_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Run bundle is incomplete after finalization: {missing}")
        return result
    finally:
        checkpoints_volume.commit()


def _metadata() -> dict[str, Any]:
    return {
        "responsibility": {
            "initial_design": "Claude",
            "audit_remediation_and_launch_gate": "Codex / GPT-5",
        },
        "dataset_contract": {
            "source": f"hitlist=={HITLIST_VERSION}",
            "mhc_parser": f"mhcseqs=={MHCSEQS_VERSION}",
            "allele_parser": f"mhcgnomes=={MHCGNOMES_VERSION}",
            "allele_filter": (
                "HLA-A*02:01; binding is single-allele, while elution retains each "
                "selected sample's complete co-expressed allele set"
            ),
            "binding": "all qualifying numeric rows",
            "elution_cap": 20_000,
            "stability_cap": 1_000,
            "included_targets": [
                "binding IC50/EC50/KD families",
                "stability half-life",
                "elution with synthetic decoys",
            ],
            "excluded_targets": ["kon", "koff", "Tm"],
            "excluded_modalities": [
                "processing",
                "T-cell",
                "TCR evidence",
                "bulk MS",
                "MHC augmentation",
                "UniProt negatives",
            ],
            "scope": (
                "narrow HLA-A*02:01 affinity/stability and elution-decoy study; "
                "not a complete unified multitask corpus"
            ),
            "split": "peptide-disjoint 80/10/10",
            "data_seed": DATA_SEED,
            "split_seeds": list(SEEDS),
            "artifact_sha256": DATA_HASHES,
            "mhc_index_sha256": MHC_INDEX_SHA256,
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "model": "Presto expanded d128/l2/h4",
            "optimizer": "AdamW lr=2.8e-4 weight_decay=0.01",
            "synthetic_data": (
                "binding pMHC ratio 0; elution decoy ratio 1.0; binding-to-elution and "
                "binding-to-T-cell cascades 0. Decoy metrics are not biological "
                "real-negative performance."
            ),
            "gpu": GPU,
        },
        "tested": [{"source_mapping_policy": policy, "seeds": list(SEEDS)} for policy in POLICIES],
    }


@app.local_entrypoint()
def main(preflight_only: bool = False) -> None:
    # This helper exists only in the local repository.  Keep the import out of
    # module scope because Modal imports this file again inside each worker.
    from experiment_registry import initialize_experiment_dir

    preflight_calls = {
        (policy, seed): preflight_condition.spawn(policy, seed)
        for seed in SEEDS
        for policy in POLICIES
    }
    preflight_results = {key: call.get() for key, call in preflight_calls.items()}

    # The experimental policy may change only the flank values and terminus
    # state. Every source observation, target, cap choice, and split assignment
    # must remain paired within a seed.
    for seed in SEEDS:
        legacy = preflight_results[("legacy_global_canonical", seed)]
        masked = preflight_results[("mask_unresolved", seed)]
        if (
            legacy["dataset_supervision_contract_sha256"]
            != masked["dataset_supervision_contract_sha256"]
        ):
            raise RuntimeError(
                f"Preflight supervision dataset mismatch between policies for seed {seed}"
            )
        if (
            legacy["split_supervision_contract_sha256"]
            != masked["split_supervision_contract_sha256"]
        ):
            raise RuntimeError(f"Preflight split mismatch between policies for seed {seed}")

    # Capped and synthetic data are controlled by DATA_SEED, so model/split
    # seeds may move rows among partitions but may not change dataset membership.
    dataset_hashes = {
        result["dataset_supervision_contract_sha256"] for result in preflight_results.values()
    }
    if len(dataset_hashes) != 1:
        raise RuntimeError(
            "Preflight dataset membership changes across model/split seeds: "
            f"{sorted(dataset_hashes)}"
        )

    preflight_payload = {
        "status": "complete",
        "data_seed": DATA_SEED,
        "conditions": [preflight_results[(policy, seed)] for seed in SEEDS for policy in POLICIES],
    }
    (EXPERIMENT_DIR / "preflight.json").write_text(
        json.dumps(preflight_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if preflight_only:
        print(json.dumps(preflight_payload, indent=2, sort_keys=True))
        return

    initialize_experiment_dir(
        out_dir=str(EXPERIMENT_DIR),
        slug="groove-corrected-baseline",
        title="Groove-corrected source-mapping baseline",
        source_script=str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        agent_label="claude_codex",
        metadata=_metadata(),
    )

    conditions = [
        {
            "run_id": _run_id(policy, seed),
            "policy": policy,
            "seed": seed,
            "requested_gpu": GPU,
            "status": "launching",
            "required_files": list(REQUIRED_RUN_FILES),
        }
        for seed in SEEDS
        for policy in POLICIES
    ]
    manifest_path = EXPERIMENT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(conditions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    calls = [
        train_condition.spawn(
            row["policy"],
            row["seed"],
            preflight_results[(row["policy"], row["seed"])]["split_support_sha256"],
        )
        for row in conditions
    ]
    results = []
    for row, call in zip(conditions, calls):
        try:
            result = call.get()
        except Exception as exc:  # noqa: BLE001 - persist every remote failure in the manifest
            result = {
                **row,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        manifest_path.write_text(
            json.dumps(results + conditions[len(results) :], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    failures = [row for row in results if row.get("status") != "complete"]
    if failures:
        raise RuntimeError(f"{len(failures)} condition(s) failed: {failures}")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit("Launch with: modal run <this-file>")
