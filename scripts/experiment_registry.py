#!/usr/bin/env python
"""Shared helpers for experiment registry directories and launcher registration."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence


SOURCE_ROOT = Path(__file__).resolve().parent.parent
# Results belong to the caller's workspace, including with an installed wheel.
# Never write experiments into site-packages or infer the destination from the
# editable installation's location.
ROOT = Path(os.environ.get("PRESTO_WORKSPACE", Path.cwd())).resolve()
EXPERIMENTS_ROOT = ROOT / "experiments"
REGISTRY_MD = EXPERIMENTS_ROOT / "experiment_log.md"
REPRO_DIRNAME = "reproduce"

REPRODUCIBLE_ENVIRONMENT_KEYS = frozenset(
    {
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_LAUNCH_BLOCKING",
        "CUDA_VISIBLE_DEVICES",
        "HITLIST_DATA_DIR",
        "MKL_NUM_THREADS",
        "MODAL_ENVIRONMENT",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PRESTO_AGENT",
        "PRESTO_ALLOW_ZERO_MHC",
        "PRESTO_BATCH_SIZE",
        "PRESTO_BULK_CELL_LINE",
        "PRESTO_BULK_MS",
        "PRESTO_CUDA_DEVICE",
        "PRESTO_D_MODEL",
        "PRESTO_EPOCHS",
        "PRESTO_EXCISION_NEGATIVE_RATIO",
        "PRESTO_EXPERIMENT_AGENT",
        "PRESTO_EXPERIMENT_DIR",
        "PRESTO_EXTRA_ARGS",
        "PRESTO_FACTORIAL",
        "PRESTO_HITLIST_ALLELE",
        "PRESTO_HITLIST_BUILD",
        "PRESTO_HITLIST_BUNDLE",
        "PRESTO_HITLIST_BUNDLE_SHA256",
        "PRESTO_HITLIST_MHC_CLASS",
        "PRESTO_LATENT_TOPOLOGY",
        "PRESTO_LR",
        "PRESTO_MAX_BINDING",
        "PRESTO_MAX_BULK_MS",
        "PRESTO_MAX_ELUTION",
        "PRESTO_MAX_KINETICS",
        "PRESTO_MAX_STABILITY",
        "PRESTO_MERGED_TSV",
        "PRESTO_MHCSEQS_SEARCH_DIR",
        "PRESTO_MODAL_GPU",
        "PRESTO_MODAL_REPO_REF",
        "PRESTO_MODAL_REPO_URL",
        "PRESTO_MODAL_SOURCE",
        "PRESTO_MODAL_TIMEOUT_SECONDS",
        "PRESTO_MODAL_VOLUME",
        "PRESTO_MODAL_WHEEL",
        "PRESTO_N_HEADS",
        "PRESTO_N_LAYERS",
        "PRESTO_QUALIFICATION_WHEEL",
        "PRESTO_REPO_ROOT",
        "PRESTO_SEED",
        "PRESTO_WORKSPACE",
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_ENABLE_MPS_FALLBACK",
    }
)
_CREDENTIAL_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "API_KEY",
    "ACCESS_KEY",
    "CREDENTIAL",
    "PRIVATE_KEY",
)
_REGISTRY_THREAD_LOCK = threading.Lock()


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(raw_temp)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def _portable_file_lock(lock_path: Path, *, blocking: bool = True):
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        fcntl = None

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        if fcntl is not None:
            flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            fcntl.flock(stream.fileno(), flags)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        else:  # pragma: no cover - Windows
            import msvcrt

            while True:
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if not blocking:
                        raise
                    time.sleep(0.01)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def registry_lock(registry_path: Path | None = None):
    """Serialize read-modify-write updates to the canonical markdown registry."""
    target = Path(registry_path) if registry_path is not None else REGISTRY_MD
    lock_path = target.parent / f".{target.name}.lock"
    with _REGISTRY_THREAD_LOCK, _portable_file_lock(lock_path, blocking=True):
        yield


def _sanitize_component(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    text = text.strip("-")
    return text or "unknown"


def default_agent_label() -> str:
    return _sanitize_component(os.environ.get("PRESTO_EXPERIMENT_AGENT", "codex"))


def default_experiment_dir(
    *,
    slug: str,
    agent_label: str,
    timestamp: datetime | None = None,
) -> Path:
    stamp = (timestamp or datetime.now()).strftime("%Y-%m-%d_%H%M")
    return (
        EXPERIMENTS_ROOT / f"{stamp}_{_sanitize_component(agent_label)}_{_sanitize_component(slug)}"
    )


def _to_repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _append_markdown_stub(entry: Mapping[str, Any]) -> None:
    with registry_lock():
        REGISTRY_MD.parent.mkdir(parents=True, exist_ok=True)
        if REGISTRY_MD.exists():
            text = REGISTRY_MD.read_text(encoding="utf-8")
        else:
            text = "# Presto Experiment Log\n\n"
        marker = f"### {entry['id']}"
        if marker in text:
            return
        dataset = json.dumps(entry.get("dataset_contract", {}), sort_keys=True)
        training = json.dumps(entry.get("training", {}), sort_keys=True)
        tested = json.dumps(entry.get("tested", []), sort_keys=True)
        block = "\n".join(
            [
                "",
                marker,
                f"- **Agent**: {entry.get('agent', '')}",
                f"- **Dir**: [{entry['id']}]({entry.get('experiment_dir', '')})",
                f"- **Source script**: `{entry.get('source_script', '')}`",
                f"- **Status**: {entry.get('status', 'launched')}",
                f"- **Dataset**: `{dataset}`",
                f"- **Training**: `{training}`",
                f"- **Tested**: `{tested}`",
                "",
            ]
        )
        _atomic_write_bytes(REGISTRY_MD, (text.rstrip() + "\n" + block).encode("utf-8"))


def _write_stub_readme(
    *,
    out_dir: Path,
    title: str,
    agent_label: str,
    source_script: str,
    metadata: Mapping[str, Any],
) -> None:
    path = out_dir / "README.md"
    if path.exists():
        return
    lines = [
        f"# {title}",
        "",
        f"- Agent: `{agent_label}`",
        f"- Source script: `{source_script}`",
        f"- Created: `{datetime.now().isoformat()}`",
        "",
        "## Dataset Contract",
        "",
        "```json",
        json.dumps(metadata.get("dataset_contract", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Training",
        "",
        "```json",
        json.dumps(metadata.get("training", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Tested Conditions",
        "",
        "```json",
        json.dumps(metadata.get("tested", []), indent=2, sort_keys=True),
        "```",
        "",
        "## Notes",
        "",
        "- Launcher-created stub. Add results, plots, and takeaways here after collection.",
        f"- Reproducibility bundle: [`{REPRO_DIRNAME}/`](./{REPRO_DIRNAME}/)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def capture_git_state() -> dict[str, str | bool | None]:
    """Capture current git commit, branch, and dirty status."""

    def _run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                args,
                cwd=str(SOURCE_ROOT),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain", "--untracked-files=normal"])
    dirty = bool(status) if status is not None else None
    return {"commit": commit, "branch": branch, "dirty": dirty}


_capture_git_state = capture_git_state


def _is_credential_key(key: str) -> bool:
    upper = key.upper()
    return any(marker in upper for marker in _CREDENTIAL_MARKERS)


def _capture_launch_environment() -> dict[str, str]:
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in REPRODUCIBLE_ENVIRONMENT_KEYS and not _is_credential_key(key):
            env[key] = value
    return dict(sorted(env.items()))


def _shell_quote(text: str) -> str:
    return "'" + str(text).replace("'", "'\"'\"'") + "'"


def write_reproducibility_bundle(
    *,
    out_dir: Path,
    source_script: str,
    argv: Sequence[str] | None = None,
    environment: Mapping[str, str] | None = None,
    git_state: Mapping[str, str | bool | None] | None = None,
) -> Path:
    resolved_git_state = dict(git_state) if git_state is not None else capture_git_state()
    reproduce_dir = out_dir / REPRO_DIRNAME
    reproduce_dir.mkdir(parents=True, exist_ok=True)

    launch_argv = list(argv or sys.argv)
    if environment is None:
        launch_env = _capture_launch_environment()
    else:
        launch_env = {
            key: str(value)
            for key, value in sorted(environment.items())
            if not _is_credential_key(key)
        }

    metadata = {
        "created_at": datetime.now().isoformat(),
        "cwd": str(ROOT),
        "python_executable": sys.executable,
        "argv": launch_argv,
        "source_script": source_script,
        "environment": launch_env,
        "git": resolved_git_state,
    }
    (reproduce_dir / "launch.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    export_lines = [f"export {key}={_shell_quote(value)}" for key, value in launch_env.items()]
    cmd = " ".join([_shell_quote(sys.executable), *(_shell_quote(arg) for arg in launch_argv)])
    dirty = resolved_git_state.get("dirty")
    dirty_label = "unknown" if dirty is None else ("yes" if dirty else "no")
    git_comment_lines = [
        "# Git state at launch:",
        f"#   commit: {resolved_git_state.get('commit') or 'unknown'}",
        f"#   branch: {resolved_git_state.get('branch') or 'unknown'}",
        f"#   dirty: {dirty_label}",
    ]
    launch_script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            *git_comment_lines,
            f"cd {_shell_quote(str(ROOT))}",
            *export_lines,
            cmd,
            "",
        ]
    )
    launch_path = reproduce_dir / "launch.sh"
    launch_path.write_text(launch_script, encoding="utf-8")
    launch_path.chmod(0o755)

    source_path = SOURCE_ROOT / source_script
    if source_path.exists():
        snapshot_dir = reproduce_dir / "source"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, snapshot_dir / source_path.name)

    readme_lines = [
        "# Reproducibility Bundle",
        "",
        f"- Source script: `{source_script}`",
        "- Launch command: `launch.sh`",
        "- Launch metadata: `launch.json`",
        "- Launcher snapshot: `source/` (if the source script existed at bundle creation time)",
        "",
        "These files freeze the launch invocation and environment for this experiment family so "
        "later agents can rerun or extend it without relying on current launcher defaults.",
        "",
    ]
    (reproduce_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    return reproduce_dir


def initialize_experiment_dir(
    *,
    out_dir: str,
    slug: str,
    title: str,
    source_script: str,
    agent_label: str,
    metadata: Mapping[str, Any],
    git_state: Mapping[str, str | bool | None] | None = None,
) -> Path:
    resolved_git_state = dict(git_state) if git_state is not None else capture_git_state()
    target = (
        Path(out_dir)
        if str(out_dir).strip()
        else default_experiment_dir(slug=slug, agent_label=agent_label)
    )
    target.mkdir(parents=True, exist_ok=True)
    _write_stub_readme(
        out_dir=target,
        title=title,
        agent_label=agent_label,
        source_script=source_script,
        metadata=metadata,
    )
    write_reproducibility_bundle(
        out_dir=target,
        source_script=source_script,
        git_state=resolved_git_state,
    )
    entry = {
        "id": target.name,
        "agent": agent_label,
        "experiment_dir": _to_repo_relative(target),
        "source_script": source_script,
        "status": "launched",
        "created_at": datetime.now().isoformat(),
        **dict(metadata),
    }
    _append_markdown_stub(entry)
    return target
