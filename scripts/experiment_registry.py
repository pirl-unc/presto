#!/usr/bin/env python
"""Shared helpers for experiment registry directories and launcher registration."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
import json


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_ROOT = ROOT / "experiments"
REGISTRY_MD = EXPERIMENTS_ROOT / "experiment_log.md"
REPRO_DIRNAME = "reproduce"


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
    return EXPERIMENTS_ROOT / f"{stamp}_{_sanitize_component(agent_label)}_{_sanitize_component(slug)}"


def _to_repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _append_markdown_stub(entry: Mapping[str, Any]) -> None:
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
    REGISTRY_MD.write_text(text.rstrip() + "\n" + block, encoding="utf-8")


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


def _capture_git_state() -> dict[str, str | bool | None]:
    """Capture current git commit, branch, and dirty status."""
    def _run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                args, cwd=str(ROOT), stderr=subprocess.DEVNULL, text=True,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty_rc = subprocess.call(
        ["git", "diff", "--stat", "--exit-code"],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ) if commit is not None else None
    dirty = dirty_rc != 0 if dirty_rc is not None else None
    return {"commit": commit, "branch": branch, "dirty": dirty}


def _capture_launch_environment() -> dict[str, str]:
    prefixes = ("PRESTO_", "MODAL_", "PYTORCH_", "CUDA_")
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.startswith(prefixes):
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
) -> Path:
    reproduce_dir = out_dir / REPRO_DIRNAME
    reproduce_dir.mkdir(parents=True, exist_ok=True)

    launch_argv = list(argv or sys.argv)
    launch_env = dict(environment or _capture_launch_environment())
    git_state = _capture_git_state()

    metadata = {
        "created_at": datetime.now().isoformat(),
        "cwd": str(ROOT),
        "python_executable": sys.executable,
        "argv": launch_argv,
        "source_script": source_script,
        "environment": launch_env,
        "git": git_state,
    }
    (reproduce_dir / "launch.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    export_lines = [f"export {key}={_shell_quote(value)}" for key, value in launch_env.items()]
    cmd = " ".join([_shell_quote(sys.executable), *(_shell_quote(arg) for arg in launch_argv)])
    git_comment_lines = [
        "# Git state at launch:",
        f"#   commit: {git_state.get('commit') or 'unknown'}",
        f"#   branch: {git_state.get('branch') or 'unknown'}",
        f"#   dirty: {'yes' if git_state.get('dirty') else 'no' if git_state.get('dirty') is not None else 'unknown'}",
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

    source_path = ROOT / source_script
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
        "These files freeze the launch invocation and environment for this experiment family so later agents can rerun or extend it without relying on current launcher defaults.",
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
) -> Path:
    target = Path(out_dir) if str(out_dir).strip() else default_experiment_dir(slug=slug, agent_label=agent_label)
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
