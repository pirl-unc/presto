"""Run artifact logger for training scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _json_default(value: Any) -> str:
    return str(value)


class RunLogger:
    """Writes lightweight training artifacts (`metrics.csv`, `metrics.jsonl`)."""

    def __init__(self, run_dir: Path, config: Optional[Dict[str, Any]] = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self._csv = self.csv_path.open("a", encoding="utf-8", newline="")
        if self.csv_path.stat().st_size == 0:
            self._csv.write("step,split,metric,value\n")
        self._jsonl = self.jsonl_path.open("a", encoding="utf-8")

        if config is not None:
            (self.run_dir / "config.json").write_text(
                json.dumps(config, indent=2, default=_json_default),
                encoding="utf-8",
            )

    def log(self, step: int, split: str, metrics: Dict[str, Any]) -> None:
        for key, value in metrics.items():
            self._csv.write(f"{step},{split},{key},{value}\n")
        payload = {"step": step, "split": split, **metrics}
        self._jsonl.write(json.dumps(payload, default=_json_default) + "\n")

    def close(self) -> None:
        self._csv.close()
        self._jsonl.close()
