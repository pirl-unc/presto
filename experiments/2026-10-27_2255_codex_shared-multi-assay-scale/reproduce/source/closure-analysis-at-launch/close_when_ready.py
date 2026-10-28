"""Terminal collection/verification only; never launch or restart a training fit."""

from pathlib import Path
import json
import os
import subprocess
import sys
import time

F = Path(__file__).resolve().parents[1]
ROOT = F.parents[1]


def main():
    while True:
        state = json.loads((F / "reproduce/controller.json").read_text())
        if state.get("failed"):
            raise RuntimeError(
                f"Training failed; explicit recovery needed: {state['failed']}"
            )
        if len(state.get("completed", [])) == 8 and not state.get("pending"):
            break
        time.sleep(30)
    steps = [
        ("seal-validation", [sys.executable, str(F / "analysis/select_validation.py")]),
        ("independent-verification", [sys.executable, str(F / "analysis/verify.py")]),
        ("comparison", [sys.executable, str(F / "analysis/compare.py")]),
        (
            "gpu-training-replay",
            [
                str(Path(sys.executable).with_name("modal")),
                "run",
                str(F / "analysis/replay_modal.py"),
            ],
        ),
    ]
    environment = dict(
        os.environ,
        PRESTO_MODAL_GPU="H100!",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        MPLCONFIGDIR="/private/tmp/presto-scale-mpl",
    )
    for name, command in steps:
        print(name, flush=True)
        with (F / "launch_logs" / f"{name}.log").open("x") as log:
            subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    print(
        "Metric checks and serving replay complete; narrative closure remains required.",
        flush=True,
    )


if __name__ == "__main__":
    main()
