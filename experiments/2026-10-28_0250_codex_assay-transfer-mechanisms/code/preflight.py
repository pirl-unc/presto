"""Admit all distinct source sets and execute one TRAIN-only software update."""

from pathlib import Path
import json
import sys
import time

F = Path(__file__).resolve().parents[1]
ROOT = F.parents[1]
UPLOAD = ROOT / "artifacts" / F.name / "upload"
sys.path.insert(0, str(UPLOAD / "runtimes/shared"))
import torch
from presto.data.modular_sources import ObservationIndex
from presto.training.modular_run import preflight
from presto.training.modular import train_step, coverage_summary, save_checkpoint
from presto.models import Presto


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started, reports = time.monotonic(), {}
    for variant in (
        "wide-ic50",
        "wide-six-primary-matched",
        "wide-six-primary-shuffled",
    ):
        path = UPLOAD / "inputs" / f"{variant}-s42.json"
        index = ObservationIndex(path)
        config, training, scales, admission = preflight(index)
        reports[variant] = dict(
            admission=admission,
            training=training,
            scales={k: vars(v) for k, v in scales.items()},
            partitions={k: len(v) for k, v in index.partitions.items()},
        )
        print(variant, reports[variant]["partitions"], flush=True)
        if variant == "wide-six-primary-matched":
            selected = {}
            for i in index.partitions["train:assay"]:
                row = index[i]
                key = row.endpoint, row.mhc[0].mhc_class
                if len(selected.get(key, [])) < 2:
                    selected.setdefault(key, []).append(row)
            rows = [row for values in selected.values() for row in values]
            torch.manual_seed(42)
            model = Presto(config)
            step = train_step(
                model,
                torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4),
                rows,
                scales,
            )
            coverage = coverage_summary([step])
            assert not coverage["untrained_parameters"]
            probe = dict(
                split="train",
                updates=1,
                observation_ids=[r.observation_id for r in rows],
                loss=step["loss"],
                counts=step["counts"],
                coverage=coverage,
                scope="Software smoke only; no validation/test inference or predictive-quality claim",
            )
            destination = ROOT / "artifacts" / F.name / "software-preflight"
            destination.mkdir(parents=True, exist_ok=True)
            save_checkpoint(destination / "probe.pt", model, scales, metadata=probe)
            (F / "results/software-probe.json").write_text(
                json.dumps(probe, indent=2) + "\n"
            )
    for endpoint, scale in reports["wide-six-primary-matched"]["scales"].items():
        other = reports["wide-six-primary-shuffled"]["scales"][endpoint]
        assert all(abs(scale[k] - other[k]) < 1e-10 for k in ("mean", "scale"))
    result = dict(
        status="passed",
        distinct_source_sets=reports,
        seconds=time.monotonic() - started,
        capacity_and_all_sampler_configurations="validated by frozen initial hashes, sampler-prefix simulations and common trainer tests",
    )
    (F / "results/preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print("All distinct source sets and software smoke passed", flush=True)


if __name__ == "__main__":
    main()
