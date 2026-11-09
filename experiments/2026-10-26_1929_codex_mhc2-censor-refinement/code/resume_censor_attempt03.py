"""Explicit canonical recovery of the preserved100-step censor snapshot."""

import torch

from presto.training.modular_resume import atomic_json
from presto.training.modular_run import fit_index

from launch import FAMILY, RAW, guarded, index_for, read


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    pin = read(FAMILY / "reproduce/interruption01.json")
    from presto.data.modular_sources import file_sha256

    assert file_sha256(RAW / "censor_refinement/resume.pt") == pin["snapshot_sha256"]
    assert file_sha256(pin["preserved_snapshot"]) == pin["snapshot_sha256"]
    saved = torch.load(pin["preserved_snapshot"], map_location="cpu", weights_only=True)
    assert saved["metadata"]["resume"]["state"]["steps"] == 100
    with guarded("resume-censor-refinement-attempt03"):
        result = fit_index(
            index_for("censor_refinement"), RAW / "censor_refinement", device="cpu", resume=True
        )
        last = torch.load(RAW / "censor_refinement/last.pt", map_location="cpu", weights_only=True)
        assert last["metadata"]["steps"] == 1476
        assert result["coverage_gate"]["status"] == "passed"
        atomic_json(FAMILY / "results/censor_refinement-fit.json", result)


if __name__ == "__main__":
    main()
