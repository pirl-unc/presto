"""Explicit post-fit production-boundary qualification; preserve original fit bytes."""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("route_core", EXP / "code/core.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    z = torch.linspace(-12, 12, 49, dtype=torch.float64)
    eta = torch.linspace(-8, 8, 49, dtype=torch.float64)
    identity = module.log_response(z, torch.full_like(z, torch.inf))
    exact = module.log_response(z, eta)
    sampled = module.log_response(z, eta, torch.Generator().manual_seed(42))
    increased = module.log_response(z, eta + 0.01)
    identity_error = float(identity.abs().max())
    sampled_forward_error = float((sampled - exact).abs().max())
    minimum_increment = float((increased - exact).min())
    assert identity_error < 1e-12 and sampled_forward_error < 1e-12
    assert minimum_increment >= 0
    baseline = np.exp(np.linspace(-10, 10, 49))
    response = exact.exp().numpy()
    ratio_error = float(
        np.max(abs(np.log10(baseline * response / baseline) - exact.numpy() / np.log(10)))
    )
    assert ratio_error < 1e-12
    result = {
        "status": "passed",
        "timing": "post-fit coverage correction",
        "identity_error": identity_error,
        "sampled_forward_error": sampled_forward_error,
        "minimum_monotonic_increment": minimum_increment,
        "baseline_amount_cancels_error": ratio_error,
        "model_or_fit_changed": False,
        "original_pre_fit_boundary_coverage_complete": False,
    }
    (EXP / "results/boundary-qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    shutil.copyfile(__file__, EXP / "reproduce/source/boundary_check.py")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
