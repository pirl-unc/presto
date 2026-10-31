"""Artificial fixture comparison for unchanged default model behavior."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import torch

import presto
from presto.data.modular import Context, ContextAssertion, MHCInput, Observation
from presto.data.modular_lineage import semantic_fingerprint
from presto.models.modular import ModularConfig, ModularPresto


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1729)
    expected_root, output = map(Path, sys.argv[1:])
    assert Path(presto.__file__).resolve().parent == expected_root.resolve()
    config = ModularConfig(
        endpoints=("ic50", "elution_observation", "tcr_binding", "bulk_observation", "ms_response"),
        processing_pathways=("fixture_route",),
        interventions=("fixture_intervention",),
        dropout=0,
    )
    model = ModularPresto(config).eval()
    rows = [
        Observation(
            observation_id=f"fixture-{endpoint}",
            source="artificial-parity-fixture",
            endpoint=endpoint,
            peptide="ACDEFGHIKLMNP",
            mhc=(MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", "II"),),
            tcr_a="ACDEFGHIKLM",
            tcr_b="NPQRSTVWYAC",
            value=100 if endpoint == "ic50" else 1,
            unit="nM" if endpoint == "ic50" else "binary",
            context=Context(
                processing_pathways={"fixture_route": ContextAssertion("present")},
                interventions={"fixture_intervention": ContextAssertion("absent")},
            ),
        )
        for endpoint in config.endpoints
    ]
    with torch.no_grad():
        values = model(rows)
    config_values = asdict(config)
    assert not config_values.pop("ordinal_scales", {})
    report = {
        "config": config_values,
        "state": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest(),
            }
            for name, value in model.state_dict().items()
        },
        "feature_layout": model.feature_layout,
        "fingerprints": [semantic_fingerprint(row) for row in rows],
        "values": [float(value.value) for value in values],
        "components": [
            {key: float(value) for key, value in prediction.components.items()}
            for prediction in values
        ],
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {"fixture_only": True, "state_tensors": len(report["state"]), "output": str(output)}
        )
    )


if __name__ == "__main__":
    main()
