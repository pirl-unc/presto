"""Run in old and new isolated import environments; artificial inputs only."""

import hashlib
import json

import torch

from presto.data.modular import MHCInput, Observation
from presto.data.modular_lineage import semantic_fingerprint
from presto.models.modular import ModularConfig, ModularPresto

torch.set_num_threads(1)
torch.manual_seed(79)
model = ModularPresto(
    ModularConfig(
        endpoints=("ic50", "elution_observation"),
        mhc_classes=("II",),
        dropout=0,
    )
).eval()
rows = [
    Observation(
        f"fixture-{endpoint}",
        "software-fixture",
        endpoint,
        "ACDEFGHIKL",
        (MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", "II"),),
        value=10 if endpoint == "ic50" else 1,
        unit="nM" if endpoint == "ic50" else "binary",
    )
    for endpoint in model.config.endpoints
]
state = model.state_dict()
report = {
    "weights": {
        name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
        for name, value in state.items()
    },
    "predictions": [float(prediction.value.detach()) for prediction in model(rows)],
    "semantic_fingerprints": [semantic_fingerprint(row) for row in rows],
    "feature_layout": model.feature_layout,
}
print(json.dumps(report, sort_keys=True))
