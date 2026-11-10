"""Artificial-fixture capacity probe, not biological model performance."""

import json
from dataclasses import replace
from pathlib import Path

import torch
from presto.data.modular import Context, ContextAssertion, MHCInput, Observation
from presto.models.modular import ModularConfig, ModularPresto


def main():
    torch.set_num_threads(1)
    results = []
    for mode in ("legacy_joint", "affinity_downstream"):
        for seed in (42, 43):
            torch.manual_seed(seed)
            cfg = ModularConfig(
                endpoints=("ic50", "elution_observation"),
                interventions=("transduction:HLA-DM",),
                dropout=0,
                d_model=32,
                n_heads=4,
                n_layers=1,
                d_ff=64,
                elution_staging=mode,
                elution_affinity_endpoint="ic50"
                if mode == "affinity_downstream"
                else None,
            )
            model = ModularPresto(cfg).eval()
            for mhc_class in ("I", "II"):
                peptides = (
                    ("ACDEFGHIK", "FGHIKLMNP", "LMNPQRSTV")
                    if mhc_class == "I"
                    else ("ACDEFGHIKLMNPQR", "FGHIKLMNPQRSTVW", "LMNPQRSTVWYACDE")
                )
                for peptide in peptides:
                    rows = []
                    for a, b in [
                        ("ACDEFGHIKLM", "NPQRSTVWYAC"),
                        ("VWYACDEFGHI", "KLMNPQRSTVW"),
                    ]:
                        for state in ("absent", "present"):
                            rows.append(
                                Observation(
                                    observation_id=str(len(rows)),
                                    source="artificial_capacity_fixture",
                                    endpoint="elution_observation",
                                    peptide=peptide,
                                    mhc=(MHCInput(a, b, mhc_class),),
                                    value=1,
                                    unit="binary",
                                    group_id="artificial",
                                    context=Context(
                                        interventions={
                                            "transduction:HLA-DM": ContextAssertion(
                                                state, {"source": "artificial_fixture"}
                                            )
                                        }
                                    ),
                                )
                            )
                    with torch.no_grad():
                        outputs = model(rows)
                        affinities = model(
                            [
                                replace(r, endpoint="ic50", value=100, unit="nM")
                                for r in rows
                            ]
                        )
                    deltas = [
                        float(outputs[i + 1].value - outputs[i].value) for i in (0, 2)
                    ]
                    affinity_deltas = [
                        float(affinities[i + 1].value - affinities[i].value)
                        for i in (0, 2)
                    ]
                    assert affinity_deltas == [0.0, 0.0]
                    regs = [
                        o.diagnostics["candidates"][0]["registers"][0] for o in outputs
                    ]
                    weight_deltas = [
                        float((regs[i + 1]["weights"] - regs[i]["weights"]).abs().max())
                        for i in (0, 2)
                    ]
                    processing = []
                    if mode == "affinity_downstream":
                        processing = [
                            float(
                                regs[i + 1]["elution_terms"]["processing"]
                                - regs[i]["elution_terms"]["processing"]
                            )
                            for i in (0, 2)
                        ]
                        assert processing[0] == processing[1]
                        assert weight_deltas == [0.0, 0.0]
                    results.append(
                        {
                            "mode": mode,
                            "seed": seed,
                            "class": mhc_class,
                            "peptide": peptide,
                            "dm_logit_deltas_by_artificial_mhc": deltas,
                            "mhc_by_dm_logit_interaction": deltas[1] - deltas[0],
                            "dm_processing_deltas": processing,
                            "max_register_weight_changes": weight_deltas,
                            "purified_affinity_deltas": affinity_deltas,
                        }
                    )
    summary = {
        "status": "passed",
        "biological_performance": False,
        "optimizer_steps": 0,
        "n_conditions": len(results),
        "results": results,
        "max_abs_mhc_dm_logit_interaction_by_mode": {
            m: max(
                abs(r["mhc_by_dm_logit_interaction"]) for r in results if r["mode"] == m
            )
            for m in ("legacy_joint", "affinity_downstream")
        },
    }
    path = (
        Path(__file__).resolve().parents[1] / "results/condition-interaction-probe.json"
    )
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))


if __name__ == "__main__":
    main()
