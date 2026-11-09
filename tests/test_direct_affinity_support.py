"""Support boundaries for the direct shared measurement decoder."""

from dataclasses import replace

import torch

from presto.data.assay_measurement import AssayMethod, AssayTrack
from presto.data.modular import MHCInput, Observation, TargetScale
from presto.models.presto import Presto, PrestoConfig
from presto.training.modular import train_step


def test_unobserved_combination_is_indirect_only_after_all_factors_learn():
    radio = AssayTrack('ic50', 'purified', 'competitive', 'radioactivity')
    direct = AssayTrack('kd', 'purified', 'direct', 'fluorescence')
    unseen = replace(direct, detector='radioactivity')
    model = Presto(
        PrestoConfig(
            endpoints=('ic50', 'kd'),
            assay_tracks=(radio, direct, unseen),
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
            dropout=0,
        )
    )

    def row(track):
        return Observation(
            track.key,
            'artificial-software',
            track.endpoint,
            'ACDEFGHIK',
            (MHCInput('ACDEFGHIKL', 'MNPQRSTVWY', 'I'),),
            value=1000,
            unit='nM',
            assay_method=AssayMethod(
                track.preparation, track.mechanism, track.detector, 'verified'
            ),
        )

    assert model.assay_capabilities()[unseen.key]['status'] == 'random/untrained'
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scales = {name: TargetScale(2, 1, 2) for name in ('ic50', 'kd')}
    train_step(model, optimizer, [row(radio), row(direct)], scales)
    support = model.assay_capabilities()
    assert support[radio.key]['status'] == 'directly_supervised'
    assert support[direct.key]['status'] == 'directly_supervised'
    assert support[unseen.key]['status'] == 'indirectly_trained'
    assert support[unseen.key]['observations'] == 0
    assert not support[unseen.key]['combination_exposed']
    assert support[unseen.key]['validation'] == 'not_evaluated'
    assert torch.isfinite(model.predict([row(unseen)])[0].value)
