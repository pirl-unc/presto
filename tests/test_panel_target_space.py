"""The assay panel must be supervised in the space it predicts.

`AffinityPredictor.predict_assay_panel` returns, per its own docstring, a
"predicted KD offset" -- normalized log10 space, the same space the main
binding spec targets via `normalize_binding_target_log10(..., assume_log10=
False)`.

The panel loss regressed it against `bind_target` **raw**, which is nM up to
`DEFAULT_MAX_AFFINITY_NM` (50,000). smooth_l1 between a log-space head and a
raw-nM target is roughly the target's magnitude, so
`loss_binding_assay_panel` sat near 142,900 -- an order of magnitude above the
total loss -- and could not fall, because no head output reaches 50,000.

It dominated the gradient. Measured on the synthetic run: validation loss moved
9924.72 -> 9923.51 across ten epochs, 0.012%, and the predictions were
constant to three decimals. With the target normalized, the same run reaches
1.74 -> 1.11 in eight epochs and the panel term falls 2.82 -> 1.56.

A scale bug like this does not fail a test or raise; the loss is finite and
training "runs". Only the magnitude gives it away, so magnitude is what this
pins.
"""

import pytest

torch = pytest.importorskip("torch")

from presto.models.affinity import DEFAULT_MAX_AFFINITY_NM  # noqa: E402
from presto.scripts.train_synthetic import (  # noqa: E402
    normalize_binding_target_log10,
)


class TestTargetNormalization:
    def test_raw_nanomolar_maps_into_log_space(self):
        raw = torch.tensor([1.0, 50.0, 500.0, 5000.0, DEFAULT_MAX_AFFINITY_NM])
        out = normalize_binding_target_log10(
            raw, max_affinity_nM=DEFAULT_MAX_AFFINITY_NM, assume_log10=False
        )
        assert float(out.min()) >= 0.0
        assert float(out.max()) < 5.0, "targets must be O(1); a head cannot regress raw nanomolar"

    def test_the_weakest_affinity_is_not_five_orders_of_magnitude(self):
        """The specific number that made the loss ~142,900."""
        weakest = normalize_binding_target_log10(
            torch.tensor([float(DEFAULT_MAX_AFFINITY_NM)]),
            max_affinity_nM=DEFAULT_MAX_AFFINITY_NM,
            assume_log10=False,
        )
        assert float(weakest) < 5.0
        assert DEFAULT_MAX_AFFINITY_NM >= 10_000, "premise of this test changed"


class TestPanelLossIsNormalized:
    """Source-level, because the transform is applied inside `compute_loss`.

    Enumerated rather than anchored on the first match, per presto#18: every
    place the panel loss reads `bind_target` must normalize it.
    """

    def test_panel_loss_normalizes_its_target(self):
        import inspect

        from presto.scripts import train_synthetic

        from source_probe import region_between

        source = inspect.getsource(train_synthetic.compute_loss)
        block = region_between(
            source,
            "panel_context = getattr(batch",
            'supervised_loss_support["binding_assay_panel"]',
            where="compute_loss",
        )
        assert "binding_assay_panel" in block
        assert "normalize_binding_target_log10" in block, (
            "the assay panel is being supervised against a raw-nM target while "
            "predicting a normalized log10 KD offset; the loss term will sit "
            "around 142,900 and swamp every other gradient"
        )

    def test_no_panel_loss_reads_bind_target_unnormalized(self):
        """Guards the shape, not one call site."""
        import inspect
        import re

        from presto.scripts import train_synthetic

        source = inspect.getsource(train_synthetic.compute_loss)
        reads = [m.start() for m in re.finditer(r"bind_target\.reshape", source)]
        assert reads, "bind_target is no longer reshaped here; update this test"
        for offset in reads:
            window = source[max(0, offset - 200) : offset + 200]
            assert "normalize_binding_target_log10" in window, (
                f"a raw bind_target read at offset {offset} is not normalized"
            )
