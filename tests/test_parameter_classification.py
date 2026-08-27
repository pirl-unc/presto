"""Tests for curriculum parameter classification.

An unmatched parameter name silently becomes "other", and every curriculum
stage freezes "other" -- so a name the table does not cover is a parameter that
never trains under the staged API. That failure is invisible: no error, no
warning, just weights that stay at their initialization.
"""

import pytest

from presto.models.presto import Presto


@pytest.mark.parametrize("topology", ["collapsed", "expanded"])
def test_every_parameter_is_classified(topology):
    """No parameter may fall through to 'other'.

    Before the rule table this missed 65 parameters in the expanded topology,
    including every positional encoding and the whole excision head.
    """
    model = Presto(d_model=32, n_layers=2, n_heads=4, latent_topology=topology)
    unmatched = sorted(
        name for name, _ in model.named_parameters()
        if model._classify_parameter(name) == "other"
    )
    assert unmatched == [], f"{topology}: unclassified parameters would be frozen"


def test_expanded_binding_latents_classify_as_binding():
    """The expanded topology renames pmhc_interaction to binding_affinity /
    binding_stability. Before those entries existed the substring tests missed
    them, so STAGE_BINDING_CLASS1 froze the latents it was meant to train."""
    model = Presto(d_model=32, n_layers=2, n_heads=4, latent_topology="expanded")
    binding = [
        name for name, _ in model.named_parameters()
        if model._classify_parameter(name) == "binding_query"
    ]
    assert any("binding_affinity" in name for name in binding)
    assert any("binding_stability" in name for name in binding)


def test_excision_head_trains_with_processing():
    """It is the processing readout; classifying it 'other' froze it always."""
    model = Presto(d_model=32, n_layers=2, n_heads=4)
    excision = [
        model._classify_parameter(name)
        for name, _ in model.named_parameters()
        if name.startswith("excision_head")
    ]
    assert excision and set(excision) == {"processing"}


def test_groove_positional_encodings_are_trunk_not_groove():
    """The one genuinely conditional rule: groove cross-attention is 'groove',
    but groove *positional* encodings belong to the trunk."""
    model = Presto(d_model=32, n_layers=2, n_heads=4)
    assert model._classify_parameter("groove_query") == "groove"
    assert model._classify_parameter("groove_pos.weight") == "trunk"
    assert model._classify_parameter("groove_1_abs_pos.weight") == "trunk"


def test_class_specific_processing_beats_the_general_rule():
    """Ordering is load-bearing: processing_class1 must not fall into
    'processing'. As a table that ordering is data rather than a comment."""
    model = Presto(d_model=32, n_layers=2, n_heads=4)
    assert model._classify_parameter("processing_class1_proj.weight") == "processing_class1"
    assert model._classify_parameter("processing_class2_proj.weight") == "processing_class2"
    assert model._classify_parameter("processing_pep_length_embed.weight") == "processing"
