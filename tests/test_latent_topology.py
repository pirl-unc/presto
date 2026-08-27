"""Tests for the expanded latent DAG topology (design.md S7.1/S7.2/S7.5).

The point of the expanded topology is that each specified latent gets its own
query with a declared token-access scope. These tests assert the scope is
actually enforced at runtime, not merely declared in a table — a latent that
silently sees tokens it should not is exactly the shortcut the DAG exists to
prevent.
"""

import pytest
import torch

from presto.models.presto import Presto

BATCH = 2
PEP_LEN = 11
MHC_LEN = 40
FLANK_LEN = 10


def _model(topology):
    model = Presto(d_model=32, n_layers=2, n_heads=4, latent_topology=topology)
    model.eval()
    return model


def _inputs(seed=0):
    generator = torch.Generator().manual_seed(seed)

    def tokens(length):
        return torch.randint(1, 20, (BATCH, length), generator=generator)

    return {
        "pep_tok": tokens(PEP_LEN),
        "mhc_a_tok": tokens(MHC_LEN),
        "mhc_b_tok": tokens(MHC_LEN),
        "flank_n_tok": tokens(FLANK_LEN),
        "flank_c_tok": tokens(FLANK_LEN),
        "mhc_class": "I",
    }


def _latents(model, inputs):
    with torch.no_grad():
        return model(**inputs)["latent_vecs"]


ALL_TWELVE = [
    "processing_class1",
    "processing_class2",
    "binding_affinity",
    "species_of_origin",
    "binding_stability",
    "presentation_class1",
    "presentation_class2",
    "recognition_cd8",
    "recognition_cd4",
    "immunogenicity_cd8",
    "immunogenicity_cd4",
    "ms_detectability",
]


class TestTopologyConstruction:
    def test_rejects_unknown_topology(self):
        with pytest.raises(ValueError, match="Unsupported latent_topology"):
            Presto(d_model=32, n_layers=2, n_heads=4, latent_topology="nonsense")

    def test_default_is_collapsed(self):
        assert Presto(d_model=32, n_layers=2, n_heads=4).latent_topology == "collapsed"

    def test_expanded_has_one_query_per_cross_attention_latent(self):
        model = _model("expanded")
        assert len(model.CROSS_ATTN_LATENTS) == 10
        assert set(model.latent_queries.keys()) == set(model.CROSS_ATTN_LATENTS)
        # Immunogenicity is an MLP over its dependencies, not a queried latent.
        assert "immunogenicity_cd8" not in model.CROSS_ATTN_LATENTS

    @pytest.mark.parametrize("topology", ["collapsed", "expanded"])
    def test_both_topologies_expose_all_twelve_latents(self, topology):
        latents = _latents(_model(topology), _inputs())
        for name in ALL_TWELVE:
            assert name in latents, f"{topology} is missing latent {name}"

    @pytest.mark.parametrize("topology", ["collapsed", "expanded"])
    def test_backward_pass_runs(self, topology):
        model = Presto(d_model=32, n_layers=2, n_heads=4, latent_topology=topology)
        outputs = model(**_inputs())
        (outputs["elution_logit"].sum() + outputs["processing_logit"].sum()).backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, "no parameter received gradient"


class TestExpandedSegmentAccess:
    """Each row of the S7.5 segment-access table, asserted behaviorally."""

    def _swap(self, key, seed=99):
        base = _inputs(seed=0)
        other = _inputs(seed=seed)
        changed = dict(base)
        changed[key] = other[key]
        return base, changed

    def _latent_changed(self, name, key):
        model = _model("expanded")
        base, changed = self._swap(key)
        before = _latents(model, base)[name]
        after = _latents(model, changed)[name]
        return not torch.allclose(before, after, atol=1e-6)

    def test_binding_does_not_see_flanks(self):
        assert not self._latent_changed("binding_affinity", "flank_n_tok")
        assert not self._latent_changed("binding_stability", "flank_c_tok")

    def test_processing_does_see_flanks(self):
        assert self._latent_changed("processing_class1", "flank_n_tok")
        assert self._latent_changed("processing_class2", "flank_c_tok")

    def _relative_change(self, name, key):
        model = _model("expanded")
        base, changed = self._swap(key)
        before = _latents(model, base)[name]
        after = _latents(model, changed)[name]
        return ((after - before).norm() / (before.norm() + 1e-9)).item()

    def test_processing_reaches_mhc_only_through_context_vec(self):
        """Processing has no MHC token access; S7.5 gives it ``context_vec``
        as "the sole channel of MHC-class information".

        So an MHC swap must still move it a little — the inferred class/species
        probabilities change — but by orders of magnitude less than it moves a
        latent that reads the MHC tokens directly. Equality would mean the
        context channel was dead; parity with binding would mean the token
        mask had leaked.
        """
        processing = self._relative_change("processing_class1", "mhc_a_tok")
        binding = self._relative_change("binding_affinity", "mhc_a_tok")
        assert processing > 0.0, "context_vec channel appears dead"
        assert binding > 100 * processing, (
            f"processing moved {processing:.2e} vs binding {binding:.2e}; "
            "processing looks like it can see MHC tokens directly"
        )

    def test_binding_does_see_mhc_tokens(self):
        assert self._latent_changed("binding_affinity", "mhc_a_tok")

    def test_peptide_only_latents_are_exactly_invariant(self):
        """These get no context_vec either, so the isolation should be exact."""
        for name in ("ms_detectability", "species_of_origin", "recognition_cd8"):
            assert self._relative_change(name, "mhc_a_tok") == 0.0, name
            assert self._relative_change(name, "flank_n_tok") == 0.0, name

    def test_peptide_only_latents_track_the_peptide(self):
        for name in ("ms_detectability", "species_of_origin"):
            assert self._latent_changed(name, "pep_tok"), name

    def test_recognition_does_not_see_mhc(self):
        assert not self._latent_changed("recognition_cd8", "mhc_a_tok")
        assert not self._latent_changed("recognition_cd4", "mhc_a_tok")

    def test_presentation_declares_no_token_access(self):
        model = _model("expanded")
        assert model.LATENT_SEGMENTS["presentation_class1"] == []
        assert model.LATENT_SEGMENTS["presentation_class2"] == []

    def test_presentation_always_has_dependencies(self):
        """The no-token-access claim depends on this.

        ``_ensure_nonempty_kv_mask`` makes key 0 valid when a latent would
        otherwise attend to nothing, and key 0 is the first *token* of the
        stream. For a latent with no token access that fallback would be a leak.
        It never fires for presentation only because the dependency block is
        always present and always valid — so if a dependency list is ever
        emptied, this fallback silently starts feeding presentation a flank
        token.
        """
        model = _model("expanded")
        for name in ("presentation_class1", "presentation_class2"):
            assert model.LATENT_SEGMENTS[name] == []
            assert model.LATENT_DEPS[name], (
                f"{name} has neither token access nor dependencies; the "
                "empty-KV fallback would expose it to token 0"
            )

    def test_presentation_still_responds_through_its_dependencies(self):
        """No token access, but it must not be constant.

        Presentation depends on processing and binding, so changing MHC tokens
        has to reach it indirectly. A presentation latent that did not move
        would mean the bottleneck had collapsed.
        """
        assert self._latent_changed("presentation_class1", "mhc_a_tok")
        assert self._latent_changed("presentation_class1", "flank_n_tok")
