"""Regressions from the third review. Two were reproduced, not inferred.

The panel-identity bug is the one worth remembering: the counterfactual panel
subtracted the *baseline* condition's contribution instead of the *observed*
one, so the column for the observed condition double-counted it. The loss
gathers exactly that column, which meant the perturbed rows the whole feature
exists for -- ERAP1-KO, TAP-KO -- were supervised on a quantity the model never
reports. Every row at index 0 agreed, so a spot check would have passed.
"""

import pytest

torch = pytest.importorskip("torch")

from presto.models.presto import Presto  # noqa: E402


@pytest.fixture
def model_with_profiles():
    """Zero-initialized profiles make every column agree; give them values."""
    torch.manual_seed(0)
    model = Presto(d_model=32, n_layers=2, n_heads=4)
    model.eval()
    head = model.excision_head
    with torch.no_grad():
        for tensor in (
            head.invivo_profile_c,
            head.invivo_profile_n,
            head.stimulus_profile_c,
            head.invivo_bias,
        ):
            tensor.normal_(0.0, 0.5)
    return model


def _batch():
    import sys

    sys.path.insert(0, "tests")
    from test_gradient_coverage import _every_modality_batch

    return _every_modality_batch()


class TestPanelIdentity:
    """panel[:, observed] must equal the scalar the model reports."""

    @pytest.mark.parametrize(
        "panel_key,index_key,modulus",
        [
            ("excision_panel_apm", "apm_perturbation_idx", 7),
            ("excision_panel_stimulus", "processing_stimulus_idx", 6),
        ],
    )
    def test_observed_column_equals_the_scalar(
        self, model_with_profiles, panel_key, index_key, modulus
    ):
        batch = _batch()
        n = batch.pep_tok.shape[0]
        provenance = dict(batch.provenance)
        # Spread across conditions: the bug was invisible at index 0.
        provenance[index_key] = torch.arange(n) % modulus
        provenance["peptide_source_idx"] = torch.ones(n, dtype=torch.long)
        with torch.no_grad():
            out = model_with_profiles(
                pep_tok=batch.pep_tok,
                mhc_a_tok=batch.mhc_a_tok,
                mhc_b_tok=batch.mhc_b_tok,
                mhc_class=batch.mhc_class,
                provenance=provenance,
            )
        gathered = out[panel_key].gather(
            1, provenance[index_key].unsqueeze(1)
        ).squeeze(1)
        assert torch.allclose(out["excision_logit"], gathered, atol=1e-4), (
            "the observed column disagrees with excision_logit; the panel is "
            "double-counting the observed condition and the loss supervises a "
            "quantity the model never reports"
        )

    def test_other_columns_differ_from_the_observed_one(self, model_with_profiles):
        """Guards the trivial way to satisfy the identity above."""
        batch = _batch()
        n = batch.pep_tok.shape[0]
        provenance = dict(batch.provenance)
        provenance["apm_perturbation_idx"] = torch.zeros(n, dtype=torch.long)
        provenance["peptide_source_idx"] = torch.ones(n, dtype=torch.long)
        with torch.no_grad():
            out = model_with_profiles(
                pep_tok=batch.pep_tok,
                mhc_a_tok=batch.mhc_a_tok,
                mhc_b_tok=batch.mhc_b_tok,
                mhc_class=batch.mhc_class,
                provenance=provenance,
            )
        panel = out["excision_panel_apm"]
        assert not torch.allclose(panel[:, 0], panel[:, 3])


class TestUnallocatedModuleKeysAreRejected:
    """Gated-off modules are now a strict-load error, deliberately.

    Several module families are built only when their mode selects them --
    alternate positional encodings, the collapsed-topology projections,
    class-specific core scorers, direct-segment residuals. A checkpoint written
    under a different mode carries their weights, and a tolerance pass used to
    drop those keys so `strict=True` would still succeed.

    That tolerance is gone with the rest of the checkpoint-compat layer. It was
    doing something subtly wrong: quietly accepting a checkpoint from a
    *different architecture configuration* and reporting success. A checkpoint
    now loads into the configuration that wrote it, and a mismatch says so.
    """

    UNALLOCATED_KEYS = {
        "pep_abs_pos.weight": (50, 32),
        "core_window_score_class1.0.weight": (32, 256),
        "processing_condition_embed.weight": (42, 32),
    }

    @pytest.mark.parametrize("topology", ["collapsed", "expanded"])
    def test_strict_load_rejects_keys_this_config_does_not_allocate(self, topology):
        model = Presto(d_model=32, n_layers=1, n_heads=2, latent_topology=topology)
        state = dict(model.state_dict())
        for key, shape in self.UNALLOCATED_KEYS.items():
            state[key] = torch.zeros(*shape)
        fresh = Presto(d_model=32, n_layers=1, n_heads=2, latent_topology=topology)
        with pytest.raises(RuntimeError) as excinfo:
            fresh.load_state_dict(state, strict=True)
        assert "Unexpected key" in str(excinfo.value)

    @pytest.mark.parametrize("topology", ["collapsed", "expanded"])
    def test_a_matching_checkpoint_still_loads_strictly(self, topology):
        """The point is to reject *mismatches*, not to break round-tripping."""
        model = Presto(d_model=32, n_layers=1, n_heads=2, latent_topology=topology)
        fresh = Presto(d_model=32, n_layers=1, n_heads=2, latent_topology=topology)
        fresh.load_state_dict(dict(model.state_dict()), strict=True)

    def test_non_strict_load_still_tolerates_a_stale_key(self):
        """`strict=False` remains the escape hatch, and is now the only one."""
        model = Presto(d_model=32, n_layers=1, n_heads=2)
        state = dict(model.state_dict())
        state["processing_condition_embed.weight"] = torch.zeros(42, 32)
        Presto(d_model=32, n_layers=1, n_heads=2).load_state_dict(state, strict=False)


class TestMILProvenanceIsComplete:
    def test_tcell_bags_do_not_claim_mhc_origin(self):
        """Stamping a T-cell bag `mhc` asserts in-vivo proteasomal origin."""
        import inspect

        from presto.data import collate

        source = inspect.getsource(collate.PrestoCollator.__call__)
        tcell_call = source[source.index("tcell_mil_tensors = ") :][:400]
        assert 'peptide_source="unknown"' in tcell_call

    def test_elution_bags_carry_the_provenance_axes(self):
        from presto.data.collate import PrestoCollator
        from presto.data.loaders import ElutionRecord, PrestoDataset

        seq = "GSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPR"
        dataset = PrestoDataset(
            elution_records=[
                ElutionRecord(
                    peptide=f"SIINFEKL{c}",
                    alleles=["HLA-A*02:01", "HLA-B*07:02"],
                    detected=True,
                    cell_type="THP-1",
                )
                for c in "AC"
            ],
            mhc_sequences={"HLA-A*02:01": seq, "HLA-B*07:02": seq},
            strict_mhc_resolution=False,
        )
        batch = PrestoCollator()([dataset[i] for i in range(2)])
        for axis in ("cell_lineage_idx", "sample_origin_idx", "disease_state_idx"):
            assert axis in batch.mil_provenance, (
                f"{axis} is missing from MIL provenance; the elution loss runs "
                "through this path whenever MIL is active"
            )
