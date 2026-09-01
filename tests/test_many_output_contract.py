"""The model is many-output: conditions are predicted, never supplied.

The DNA-model analogy is exact. Enformer-style trunks read sequence alone and
emit thousands of tracks, one per (cell type, assay); the track identity is an
output index, never an input feature. Presto reads peptide and MHC and emits
one output per assay configuration and per condition.

`docs/assay_modeling_contract.md` states the rule and names the forbidden
inputs. These tests assert it as an observable property -- passing a forbidden
input must not move the prediction -- rather than by reading the source, since
the source has been wrong twice.
"""

import pytest

torch = pytest.importorskip("torch")

from presto.models.presto import Presto  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = Presto(d_model=32, n_layers=2, n_heads=4)
    m.eval()
    return m


@pytest.fixture(scope="module")
def sequence_inputs():
    torch.manual_seed(1)
    return dict(
        pep_tok=torch.randint(4, 24, (3, 10)),
        mhc_a_tok=torch.randint(4, 24, (3, 40)),
        mhc_b_tok=torch.randint(4, 24, (3, 40)),
        mhc_class="I",
    )


#: Every key the contract names as a forbidden per-example input, grouped by
#: the argument that used to carry it.
FORBIDDEN_TCELL_CONTEXT = {
    "assay_method_idx": torch.tensor([1, 2, 1]),
    "assay_readout_idx": torch.tensor([1, 2, 1]),
    "apc_type_idx": torch.tensor([1, 2, 1]),
    "culture_context_idx": torch.tensor([1, 2, 1]),
    "stim_context_idx": torch.tensor([1, 2, 1]),
    "peptide_format_idx": torch.tensor([1, 2, 1]),
}
FORBIDDEN_BINDING_CONTEXT = {
    "assay_type_idx": torch.tensor([1, 2, 3]),
    "assay_prep_idx": torch.tensor([1, 2, 3]),
    "assay_geometry_idx": torch.tensor([1, 2, 3]),
    "assay_readout_idx": torch.tensor([1, 2, 3]),
}


class TestBindingIsAssayInvariant:
    def test_binding_prediction_ignores_the_assay_label(self, model, sequence_inputs):
        with torch.no_grad():
            base = model(**sequence_inputs)
            with_ctx = model(**sequence_inputs, binding_context=FORBIDDEN_BINDING_CONTEXT)
        for key in ("binding_logit", "binding_affinity_score", "binding_mixed_kd_log10"):
            assert torch.allclose(base[key], with_ctx[key]), (
                f"{key} moved when an assay label was attached; the prediction "
                "is conditioned on the assay rather than predicting it"
            )

    def test_the_input_side_context_vector_is_gone(self, model, sequence_inputs):
        """Not merely zeroed -- absent, so nothing can refill it."""
        with torch.no_grad():
            out = model(**sequence_inputs)
        assert "binding_factorized_assay_context_vec" not in out


class TestTCellIsContextInvariant:
    def test_head_is_structurally_context_invariant(self, model, sequence_inputs):
        """Gap 7, closed in the head and not only in the callers.

        This test previously asserted the opposite and said so: it recorded
        that `TCellAssayHead` still accepted the seven forbidden keys and still
        moved when given them, and instructed whoever removed them to convert
        it to an equality assertion. The arguments are gone, so it is one.

        Supplying `tcell_context` is now inert -- the model cannot be
        conditioned on assay setup even by a caller that tries.
        """
        with torch.no_grad():
            base = model(**sequence_inputs)
            with_ctx = model(**sequence_inputs, tcell_context=FORBIDDEN_TCELL_CONTEXT)
        assert torch.allclose(base["tcell_logit"], with_ctx["tcell_logit"]), (
            "the T-cell prediction moved when an assay setup was supplied"
        )

    @pytest.mark.parametrize("key", sorted(FORBIDDEN_TCELL_CONTEXT))
    def test_head_signature_rejects_each_forbidden_key(self, key):
        """Not merely ignored -- absent, so it cannot be reintroduced quietly."""
        import inspect

        from presto.models.heads import TCellAssayHead

        parameters = inspect.signature(TCellAssayHead.forward).parameters
        assert key not in parameters, f"{key} is back on the T-cell head's forward signature"

    def test_training_loop_does_not_supply_tcell_context(self):
        import inspect

        from presto.scripts import train_synthetic

        from source_probe import region_between

        # Windowed, not whole-function: `tcell_context=` legitimately appears
        # further down in the MIL call, which passes it as None on purpose.
        # `region_between` requires both markers to be unique, so a second
        # forward call fails here instead of being silently skipped.
        source = inspect.getsource(train_synthetic.compute_loss)
        forward = region_between(source, "outputs = model(", "provenance=", where="compute_loss")
        assert "tcell_context=" not in forward


class TestPanelsCoverEveryConfiguration:
    @pytest.mark.parametrize(
        "key,expected_axis",
        [
            ("binding_assay_panel_assay_type", "assay type"),
            ("binding_assay_panel_assay_prep", "sample prep"),
            ("binding_assay_panel_assay_geometry", "assay geometry"),
            ("binding_assay_panel_assay_readout", "readout chemistry"),
        ],
    )
    def test_one_output_track_per_value(self, model, sequence_inputs, key, expected_axis):
        with torch.no_grad():
            out = model(**sequence_inputs)
        assert key in out, f"no output track for {expected_axis}"
        assert out[key].shape[0] == 3
        assert out[key].shape[1] > 1, f"{expected_axis} has only one track"

    def test_tcell_panel_still_present(self, model, sequence_inputs):
        with torch.no_grad():
            out = model(**sequence_inputs)
        panel = out.get("tcell_panel_logits") or out.get("tcell_context_logits")
        assert panel, "the T-cell condition panel disappeared"
        assert "assay_method" in panel


class TestCellularStateIsAnOutputAxis:
    """APM state and stimulus are predicted across, not conditioned on.

    `docs/assay_modeling_contract.md` names "stimulation context" as a
    forbidden input. The gap-2 fix had routed it inward on purpose -- into the
    processing latent -- because that was the only way found at the time to
    give the in-vivo excision profiles gradient. Sweeping the profiles instead
    keeps the gradient and drops the input.
    """

    def _provenance_pair(self, batch):
        """Two provenances differing ONLY in cellular state.

        Holding `peptide_source_idx` fixed matters: it selects which branch
        computes the termini, which is structural routing rather than a
        condition, and varying it legitimately changes the prediction. An
        earlier version of this check compared against *no* provenance at all
        and so measured the source fork, not the conditions.
        """
        base = dict(batch.provenance)
        altered = dict(batch.provenance)
        altered["apm_perturbation_idx"] = torch.full_like(base["apm_perturbation_idx"], 3)
        altered["processing_stimulus_idx"] = torch.full_like(base["processing_stimulus_idx"], 2)
        return base, altered

    @pytest.mark.parametrize(
        "output_key",
        ["binding_logit", "presentation_logit", "elution_logit", "excision_logit"],
    )
    def test_prediction_does_not_move_with_cellular_state(self, output_key):
        import sys

        sys.path.insert(0, "tests")
        from test_gradient_coverage import _every_modality_batch

        batch = _every_modality_batch()
        torch.manual_seed(0)
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        model.eval()
        base_prov, altered_prov = self._provenance_pair(batch)
        kwargs = dict(
            pep_tok=batch.pep_tok,
            mhc_a_tok=batch.mhc_a_tok,
            mhc_b_tok=batch.mhc_b_tok,
            mhc_class=batch.mhc_class,
        )
        with torch.no_grad():
            base = model(**kwargs, provenance=base_prov)
            altered = model(**kwargs, provenance=altered_prov)
        assert torch.allclose(base[output_key], altered[output_key]), (
            f"{output_key} moved when the cellular condition changed; the "
            "model is conditioned on cell state rather than predicting across it"
        )

    def test_the_panel_distinguishes_conditions_once_trained(self):
        """Invariance must not be achieved by the panel being degenerate.

        At initialization the profiles are zero, so every column agrees. Give
        one condition a profile and its column must move -- otherwise the
        conditions are represented but unusable.
        """
        import sys

        sys.path.insert(0, "tests")
        from test_gradient_coverage import _every_modality_batch

        batch = _every_modality_batch()
        torch.manual_seed(0)
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        model.eval()
        with torch.no_grad():
            model.excision_head.invivo_profile_c[3].fill_(1.5)
            model.excision_head.invivo_bias[3] = 0.7
            out = model(
                pep_tok=batch.pep_tok,
                mhc_a_tok=batch.mhc_a_tok,
                mhc_b_tok=batch.mhc_b_tok,
                mhc_class=batch.mhc_class,
                provenance=batch.provenance,
            )
        panel = out["excision_panel_apm"]
        assert not torch.allclose(panel[:, 0], panel[:, 3])

    def test_no_cellular_state_embedding_on_the_input_path(self):
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        assert not hasattr(model, "processing_condition_embed")


class TestBiologicalStateIsAnInput:
    """The other half of the contract: causal state may be, and is, an input.

    The line is not "is it categorical metadata" -- assay method and APC type
    are both that. It is whether the field describes the system being measured
    or the instrument measuring it. A TAP-null cell genuinely presents a
    different repertoire, and someone asking "will this tumour present this
    peptide" knows which tumour they mean, so conditioning on it uses
    information rather than leaking it.

    An earlier pass removed these along with the assay descriptors, which
    discarded real signal. See docs/assay_modeling_contract.md.
    """

    def _batch(self):
        import sys

        sys.path.insert(0, "tests")
        from test_gradient_coverage import _every_modality_batch

        return _every_modality_batch()

    @pytest.mark.parametrize(
        "provenance_key,alternate",
        [
            ("apm_perturbation_idx", 3),
            ("processing_stimulus_idx", 2),
            ("cell_lineage_idx", 5),
            ("sample_origin_idx", 2),
            ("disease_state_idx", 3),
        ],
    )
    def test_state_reaches_the_prediction_once_trained(self, provenance_key, alternate):
        """Zero-init means no effect at init; the effect must appear with values.

        Asserting invariance here would be asserting the bug: these axes exist
        precisely so the model can tell a dendritic cell from a carcinoma line.
        """
        batch = self._batch()
        torch.manual_seed(0)
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        model.eval()
        with torch.no_grad():
            model.cell_lineage_embed.weight.normal_(0.0, 0.5)
            model.sample_origin_embed.weight.normal_(0.0, 0.5)
            model.disease_state_embed.weight.normal_(0.0, 0.5)
            model.excision_head.invivo_profile_c.normal_(0.0, 0.5)
            model.excision_head.stimulus_profile_c.normal_(0.0, 0.5)
        base_prov = dict(batch.provenance)
        altered = dict(batch.provenance)
        altered[provenance_key] = torch.full_like(base_prov[provenance_key], alternate)
        kwargs = dict(
            pep_tok=batch.pep_tok,
            mhc_a_tok=batch.mhc_a_tok,
            mhc_b_tok=batch.mhc_b_tok,
            mhc_class=batch.mhc_class,
        )
        with torch.no_grad():
            a = model(**kwargs, provenance=base_prov)
            b = model(**kwargs, provenance=altered)
        moved = any(
            not torch.allclose(a[key], b[key])
            for key in ("presentation_logit", "elution_logit", "excision_logit")
        )
        assert moved, (
            f"{provenance_key} had no effect on any presentation-path output; "
            "the biological state is being ignored"
        )

    def test_provenance_axes_are_plumbed_from_the_record(self):
        """The record field existed for months without reaching the model."""
        from presto.data.collate import PrestoCollator
        from presto.data.loaders import ElutionRecord, PrestoDataset

        seq = "GSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPR"
        dataset = PrestoDataset(
            elution_records=[
                ElutionRecord(
                    peptide="LLDGTATLRF",
                    alleles=["HLA-A*02:01"],
                    detected=True,
                    cell_type="Monocyte",
                ),
                ElutionRecord(
                    peptide="SIINFEKLAA",
                    alleles=["HLA-A*02:01"],
                    detected=True,
                    cell_type="Epithelial cell",
                ),
            ],
            mhc_sequences={"HLA-A*02:01": seq},
            strict_mhc_resolution=False,
        )
        assert dataset[0].cell_lineage == "myeloid"
        assert dataset[1].cell_lineage == "epithelial"

        batch = PrestoCollator()([dataset[i] for i in range(2)])
        assert "cell_lineage_idx" in batch.provenance
        assert "sample_origin_idx" in batch.provenance
        assert "disease_state_idx" in batch.provenance

    def test_unrecognized_cell_types_are_unknown_not_guessed(self):
        """A wrong processing phenotype is worse than an absent one."""
        from presto.data.vocab import cell_lineage_for

        assert cell_lineage_for(None) == "unknown"
        assert cell_lineage_for("") == "unknown"
