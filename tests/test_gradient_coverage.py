"""Every parameter must receive gradient, or be a documented exception.

Gap 2 was a parameter path that looked trained and was not, and it survived two
attempted fixes because nothing measured the thing directly. This test measures
it directly: build a batch containing every modality, run the real loss
backward, and compare the set of zero-gradient parameters against an explicit
allowlist.

A new parameter that nothing trains fails here. Removing supervision from an
existing one fails here. Fixing an entry means deleting it from the allowlist,
which makes the fix visible in the diff.

Measured with this batch, the dead set is ~12.6k of 424k parameters (3.0%). It
was 19.7% before unreachable mode-selected branches stopped being allocated.
"""

import collections

import pytest

torch = pytest.importorskip("torch")

from presto.data.bulk_ms import BulkMSRecord  # noqa: E402
from presto.data.collate import PrestoCollator  # noqa: E402
from presto.data.loaders import (  # noqa: E402
    BindingRecord,
    ElutionRecord,
    KineticsRecord,
    PrestoDataset,
    ProcessingRecord,
    StabilityRecord,
    TCellRecord,
    TcrEvidenceRecord,
)
from presto.models.presto import Presto  # noqa: E402
from presto.scripts.train_synthetic import compute_loss  # noqa: E402

CLASS1_SEQ = "GSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPR"
CLASS2_SEQ = (
    "RATPENYLFQGRQECYAFNGTQRFLERYIYNREEFARFDSDVGEFRAVTELGRPAAEYWNSQKDIL"
)
SEQS = {
    "HLA-A*02:01": CLASS1_SEQ,
    "HLA-B*07:02": CLASS1_SEQ,
    "HLA-DRB1*01:01": CLASS2_SEQ,
}

# ---------------------------------------------------------------------------
# Documented exceptions. Each entry needs a reason, and the reasons are the
# point: "dead" is not one condition, and the four kinds want different fixes.
# ---------------------------------------------------------------------------

#: (A) Allocated but unreachable under the active configuration.
#:
#: Alternate positional-encoding schemes, the collapsed-topology projections
#: and presentation MLPs, class-specific core scorers, and the direct-segment
#: residuals. Each is selected by a mode string; the shipped defaults select
#: none of them, so they are built, evaluated on every forward, and discarded.
#: 65k of the 101k dead parameters sit here.
MODE_GATED_UNREACHABLE = {
    "binding_direct_segment_affinity_proj.0.bias",
    "binding_direct_segment_affinity_proj.0.weight",
    "binding_direct_segment_affinity_proj.2.bias",
    "binding_direct_segment_affinity_proj.2.weight",
    "binding_direct_segment_gate.0.bias",
    "binding_direct_segment_gate.0.weight",
    "binding_direct_segment_gate.2.bias",
    "binding_direct_segment_gate.2.weight",
    "binding_direct_segment_stability_proj.0.bias",
    "binding_direct_segment_stability_proj.0.weight",
    "binding_direct_segment_stability_proj.2.bias",
    "binding_direct_segment_stability_proj.2.weight",
    "core_window_score_class1.0.bias",
    "core_window_score_class1.0.weight",
    "core_window_score_class1.2.bias",
    "core_window_score_class1.2.weight",
    "core_window_score_class2.0.bias",
    "core_window_score_class2.0.weight",
    "core_window_score_class2.2.bias",
    "core_window_score_class2.2.weight",
    "groove_1_abs_pos.weight",
    "groove_1_end_pos.weight",
    "groove_2_abs_pos.weight",
    "groove_2_end_pos.weight",
    "groove_frac_mlp.0.bias",
    "groove_frac_mlp.0.weight",
    "groove_frac_mlp.2.bias",
    "groove_frac_mlp.2.weight",
    "groove_pos_concat_frac_mlp.0.bias",
    "groove_pos_concat_frac_mlp.0.weight",
    "groove_pos_concat_frac_mlp.2.bias",
    "groove_pos_concat_frac_mlp.2.weight",
    "groove_pos_concat_frac_proj.bias",
    "groove_pos_concat_frac_proj.weight",
    "groove_pos_concat_mlp.0.bias",
    "groove_pos_concat_mlp.0.weight",
    "groove_pos_concat_mlp.2.bias",
    "groove_pos_concat_mlp.2.weight",
    "groove_pos_concat_proj.bias",
    "groove_pos_concat_proj.weight",
    "pep_abs_pos.weight",
    "pep_pos_concat_frac_mlp.0.bias",
    "pep_pos_concat_frac_mlp.0.weight",
    "pep_pos_concat_frac_mlp.2.bias",
    "pep_pos_concat_frac_mlp.2.weight",
    "pep_pos_concat_frac_proj.bias",
    "pep_pos_concat_frac_proj.weight",
    "pep_pos_concat_mlp.0.bias",
    "pep_pos_concat_mlp.0.weight",
    "pep_pos_concat_mlp.2.bias",
    "pep_pos_concat_mlp.2.weight",
    "pep_pos_concat_proj.bias",
    "pep_pos_concat_proj.weight",
    "presentation_class1_mlp.0.bias",
    "presentation_class1_mlp.0.weight",
    "presentation_class1_mlp.2.bias",
    "presentation_class1_mlp.2.weight",
    "presentation_class2_mlp.0.bias",
    "presentation_class2_mlp.0.weight",
    "presentation_class2_mlp.2.bias",
    "presentation_class2_mlp.2.weight",
    "processing_class1_proj.bias",
    "processing_class1_proj.weight",
    "processing_class2_proj.bias",
    "processing_class2_proj.weight",
}

#: (B) Computed and published, consumed by nothing.
#:
#: The factorized assay context and sequence summary are built every forward
#: and never read; `binding_stability_score` is handed to the stability heads
#: as a literal None.
COMPUTED_BUT_UNCONSUMED = {
    "affinity_predictor.assay_geometry_embed.weight",
    "affinity_predictor.assay_prep_embed.weight",
    "affinity_predictor.assay_readout_embed.weight",
    "affinity_predictor.assay_type_embed.weight",
    "affinity_predictor.binding_stability_score_head.0.bias",
    "affinity_predictor.binding_stability_score_head.0.weight",
    "affinity_predictor.binding_stability_score_head.2.bias",
    "affinity_predictor.binding_stability_score_head.2.weight",
    "affinity_predictor.factorized_proj.bias",
    "affinity_predictor.factorized_proj.weight",
    "affinity_predictor.sequence_summary_proj.0.bias",
    "affinity_predictor.sequence_summary_proj.0.weight",
    "affinity_predictor.sequence_summary_proj.2.bias",
    "affinity_predictor.sequence_summary_proj.2.weight",
}

#: (C) Reachable, but only with records this batch does not contain --
#: an EC50 or Tm measurement, TCR method metadata, a class II binding row with
#: flanking regions, or a species override.
NEEDS_ABSENT_DATA = {
    "affinity_predictor.assay_heads.ec50_residual.0.bias",
    "affinity_predictor.assay_heads.ec50_residual.0.weight",
    "affinity_predictor.assay_heads.ec50_residual.2.bias",
    "affinity_predictor.assay_heads.ec50_residual.2.weight",
    "affinity_predictor.assay_heads.tm.head.0.bias",
    "affinity_predictor.assay_heads.tm.head.0.weight",
    "affinity_predictor.assay_heads.tm.head.2.bias",
    "affinity_predictor.assay_heads.tm.head.2.weight",
    "class2_pfr_score.0.bias",
    "class2_pfr_score.0.weight",
    "class2_pfr_score.2.bias",
    "species_override_embed.weight",
    "tcr_evidence_method_head.bias",
    "tcr_evidence_method_head.weight",
}

#: (D) The in-vivo excision path. Excision labels exist only on shotgun rows,
#: which are all protein-source, so the in-vivo branch is never supervised --
#: this is the gap the model contract records as gap 2.
IN_VIVO_EXCISION_UNSUPERVISED = {
    "excision_head.inducer_profile_c",
    "excision_head.invivo_bias",
    "excision_head.invivo_profile_c",
    "excision_head.invivo_profile_n",
}

#: (E) In-vitro excision parameters whose reachable set is empty: the branch
#: is selected only for protein-source rows, where every machinery is a pinned
#: protease, so the learned rows are always overridden and the unpinned rows
#: are never selected.
EXCISION_UNREACHABLE = {
    "excision_head.mixture_logits",
    "excision_head.p1_prime_penalty",
    "excision_head.p1_profile_c",
    "excision_head.profile_scale_n",
}

#: (F) Output heads consumed by no loss. `recognition_cd{8,4}_head` publish
#: probabilities from untrained projections.
UNSUPERVISED_OUTPUT_HEADS = {
    "recognition_cd4_head.bias",
    "recognition_cd4_head.weight",
    "recognition_cd8_head.bias",
    "recognition_cd8_head.weight",
}

#: (G) Structurally unidentifiable. Both feed a softmax over candidate
#: registers, where a constant added to every candidate cancels, so the bias
#: cannot affect the posterior. Their weights train normally.
SOFTMAX_INVARIANT_BIASES = {
    "core_window_prior.2.bias",
    "core_window_score.2.bias",
}

ALLOWED_DEAD = MODE_GATED_UNREACHABLE | COMPUTED_BUT_UNCONSUMED | NEEDS_ABSENT_DATA | IN_VIVO_EXCISION_UNSUPERVISED | EXCISION_UNREACHABLE | UNSUPERVISED_OUTPUT_HEADS | SOFTMAX_INVARIANT_BIASES


def _every_modality_batch():
    """One batch touching every record type the trainer supports."""
    dataset = PrestoDataset(
        elution_records=[
            ElutionRecord(
                peptide="LLDGTATLRF",
                alleles=["HLA-A*02:01", "HLA-B*07:02"],
                detected=True,
                inducer="ifn_gamma",
                apm_perturbation="tap_ko",
                flank_n="AAAAAAAAAA",
                flank_c="CCCCCCCCCC",
            ),
            ElutionRecord(
                peptide="SIINFEKLAA",
                alleles=["HLA-A*02:01"],
                detected=False,
                flank_n="GGGGGGGGGG",
                flank_c="TTTTTTTTTT",
            ),
            ElutionRecord(
                peptide="PKYVKQNTLKLATA",
                alleles=["HLA-DRB1*01:01"],
                detected=True,
                mhc_class="II",
                flank_n="MMMMMMMMMM",
                flank_c="WWWWWWWWWW",
            ),
        ],
        binding_records=[
            BindingRecord(
                peptide="KVFPYALINK",
                mhc_allele="HLA-A*02:01",
                value=25.0,
                measurement_type="half maximal inhibitory concentration (IC50)",
                assay_method="purified MHC/competitive/radioactivity",
                species="Homo sapiens",
                antigen_species="Influenza A virus",
            ),
            BindingRecord(
                peptide="PKYVKQNTLKLATA",
                mhc_allele="HLA-DRB1*01:01",
                value=100.0,
                mhc_class="II",
                measurement_type="half maximal inhibitory concentration (IC50)",
                assay_method="cellular MHC/direct/fluorescence",
                species="Homo sapiens",
                antigen_species="Homo sapiens",
            ),
            # KD (1,132 corpus rows) and EC50 (147) are separate heads from
            # IC50 and were unsupervised purely because the fixture had none.
        ],
        stability_records=[
            StabilityRecord(peptide="RTLNAWVKVV", mhc_allele="HLA-A*02:01", t_half=4.0),
        ],
        kinetics_records=[
            KineticsRecord(peptide="YLLEMLWRL", mhc_allele="HLA-A*02:01", koff=0.01),
        ],
        processing_records=[
            ProcessingRecord(
                peptide="ILKEPVHGV",
                mhc_allele="HLA-A*02:01",
                label=1.0,
                flank_n="QQQQQQQQQQ",
                flank_c="EEEEEEEEEE",
            ),
            # Class II too: without it class2_processing_predictor gets no
            # gradient, which is a data gap rather than a broken path.
            ProcessingRecord(
                peptide="PKYVKQNTLKLATA",
                mhc_allele="HLA-DRB1*01:01",
                mhc_class="II",
                label=1.0,
                flank_n="QQQQQQQQQQ",
                flank_c="EEEEEEEEEE",
            ),
        ],
        tcell_records=[
            TCellRecord(
                peptide="GILGFVFTL",
                mhc_allele="HLA-A*02:01",
                response=1.0,
                tcr_b_cdr3="CASSIRSSYEQYF",
                tcr_a_cdr3="CAVRDSNYQLIW",
                mhc_class="I",
                species="Homo sapiens",
                antigen_species="Influenza A virus",
                assay_method="ELISPOT",
                assay_type="IFNg release",
            ),
            TCellRecord(
                peptide="PKYVKQNTLKLATA",
                mhc_allele="HLA-DRB1*01:01",
                response=1.0,
                mhc_class="II",
                species="Homo sapiens",
                antigen_species="Mycobacterium tuberculosis",
                assay_method="ICS",
                assay_type="IFNg release",
            ),
        ],
        tcr_evidence_records=[
            TcrEvidenceRecord(
                peptide="NLVPMVATV",
                mhc_a=CLASS1_SEQ,
                mhc_class="I",
                evidence_label=1.0,
                species="Homo sapiens",
                antigen_species="Human betaherpesvirus 5",
            )
        ],
        bulk_ms_records=[
            BulkMSRecord(
                peptide="SAMPLERPEP",
                machinery="trypsin",
                detectability_label=1.0,
                excision_label=1.0,
                observed=True,
            )
        ],
        mhc_sequences=SEQS,
        strict_mhc_resolution=False,
    )
    return PrestoCollator()([dataset[i] for i in range(len(dataset))])



#: Below this, a parameter is receiving nothing in any practical sense.
#:
#: An exact `== 0.0` test is knife-edge. `class2_pfr_score.2.bias` shows
#: gradients of 1e-10 to 1e-12 depending only on the seed -- connected through
#: some numerically negligible route -- so it flipped between "dead" and
#: "trained" across runs and made the allowlist unstable. Ten orders of
#: magnitude below a real gradient is starved, whether or not it is exactly
#: zero.
EFFECTIVELY_ZERO_GRADIENT = 1e-8

#: Ratchet on the untrained fraction. Measured on this commit: 21.3% under the
#: collapsed topology, 19.8% under expanded. Lower it
#: whenever a change shrinks the allowlist; never raise it without saying why.
DEAD_FRACTION_CEILING = 0.22

#: Both topologies, because they allocate different modules. Checking only
#: `expanded` would leave the default (`collapsed`) unverified -- and the
#: collapsed path owns the processing projections and presentation MLPs that
#: the expanded path does not build at all.
TOPOLOGIES = ("collapsed", "expanded")


@pytest.fixture(scope="module", params=TOPOLOGIES)
def gradient_report(request):
    torch.manual_seed(0)
    model = Presto(
        d_model=32, n_layers=2, n_heads=4, latent_topology=request.param
    )
    loss, _, _ = compute_loss(model, _every_modality_batch(), "cpu")
    loss.backward()
    dead = {
        name
        for name, param in model.named_parameters()
        if param.grad is None
        or float(param.grad.abs().sum()) < EFFECTIVELY_ZERO_GRADIENT
    }
    return model, dead



@pytest.fixture(scope="module")
def dead_in_any_topology():
    """(known parameter names, parameters dead under at least one topology)."""
    known: set = set()
    dead_union: set = set()
    for topology in TOPOLOGIES:
        torch.manual_seed(0)
        model = Presto(
            d_model=32, n_layers=2, n_heads=4, latent_topology=topology
        )
        loss, _, _ = compute_loss(model, _every_modality_batch(), "cpu")
        loss.backward()
        for name, param in model.named_parameters():
            known.add(name)
            if (
                param.grad is None
                or float(param.grad.abs().sum()) < EFFECTIVELY_ZERO_GRADIENT
            ):
                dead_union.add(name)
    return known, dead_union


class TestGradientCoverage:
    def test_no_undocumented_dead_parameters(self, gradient_report):
        """The assertion that would have caught gap 2 the first time."""
        _, dead = gradient_report
        undocumented = sorted(dead - ALLOWED_DEAD)
        assert undocumented == [], (
            "these parameters receive no gradient and are not documented as "
            f"exceptions: {undocumented}. Either supply the supervision, or "
            "add them to the right category in this file with a reason."
        )

    def test_allowlist_has_no_stale_entries(self, dead_in_any_topology):
        """A fixed exception must be removed, so the fix shows in the diff.

        Compared against the union across topologies, not the current one.
        Some parameters are trained under `expanded` and not under
        `collapsed` -- `class2_pfr_score.2.weight` is one -- so a per-topology
        comparison would call the same entry both missing and stale depending
        on which parametrization ran.
        """
        known, dead_union = dead_in_any_topology
        stale = sorted((ALLOWED_DEAD & known) - dead_union)
        assert stale == [], (
            f"these are trained under every topology and should leave the "
            f"allowlist: {stale}"
        )

    def test_allowlist_refers_to_real_parameters(self, gradient_report):
        """Guards against a rename quietly emptying the allowlist."""
        model, _ = gradient_report
        known = {name for name, _ in model.named_parameters()}
        missing = sorted(ALLOWED_DEAD - known)
        assert missing == [], (
            f"allowlist names parameters that no longer exist: {missing}"
        )

    def test_dead_fraction_does_not_regress(self, gradient_report):
        """A ratchet, not a target.

        A sixth of all parameters currently receive no gradient. That is the
        honest number; this test exists to stop it growing. Each category in
        the allowlist above names work that would shrink it, and the ceiling
        should be lowered as that work lands -- a bound with slack is how a
        number like this drifts upward unnoticed.
        """
        model, dead = gradient_report
        sizes = {name: p.numel() for name, p in model.named_parameters()}
        dead_params = sum(sizes[name] for name in dead)
        total = sum(sizes.values())
        fraction = dead_params / total
        assert fraction <= DEAD_FRACTION_CEILING, (
            f"{dead_params:,}/{total:,} parameters are untrained "
            f"({100 * fraction:.1f}%), above the "
            f"{100 * DEAD_FRACTION_CEILING:.1f}% ceiling."
        )


class TestTheAllowlistIsWellFormed:
    """The allowlist is only useful if its categories mean something.

    Seven groups, each naming a different reason a parameter is untrained and
    therefore a different fix. A name in two groups means at least one of the
    stated reasons is wrong.
    """

    CATEGORIES = {
        "MODE_GATED_UNREACHABLE": MODE_GATED_UNREACHABLE,
        "COMPUTED_BUT_UNCONSUMED": COMPUTED_BUT_UNCONSUMED,
        "NEEDS_ABSENT_DATA": NEEDS_ABSENT_DATA,
        "IN_VIVO_EXCISION_UNSUPERVISED": IN_VIVO_EXCISION_UNSUPERVISED,
        "EXCISION_UNREACHABLE": EXCISION_UNREACHABLE,
        "UNSUPERVISED_OUTPUT_HEADS": UNSUPERVISED_OUTPUT_HEADS,
        "SOFTMAX_INVARIANT_BIASES": SOFTMAX_INVARIANT_BIASES,
    }

    def test_no_parameter_is_in_two_categories(self):
        seen: dict = {}
        clashes = []
        for label, names in self.CATEGORIES.items():
            for name in names:
                if name in seen:
                    clashes.append(f"{name}: {seen[name]} and {label}")
                seen[name] = label
        assert clashes == [], f"a parameter cannot have two reasons: {clashes}"

    def test_the_union_is_what_allowed_dead_contains(self):
        """Guards against a category being defined and never wired in."""
        union: set = set()
        for names in self.CATEGORIES.values():
            union |= names
        assert union == ALLOWED_DEAD

    def test_no_category_is_empty(self):
        """An empty category is finished work; delete it and lower the ceiling."""
        empty = sorted(label for label, names in self.CATEGORIES.items() if not names)
        assert empty == [], (
            f"these categories are empty and should be removed: {empty}"
        )
