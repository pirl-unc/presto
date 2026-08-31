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

#: (D) Structurally masked by design. `p1_profile_c` is overridden by
#: `pinned_mask` wherever a protease's P1 rule is pinned, and the proteasome
#: mixture needs rows from more than one component to have anything to weigh.
BY_DESIGN = {
    "excision_head.p1_profile_c",
    "excision_head.p1_prime_penalty",
    "excision_head.mixture_logits",
    "excision_head.profile_scale_n",
}

#: (B) Reachable, but only with data this batch does not contain -- an EC50 or
#: Tm measurement, a class II binding row carrying flanking regions, a species
#: override, or TCR method metadata. Not defects; add the record and they
#: train. Verified individually: a class II T-cell record moved
#: `immunogenicity_cd4_latent_head` out of this set, and a class II processing
#: record moved `class2_processing_predictor` out.
NEEDS_ABSENT_DATA = {
    "affinity_predictor.assay_heads.ec50_residual.0.weight",
    "affinity_predictor.assay_heads.ec50_residual.0.bias",
    "affinity_predictor.assay_heads.ec50_residual.2.weight",
    "affinity_predictor.assay_heads.ec50_residual.2.bias",
    "affinity_predictor.assay_heads.tm.head.0.weight",
    "affinity_predictor.assay_heads.tm.head.0.bias",
    "affinity_predictor.assay_heads.tm.head.2.weight",
    "affinity_predictor.assay_heads.tm.head.2.bias",
    "class2_pfr_score.0.weight",
    "class2_pfr_score.0.bias",
    "species_override_embed.weight",
    "tcr_evidence_method_head.weight",
    "tcr_evidence_method_head.bias",
}

#: (C) Computed, published, and consumed by nothing.
#:
#: **This category is now empty**, and keeping it here with that statement is
#: deliberate: it held 18 tensors, and every one turned out to be a distinct
#: kind of disconnection rather than a design choice.
#:
#:   assay_{type,prep,geometry,readout}_embed + factorized_proj
#:       Dead in *every* mode, because `binding_context` was never passed from
#:       the training loop -- the collator built the metadata and the model
#:       accepted it, with nothing in between. Now passed; under residual modes
#:       that read the factorized context they train, and under `legacy`, which
#:       cannot reach them, they are no longer allocated.
#:   sequence_summary_proj
#:       Mode-gated only. Allocated when a mode consumes it.
#:   binding_stability_score_head
#:       Passed to the stability heads as a literal `None`, so a reserved input
#:       channel was permanently zeroed and the head starved. Now fed.
#:   recognition_cd{8,4}_head
#:       Published probabilities from untrained weights. Now upstream of
#:       immunogenicity, which is what the DAG said all along: recognition is
#:       repertoire precursor frequency (S9.4), immunogenicity is the response
#:       requiring it (S9.5).
#:
#: If something lands here again, it means an output is being published that
#: nothing trains -- which is worth a fix, not an entry.
COMPUTED_BUT_UNSUPERVISED: set = set()

ALLOWED_DEAD = BY_DESIGN | NEEDS_ABSENT_DATA | COMPUTED_BUT_UNSUPERVISED


def _every_modality_batch():
    """One batch touching every record type the trainer supports."""
    dataset = PrestoDataset(
        elution_records=[
            ElutionRecord(
                peptide="LLDGTATLRF",
                alleles=["HLA-A*02:01", "HLA-B*07:02"],
                detected=True,
                stimulus="ifn_gamma",
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
        ],
        stability_records=[
            StabilityRecord(peptide="RTLNAWVKVV", mhc_allele="HLA-A*02:01", t_half=4.0)
        ],
        kinetics_records=[
            KineticsRecord(peptide="YLLEMLWRL", mhc_allele="HLA-A*02:01", koff=0.01)
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
        if param.grad is None or float(param.grad.abs().sum()) == 0.0
    }
    return model, dead


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

    def test_allowlist_has_no_stale_entries(self, gradient_report):
        """A fixed exception must be removed, so the fix shows in the diff."""
        model, dead = gradient_report
        known = {name for name, _ in model.named_parameters()}
        stale = sorted((ALLOWED_DEAD & known) - dead)
        assert stale == [], (
            f"these are now trained and should leave the allowlist: {stale}"
        )

    def test_allowlist_refers_to_real_parameters(self, gradient_report):
        """Guards against a rename quietly emptying the allowlist."""
        model, _ = gradient_report
        known = {name for name, _ in model.named_parameters()}
        missing = sorted(ALLOWED_DEAD - known)
        assert missing == [], (
            f"allowlist names parameters that no longer exist: {missing}"
        )

    def test_the_vast_majority_of_parameters_train(self, gradient_report):
        model, dead = gradient_report
        sizes = {name: p.numel() for name, p in model.named_parameters()}
        dead_params = sum(sizes[name] for name in dead)
        total = sum(sizes.values())
        assert dead_params / total < 0.05, (
            f"{dead_params}/{total} parameters are untrained "
            f"({100 * dead_params / total:.1f}%)"
        )


class TestGapTwoStaysClosed:
    """The specific regression this whole effort started from."""

    @pytest.mark.parametrize(
        "name",
        [
            "excision_head.invivo_profile_c",
            "excision_head.invivo_profile_n",
            "excision_head.stimulus_profile_c",
            "excision_head.invivo_bias",
        ],
    )
    def test_in_vivo_excision_parameters_are_trained(self, gradient_report, name):
        _, dead = gradient_report
        assert name not in dead


class TestCategoriesAreDisjoint:
    def test_no_parameter_is_in_two_categories(self):
        counts = collections.Counter(
            list(BY_DESIGN) + list(NEEDS_ABSENT_DATA) + list(COMPUTED_BUT_UNSUPERVISED)
        )
        assert [name for name, n in counts.items() if n > 1] == []
