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

#: (D) Unreachable by the intersection of branch and machinery -- and harmless.
#:
#: These four belong to the *in-vitro* excision branch, which is selected only
#: for protein-source (shotgun) rows. On those rows the machinery is always one
#: of the four proteases, and all four are pinned to their known P1 rules:
#:
#:     unknown / proteasome / cathepsin   unpinned, but in-vivo -- these rows
#:                                        use invivo_profile_c, not p1_profile_c
#:     trypsin / chymotrypsin / lysc /    pinned, so the learned rows are
#:     gluc                               overridden wherever they are read
#:
#: So the free rows are never selected and the selected rows are always
#: overridden. Calling this "masked by design" was imprecise: nothing masks
#: them deliberately, the reachable set is simply empty.
#:
#: Left allocated rather than reshaped. Shrinking `p1_profile_c` to the four
#: in-vitro machineries would need index remapping through every call site for
#: 196 parameters at d_model=32, and the in-vitro P1 rules are *known* -- that
#: is why they are pinned -- so there is nothing to learn there anyway.
BY_DESIGN = {
    "excision_head.p1_profile_c",
    "excision_head.p1_prime_penalty",
    "excision_head.mixture_logits",
    "excision_head.profile_scale_n",
}

#: (B) Reachable, but only with data this batch does not contain.
#:
#: This set has shrunk to two entries. EC50, Tm and TCR-method parameters left
#: it once the fixture supplied those record types -- the corpus carries 147
#: EC50 and 27 Tm rows for HLA-A*02:01 alone, so they were never untrainable,
#: only unfed. What remains needs a class II binding row carrying flanking
#: regions, and a species override.
NEEDS_ABSENT_DATA = {
    "class2_pfr_score.0.weight",
    "class2_pfr_score.0.bias",
    "species_override_embed.weight",
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
            # KD (1,132 corpus rows) and EC50 (147) are separate heads from
            # IC50 and were unsupervised purely because the fixture had none.
            BindingRecord(
                peptide="FLPSDFFPSV",
                mhc_allele="HLA-A*02:01",
                value=12.0,
                measurement_type="dissociation constant KD",
                assay_method="purified MHC/direct/fluorescence",
            ),
            BindingRecord(
                peptide="YLLEMLWRL",
                mhc_allele="HLA-A*02:01",
                value=250.0,
                measurement_type="half maximal effective concentration (EC50)",
                assay_method="cellular MHC/direct/fluorescence",
            ),
        ],
        stability_records=[
            StabilityRecord(peptide="RTLNAWVKVV", mhc_allele="HLA-A*02:01", t_half=4.0),
            # Tm exists in the corpus (27 rows for HLA-A*02:01 alone) and the
            # fixture lacked it, so the `tm` head went unsupervised here while
            # being perfectly trainable from real data.
            StabilityRecord(peptide="FLPSDFFPSV", mhc_allele="HLA-A*02:01", tm=62.0),
        ],
        kinetics_records=[
            KineticsRecord(peptide="YLLEMLWRL", mhc_allele="HLA-A*02:01", koff=0.01),
            KineticsRecord(peptide="GILGFVFTL", mhc_allele="HLA-A*02:01", kon=1e5),
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
                apc_name="Dendritic cell",
                effector_culture_condition="PBMC restimulated in vitro",
                apc_culture_condition="peptide-pulsed",
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
                # Method metadata drives tcr_evidence_method, which was
                # unsupervised only because the fixture omitted these.
                method_identification="tetramer",
                method_verification="sequencing",
                method_singlecell="yes",
                method_sequencing="10x",
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
            if param.grad is None or float(param.grad.abs().sum()) == 0.0:
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


# ---------------------------------------------------------------------------
# Every declared task must actually be supervised.
# ---------------------------------------------------------------------------

#: The one task with no data source -- and it does not need one.
#:
#: `core_start` wants gold-standard binding-core positions. No record type
#: carries them and no loader populates them, verified by scanning every
#: dataclass in `data/loaders.py`.
#:
#: That is not a gap, because **the register is already a learned latent**.
#: The model enumerates every candidate register, scores each, softmaxes into
#: `core_window_posterior_prob`, and marginalizes:
#:
#:     interaction_vec = sum(posterior * candidate_vec)
#:
#: so the binding prediction is an expectation over registers and gradient
#: reaches the register scorer from the binding label alone. Measured: a class
#: II binding row carrying no core label trains `core_window_score`,
#: `core_window_prior` and `core_position_embed`. Registers scored per peptide:
#: 1 for a class I 9mer, 3 for an 11mer, 7 for a 15mer, 12 for a 20mer.
#:
#: `core_start` would be a *sharpening* auxiliary on that latent, not a
#: prerequisite. Kept as a spec in case structural alignments are ever
#: available; nothing depends on it.
TASKS_WITHOUT_A_DATA_SOURCE = {"core_start"}


class TestEveryTaskIsSupervised:
    """A declared loss with no supervision is a task nobody is training.

    Six of these were found at once: `binding_kd`, `binding_ec50`, `tm` and
    `kon` were unsupervised only because the fixture lacked those measurement
    types, while the corpus carries 1,132 / 147 / 27 rows of the first three.
    `tcell_apc_type` and `tcell_culture_context` needed APC and culture fields
    on the T-cell record. `tcr_evidence_method` needed method bins that were
    derived in one construction path and silently empty in every other.

    None of that was visible from the loss aggregate, which happily sums
    whatever it is given.
    """

    def test_every_spec_receives_supervision(self):
        from presto.scripts.train_synthetic import LOSS_TASK_SPECS, compute_loss

        torch.manual_seed(0)
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        _, parts, _ = compute_loss(model, _every_modality_batch(), "cpu")
        declared = {getattr(spec, "name", "?") for spec in LOSS_TASK_SPECS}
        unsupervised = sorted(declared - set(parts) - TASKS_WITHOUT_A_DATA_SOURCE)
        assert unsupervised == [], (
            f"these tasks are declared but never supervised: {unsupervised}. "
            "Either supply a record that feeds them, or record why no data "
            "source exists in TASKS_WITHOUT_A_DATA_SOURCE."
        )

    def test_the_exception_still_has_no_data_source(self):
        """If a loader starts populating it, the exemption must go."""
        import dataclasses

        from presto.data import loaders

        carriers = [
            name
            for name in dir(loaders)
            if dataclasses.is_dataclass(getattr(loaders, name))
            and name != "PrestoSample"
            and any(
                field.name == "core_start"
                for field in dataclasses.fields(getattr(loaders, name))
            )
        ]
        assert carriers == [], (
            f"{carriers} now carry core_start; remove it from "
            "TASKS_WITHOUT_A_DATA_SOURCE and supply it in the fixture"
        )

    def test_the_panel_losses_are_supervised_too(self):
        """They sit outside LOSS_TASK_SPECS, so the check above misses them."""
        from presto.scripts.train_synthetic import compute_loss

        torch.manual_seed(0)
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        _, parts, _ = compute_loss(model, _every_modality_batch(), "cpu")
        for name in ("binding_assay_panel", "excision_condition_panel"):
            assert name in parts, f"{name} is computed but never supervised"
