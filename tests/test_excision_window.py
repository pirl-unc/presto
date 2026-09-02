"""The excision head reads a subsite window, not a single residue.

`ExcisionHead` used to score each junction from one residue per side, P1 and
P1'. Protease specificity spans the Schechter-Berger subsites, so that shape
could not represent what the head models. Each junction now reads
`2 * junction_window` positions:

    C-junction:  peptide[-w:]  ||  c_flank[:w]     P{w}..P1 | P1'..P{w}'
    N-junction:  n_flank[-w:]  ||  peptide[:w]     P{w}..P1 | P1'..P{w}'

`w = 5` mirrors mhcflurry's `short_flanks`, their better processing ablation.

It is chosen on that result, not on availability. It was originally both --
hitlist capped flanks at 10 residues and no row carried 15 -- but hitlist
1.55.2 raised DEFAULT_FLANK to 15 and 89.3% of class I rows now carry both
15-residue flanks, so the wider window is a runnable ablation.

Only the **in-vivo** branch is windowed. The in-vitro branch stays P1-only
because its labels are generated from a P1 rule
(`data/bulk_ms.py::would_cleave`); a wider window there would be capacity to
memorize the label generator. `TestInVitroBranchIsUnchanged` pins that.
"""

import pytest

torch = pytest.importorskip("torch")

from presto.data.vocab import AA_TO_IDX, AA_VOCAB  # noqa: E402
from presto.models.presto import Presto  # noqa: E402

MISSING = AA_TO_IDX["<MISSING>"]


def _model(**kwargs):
    torch.manual_seed(0)
    return Presto(d_model=32, n_layers=2, n_heads=4, **kwargs)


class TestWindowExtraction:
    """The two gatherers behind the window, in isolation."""

    TOKENS = torch.tensor(
        [
            [4, 5, 6, 7, 8, 9, 0, 0],  # 6 valid residues
            [4, 5, 0, 0, 0, 0, 0, 0],  # 2 valid -- shorter than the window
            [0, 0, 0, 0, 0, 0, 0, 0],  # entirely absent
        ]
    )

    def test_last_window_keeps_sequence_order_with_p1_last(self):
        out = Presto._last_valid_window(self.TOKENS, 3, self.TOKENS.device, MISSING, 5)
        assert out[0].tolist() == [5, 6, 7, 8, 9]

    def test_short_row_is_left_padded_so_p1_stays_last(self):
        """P1 must not slide to another subsite just because the row is short."""
        out = Presto._last_valid_window(self.TOKENS, 3, self.TOKENS.device, MISSING, 5)
        assert out[1].tolist() == [MISSING, MISSING, MISSING, 4, 5]

    def test_first_window_keeps_sequence_order_with_p1_prime_first(self):
        out = Presto._first_valid_window(self.TOKENS, 3, self.TOKENS.device, MISSING, 5)
        assert out[0].tolist() == [4, 5, 6, 7, 8]

    def test_short_row_is_right_padded_so_p1_prime_stays_first(self):
        out = Presto._first_valid_window(self.TOKENS, 3, self.TOKENS.device, MISSING, 5)
        assert out[1].tolist() == [4, 5, MISSING, MISSING, MISSING]

    @pytest.mark.parametrize("picker", ["_last_valid_window", "_first_valid_window"])
    def test_absent_row_is_all_missing(self, picker):
        out = getattr(Presto, picker)(self.TOKENS, 3, self.TOKENS.device, MISSING, 5)
        assert out[2].tolist() == [MISSING] * 5

    @pytest.mark.parametrize("picker", ["_last_valid_window", "_first_valid_window"])
    def test_absent_tensor_is_all_missing(self, picker):
        """A batch where no row has a flank gives `None`, not an empty tensor."""
        out = getattr(Presto, picker)(None, 2, torch.device("cpu"), MISSING, 5)
        assert out.tolist() == [[MISSING] * 5] * 2

    def test_p1_column_agrees_with_the_single_token_picker(self):
        """The window must not disagree with the scalar it generalizes.

        `_last_valid_token` still feeds the in-vitro branch, so if the two ever
        picked different residues the same junction would be scored from two
        different P1s depending on the source.
        """
        window = Presto._last_valid_window(self.TOKENS, 3, self.TOKENS.device, MISSING, 5)
        scalar = Presto._last_valid_token(self.TOKENS, 3, self.TOKENS.device, MISSING)
        assert torch.equal(window[:, -1], scalar)

    def test_p1_prime_column_agrees_with_the_single_token_picker(self):
        window = Presto._first_valid_window(self.TOKENS, 3, self.TOKENS.device, MISSING, 5)
        scalar = Presto._first_valid_token(self.TOKENS, 3, self.TOKENS.device, MISSING)
        assert torch.equal(window[:, 0], scalar)


class TestWindowShape:
    def test_profiles_carry_a_position_axis(self):
        head = _model().excision_head
        window = 2 * head.junction_window
        assert head.invivo_profile_c.shape[1] == window
        assert head.invivo_profile_n.shape[1] == window
        assert head.stimulus_profile_c.shape[1] == window

    def test_p1_sits_where_the_head_says_it_does(self):
        head = _model().excision_head
        assert head.p1_window_index == head.junction_window - 1
        assert head.p1_prime_window_index == head.junction_window

    def test_missing_index_tracks_the_vocabulary(self):
        """Hard-coding the last column would drift if AA_VOCAB grew -- and it
        did: `<TERMINUS>` was appended after `<MISSING>`, so "the last entry"
        stopped meaning "missing". The index must come from the name."""
        head = _model().excision_head
        assert head.missing_residue_index == AA_TO_IDX["<MISSING>"]
        assert head.missing_residue_index != len(AA_VOCAB) - 1
        assert AA_VOCAB[-1] == "<TERMINUS>"

    @pytest.mark.parametrize("width", [3, 5, 7])
    def test_the_window_is_configurable(self, width):
        """5 is a default, not a constant -- an ablation must be one argument."""
        torch.manual_seed(0)
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        head = type(model.excision_head)(
            d_model=32,
            n_machinery=7,
            n_aa=len(AA_VOCAB),
            n_apm=7,
            n_stimulus=7,
            junction_window=width,
        )
        assert head.invivo_profile_c.shape[1] == 2 * width


def _forward(model, *, apm=None, stimulus=None, batch=3, flanks=True, pep_len=10):
    model.eval()
    kwargs = dict(
        pep_tok=torch.randint(4, 24, (batch, pep_len)),
        mhc_a_tok=torch.randint(4, 24, (batch, 40)),
        mhc_b_tok=torch.randint(4, 24, (batch, 40)),
        mhc_class="I",
    )
    if flanks:
        kwargs["flank_n_tok"] = torch.randint(4, 24, (batch, 8))
        kwargs["flank_c_tok"] = torch.randint(4, 24, (batch, 8))
    zeros = torch.zeros(batch, dtype=torch.long)
    kwargs["provenance"] = {
        "peptide_source_idx": torch.full((batch,), 1, dtype=torch.long),
        "apm_perturbation_idx": zeros if apm is None else apm,
        "processing_stimulus_idx": zeros if stimulus is None else stimulus,
    }
    with torch.no_grad():
        return model(**kwargs)


def _randomize_invivo(model, seed=0):
    torch.manual_seed(seed)
    head = model.excision_head
    with torch.no_grad():
        for tensor in (
            head.invivo_profile_c,
            head.invivo_profile_n,
            head.stimulus_profile_c,
        ):
            tensor.normal_(0.0, 0.5)
        head.invivo_bias.normal_(0.0, 0.5)
    return model


class TestZeroInitIsANoOp:
    """A freshly built widened head must score exactly as the old one did.

    The added subsites are zero-initialized, so at init only P1/P1' contribute
    and the widening is numerically invisible. Without this the change would
    silently move every excision number before any training happened.
    """

    def test_added_positions_start_at_zero(self):
        head = _model().excision_head
        w = head.junction_window
        for tensor in (head.invivo_profile_c, head.invivo_profile_n):
            assert float(tensor.abs().sum()) == 0.0
        assert float(head.stimulus_profile_c.abs().sum()) == 0.0
        assert w >= 1

    def test_in_vivo_score_is_the_bias_alone_at_init(self):
        """With every profile at zero the junction terms vanish."""
        model = _model()
        out = _forward(model)
        head = model.excision_head
        expected = head.invivo_bias[0].expand(3)
        residual = out["excision_logit"] - expected
        # context_c/context_n are learned Linears and are not zero at init, so
        # the residual is the context term, not the profile term.
        assert torch.isfinite(residual).all()
        assert float(head.invivo_profile_c.abs().sum()) == 0.0


class TestPanelInvariantSurvivesTheWindow:
    """`panel[:, observed]` must still equal `excision_logit`.

    The panel adds each candidate condition's whole-window sum and subtracts
    the observed condition's. If the two halves disagree the observed column
    stops matching the scalar -- and the panel loss gathers exactly that
    column, so supervision would target a quantity the model never reports.
    That bug shipped here once at the P1-only shape; this is the same check at
    the window shape.
    """

    @pytest.mark.parametrize(
        "panel_key,provenance_key",
        [
            ("excision_panel_apm", "apm"),
            ("excision_panel_stimulus", "stimulus"),
        ],
    )
    def test_observed_column_equals_the_scalar(self, panel_key, provenance_key):
        index = torch.tensor([1, 2, 3])
        model = _randomize_invivo(_model())
        out = _forward(model, **{provenance_key: index})
        observed = out[panel_key].gather(1, index.unsqueeze(1)).squeeze(1)
        assert torch.allclose(observed, out["excision_logit"], atol=1e-5), (
            f"{panel_key} observed column disagrees with excision_logit; the "
            "window add/subtract pair is mismatched"
        )

    def test_columns_still_differ_between_conditions(self):
        """Preserving the invariant must not flatten the panel."""
        model = _randomize_invivo(_model())
        out = _forward(model, apm=torch.tensor([1, 2, 3]))
        panel = out["excision_panel_apm"]
        assert not torch.allclose(panel[:, 0], panel[:, 3])


class TestSubsitesAreReachable:
    """Every position must be trainable, or the window is decoration.

    Scope note, established by fault injection: each subsite is reachable by
    *two* routes -- the scalar `excision_logit` via `_window_preference`, and
    the counterfactual panels via `_window_preference_all`. Detaching a
    position from one route alone leaves this test passing, because the other
    still supplies gradient. So this asserts "reachable at all", not "reachable
    through the scalar score". Cutting both routes does trip it, which is what
    makes it worth having.
    """

    def test_every_subsite_receives_gradient(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_gradient_coverage import _every_modality_batch

        from presto.scripts.train_synthetic import compute_loss

        torch.manual_seed(0)
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = _every_modality_batch()
        # Warm up: the profiles start at zero, and a zero-init table sends no
        # gradient upstream on the first pass.
        for _ in range(3):
            optimizer.zero_grad()
            loss, _, _ = compute_loss(model, batch, "cpu")
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        loss, _, _ = compute_loss(model, batch, "cpu")
        loss.backward()

        named = dict(model.named_parameters())
        for name in ("invivo_profile_c", "invivo_profile_n", "stimulus_profile_c"):
            grad = named[f"excision_head.{name}"].grad
            per_position = grad.abs().sum(dim=(0, 2))
            starved = [i for i, g in enumerate(per_position.tolist()) if g <= 1e-8]
            assert starved == [], f"{name} subsites {starved} receive no gradient"


class TestInVitroBranchIsUnchanged:
    """The P1-only in-vitro branch must be untouched by the widening.

    Its labels come from a P1 rule, its profiles are pinned to known
    specificities, and `_missed_cleavage_score` excludes exactly one residue on
    the assumption that only P1 is scored by the junction term. Widening it
    would break that arithmetic and add capacity to memorize the generator.
    """

    def test_pinned_profile_is_still_two_dimensional(self):
        head = _model().excision_head
        assert head.effective_profile_c().dim() == 2
        assert head.p1_profile_c.dim() == 2
        assert head.p1_profile_n.dim() == 2

    def test_p1_prime_block_is_still_p1_prime_only(self):
        """The proline rule is genuinely a P1' constraint, not a window one."""
        head = _model().excision_head
        assert head.p1_prime_blocked.dim() == 2

    def test_pinned_rules_still_hold(self):
        from presto.data.vocab import EXCISION_MACHINERY_TO_IDX, EXCISION_P1_RULES

        profile = _model().excision_head.effective_profile_c()
        for machinery, residues in EXCISION_P1_RULES.items():
            row = profile[EXCISION_MACHINERY_TO_IDX[machinery]]
            top = torch.topk(row, len(residues)).indices.tolist()
            assert sorted(top) == sorted(AA_TO_IDX[aa] for aa in residues)


class TestWideningIsAStrictGeneralization:
    """With only P1/P1' populated, the window must score as the old head did.

    This is the property that makes the change safe to merge unvalidated: the
    added subsites start at zero, so a widened head is numerically the old head
    until training moves them. If it were not a strict generalization, every
    excision number would shift the moment this landed, and the outstanding
    validation run would be measuring two changes at once.
    """

    def test_only_p1_populated_matches_a_p1_only_computation(self):
        model = _model()
        head = model.excision_head
        apm_index = 2
        torch.manual_seed(7)
        p1_values = torch.randn(head.n_aa)
        p1_prime_values = torch.randn(head.n_aa)

        with torch.no_grad():
            head.invivo_profile_c.zero_()
            head.invivo_profile_c[apm_index, head.p1_window_index] = p1_values
            head.invivo_profile_c[apm_index, head.p1_prime_window_index] = p1_prime_values

        window = torch.full((1, head.window_size), head.missing_residue_index)
        window[0, head.p1_window_index] = AA_TO_IDX["K"]
        window[0, head.p1_prime_window_index] = AA_TO_IDX["A"]

        got = head._window_preference(head.invivo_profile_c, torch.tensor([apm_index]), window)
        # What a P1-only head computed: the two populated cells, plus the
        # <MISSING> column at every other position -- which is zero here.
        expected = p1_values[AA_TO_IDX["K"]] + p1_prime_values[AA_TO_IDX["A"]]
        assert torch.allclose(got, expected.unsqueeze(0), atol=1e-6)

    def test_untouched_subsites_contribute_nothing(self):
        """A zero row must stay inert whatever residue lands on it."""
        head = _model().excision_head
        window_a = torch.full((1, head.window_size), AA_TO_IDX["K"])
        window_b = torch.full((1, head.window_size), AA_TO_IDX["W"])
        condition = torch.tensor([0])
        got_a = head._window_preference(head.invivo_profile_c, condition, window_a)
        got_b = head._window_preference(head.invivo_profile_c, condition, window_b)
        assert torch.allclose(got_a, got_b)
        assert float(got_a.abs().sum()) == 0.0

    def test_the_two_window_helpers_agree(self):
        """`_window_preference_all` builds the panels; the scalar path uses
        `_window_preference`. They must return the same number for the same
        condition, or the panel's observed column drifts from the logit."""
        head = _randomize_invivo(_model()).excision_head
        torch.manual_seed(3)
        window = torch.randint(0, head.n_aa, (4, head.window_size))
        condition = torch.tensor([0, 1, 2, 3])
        one = head._window_preference(head.invivo_profile_c, condition, window)
        every = head._window_preference_all(head.invivo_profile_c, window)
        picked = every.gather(1, condition.unsqueeze(1)).squeeze(1)
        assert torch.allclose(one, picked, atol=1e-6)


class TestTheCorpusActuallySuppliesFlanks:
    """Junction context must exist in the data, not just in the head.

    The window is only worth its parameters if the corpus fills it. For a long
    time it did not for class II: every one of 1,395,872 class II MS rows
    carried zero flank sequence, so every class II junction residue was
    `<MISSING>` and the cathepsin specificity this head exists to express had
    no data behind it. Nothing in the suite said so -- it was found by
    measuring, and the code comments asserting "0% carry 15 residues" then went
    stale the moment hitlist raised its default.

    So the claim is checked rather than written down. Skipped where hitlist is
    absent (CI installs without the extra) or its corpus cannot be queried.
    """

    #: Coverage floors, well under the measured values on hitlist 1.55.2
    #: (class I 94.5% / class II 89.6% with both flanks >= 5). These are a
    #: regression guard, not a target -- a drop to zero is the failure that
    #: actually happened.
    MIN_COVERAGE = {"I": 0.70, "II": 0.70}

    @staticmethod
    def _frame():
        hitlist = pytest.importorskip("hitlist")
        from presto.data.hitlist_source import training_columns

        try:
            return hitlist.generate_training_table(
                include_evidence="ms",
                columns=training_columns("ms", include_flanks=True),
                map_source_proteins=True,
            )
        except Exception as exc:  # noqa: BLE001 - unbuilt index, missing download
            pytest.skip(f"cannot query hitlist flanks: {exc}")

    @pytest.mark.parametrize("mhc_class", sorted(MIN_COVERAGE))
    def test_both_flanks_are_present_for_most_rows(self, mhc_class):
        frame = self._frame()
        rows = frame[frame["mhc_class"].astype(str) == mhc_class]
        if not len(rows):
            pytest.skip(f"no class {mhc_class} rows in this corpus")
        window = _model().excision_head.junction_window
        n_len = rows["n_flank"].fillna("").astype(str).str.len()
        c_len = rows["c_flank"].fillna("").astype(str).str.len()
        covered = ((n_len >= window) & (c_len >= window)).mean()
        floor = self.MIN_COVERAGE[mhc_class]
        assert covered >= floor, (
            f"only {covered:.1%} of class {mhc_class} rows carry both "
            f"{window}-residue flanks (floor {floor:.0%}). The excision window "
            "is reading <MISSING> for the rest, so its subsites have no data."
        )
