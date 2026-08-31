"""Tests for the machinery-conditioned excision head.

Scope note: these run with ``pin_profiles=True`` (the default), so the tests in
``TestPinnedCleavageRules`` assert that the constraint is wired correctly and
reaches the junction score — they are not evidence that the model *learned* the
rules. The genuine known-answer calibration is ``pin_profiles=False`` trained on
a single protease arm, which belongs in an experiment rather than a unit test.
"""

import pytest
import torch

from presto.data.vocab import (
    AA_TO_IDX,
    EXCISION_MACHINERY_TO_IDX,
    EXCISION_P1_RULES,
    PROTEASOME_MIXTURE_COMPONENTS,
)
from presto.models.presto import Presto

REAL_AAS = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture(scope="module")
def model():
    net = Presto(d_model=32, n_layers=2, n_heads=4)
    net.eval()
    return net


class TestPinnedCleavageRules:
    @pytest.mark.parametrize("enzyme,rule", sorted(EXCISION_P1_RULES.items()))
    def test_enzyme_prefers_exactly_its_p1_residues(self, model, enzyme, rule):
        profile = model.excision_head.effective_profile_c().detach()
        row = profile[EXCISION_MACHINERY_TO_IDX[enzyme]]
        scores = {aa: row[AA_TO_IDX[aa]].item() for aa in REAL_AAS}
        top = sorted(scores, key=scores.get, reverse=True)[: len(rule)]
        assert set(top) == set(rule), f"{enzyme}: expected {sorted(rule)}, got {sorted(top)}"

    def test_proteasome_is_a_convex_mixture_of_the_invitro_analogs(self, model):
        weights = model.excision_head.mixture_weights().detach()
        assert weights.numel() == len(PROTEASOME_MIXTURE_COMPONENTS)
        assert weights.sum().item() == pytest.approx(1.0)
        assert (weights >= 0).all()

    def test_proteasome_covers_all_three_catalytic_specificities(self, model):
        """beta1 caspase-like (D/E), beta2 trypsin-like (K/R), beta5
        chymotrypsin-like (F/W/Y/L/M) — the mixture should span all three."""
        profile = model.excision_head.effective_profile_c().detach()
        row = profile[EXCISION_MACHINERY_TO_IDX["proteasome"]]
        scores = {aa: row[AA_TO_IDX[aa]].item() for aa in REAL_AAS}
        top = set(sorted(scores, key=scores.get, reverse=True)[:8])
        assert top & set("DE"), "no beta1 caspase-like preference"
        assert top & set("KR"), "no beta2 trypsin-like preference"
        assert top & set("FWYLM"), "no beta5 chymotrypsin-like preference"


class TestJunctionScoring:
    """c_terminus_score must respond to the residue at the C-terminal junction."""

    def _score(self, model, peptide, machinery, cflank="AAAAA"):
        def encode(seq):
            return torch.tensor([[AA_TO_IDX[c] for c in seq]], dtype=torch.long)

        with torch.no_grad():
            out = model(
                pep_tok=encode(peptide),
                mhc_a_tok=torch.randint(4, 24, (1, 40), generator=torch.Generator().manual_seed(0)),
                mhc_b_tok=torch.randint(4, 24, (1, 40), generator=torch.Generator().manual_seed(1)),
                flank_n_tok=encode("GGGGG"),
                flank_c_tok=encode(cflank),
                mhc_class="I",
                machinery=[machinery],
            )
        return out

    def test_trypsin_scores_k_terminus_above_a_terminus(self, model):
        with_k = self._score(model, "SIINFEKLK", "trypsin")["excision_c_terminus_score"].item()
        with_a = self._score(model, "SIINFEKLA", "trypsin")["excision_c_terminus_score"].item()
        assert with_k > with_a

    def test_gluc_scores_e_terminus_above_k_terminus(self, model):
        with_e = self._score(model, "SIINFEKLE", "gluc")["excision_c_terminus_score"].item()
        with_k = self._score(model, "SIINFEKLK", "gluc")["excision_c_terminus_score"].item()
        assert with_e > with_k

    def test_machinery_changes_the_ranking(self, model):
        """The same peptide is a good trypsin product and a poor GluC one."""
        k_trypsin = self._score(model, "SIINFEKLK", "trypsin")["excision_c_terminus_score"].item()
        k_gluc = self._score(model, "SIINFEKLK", "gluc")["excision_c_terminus_score"].item()
        assert k_trypsin > k_gluc

    def test_proline_at_p1_prime_penalizes_trypsin(self, model):
        """"not before P" — a proline immediately after the cut blocks it."""
        blocked = self._score(model, "SIINFEKLK", "trypsin", cflank="PAAAA")
        allowed = self._score(model, "SIINFEKLK", "trypsin", cflank="AAAAA")
        assert blocked["excision_c_terminus_score"].item() < allowed["excision_c_terminus_score"].item()

    def test_lysc_allows_proline_after_lysine(self, model):
        """LysC's MaxQuant spec explicitly permits K-P, unlike trypsin.

        Changing the C-flank also perturbs the processing latent, so ``c_terminus_score``
        cannot be exactly equal — what must be absent is the *penalty*. Assert
        the declared rule, then that LysC's shift is a small fraction of the
        penalty trypsin takes on the same substitution.
        """
        head = model.excision_head
        lysc = EXCISION_MACHINERY_TO_IDX["lysc"]
        assert head.p1_prime_blocked[lysc, AA_TO_IDX["P"]].item() == 0.0

        def shift(machinery):
            blocked = self._score(model, "SIINFEKLK", machinery, cflank="PAAAA")
            allowed = self._score(model, "SIINFEKLK", machinery, cflank="AAAAA")
            return allowed["excision_c_terminus_score"].item() - blocked["excision_c_terminus_score"].item()

        assert shift("lysc") < 0.1 * shift("trypsin")


class TestOutputContract:
    @pytest.mark.parametrize("topology", ["collapsed", "expanded"])
    def test_excision_outputs_present_in_both_topologies(self, topology):
        net = Presto(d_model=32, n_layers=2, n_heads=4, latent_topology=topology)
        net.eval()
        with torch.no_grad():
            out = net(
                pep_tok=torch.randint(4, 24, (3, 11)),
                mhc_a_tok=torch.randint(4, 24, (3, 40)),
                mhc_b_tok=torch.randint(4, 24, (3, 40)),
                flank_n_tok=torch.randint(4, 24, (3, 10)),
                flank_c_tok=torch.randint(4, 24, (3, 10)),
                mhc_class="I",
            )
        for key in (
            "excision_logit",
            "excision_prob",
            "excision_n_terminus_score",
            "excision_c_terminus_score",
            "excision_length_score",
        ):
            assert key in out, f"{topology} missing {key}"
            assert out[key].shape == (3,)

    def test_machinery_defaults_to_the_in_vivo_pathway_by_class(self):
        net = Presto(d_model=32, n_layers=2, n_heads=4)
        net.eval()
        args = dict(
            pep_tok=torch.randint(4, 24, (2, 11)),
            mhc_a_tok=torch.randint(4, 24, (2, 40)),
            mhc_b_tok=torch.randint(4, 24, (2, 40)),
        )
        with torch.no_grad():
            class1 = net(mhc_class="I", **args)["excision_machinery_idx"]
            class2 = net(mhc_class="II", **args)["excision_machinery_idx"]
        assert (class1 == EXCISION_MACHINERY_TO_IDX["proteasome"]).all()
        assert (class2 == EXCISION_MACHINERY_TO_IDX["cathepsin"]).all()


class TestInternalCleavageSites:
    """Missed-cleavage constraint: both termini matching is not sufficient.

    A peptide can match an enzyme's rule at both ends and still be an
    implausible product if it carries internal sites the enzyme would also have
    cut. In the observed corpus 59.4% of tryptic peptides carry zero internal
    K/R, 29.7% one, 8.6% two.
    """

    def _score(self, model, peptide, machinery):
        def encode(seq):
            return torch.tensor([[AA_TO_IDX[c] for c in seq]], dtype=torch.long)

        with torch.no_grad():
            return model(
                pep_tok=encode(peptide),
                mhc_a_tok=torch.randint(4, 24, (1, 40), generator=torch.Generator().manual_seed(0)),
                mhc_b_tok=torch.randint(4, 24, (1, 40), generator=torch.Generator().manual_seed(1)),
                flank_n_tok=encode("GGGGG"),
                flank_c_tok=encode("AAAAA"),
                mhc_class="I",
                machinery=[machinery],
            )

    def test_penalty_grows_with_internal_site_count(self, model):
        scores = [
            self._score(model, peptide, "trypsin")["excision_missed_cleavage_score"].item()
            for peptide in ("SIINFEAAK", "SIINFEKAK", "SIINKEKAK", "SKINKEKAK")
        ]
        assert scores == sorted(scores, reverse=True), scores
        assert scores[0] == pytest.approx(0.0)
        assert scores[-1] < scores[0]

    def test_c_terminal_residue_is_not_counted_as_internal(self, model):
        """That junction is c_terminus_score's job; counting it would penalize every product."""
        score = self._score(model, "SIINFEAAK", "trypsin")["excision_missed_cleavage_score"].item()
        assert score == pytest.approx(0.0)

    def test_penalty_is_machinery_specific(self, model):
        """K is a site for trypsin and LysC, not for GluC.

        The peptide must carry internal K but no internal E/D, or GluC
        legitimately penalizes it too and the contrast says nothing.
        """
        peptide = "SIINKAKAE"
        assert not set(peptide[:-1]) & set("ED"), "interior must hold no GluC site"
        trypsin = self._score(model, peptide, "trypsin")["excision_missed_cleavage_score"].item()
        gluc = self._score(model, peptide, "gluc")["excision_missed_cleavage_score"].item()
        assert trypsin < 0.0
        assert gluc == pytest.approx(0.0)

    def test_processive_machinery_is_exempt(self, model):
        """The proteasome does not cut at every available site, so an internal
        hydrophobic residue says nothing about whether it produced the peptide."""
        for peptide in ("SIINFEAAK", "SKINKEKAK"):
            score = self._score(model, peptide, "proteasome")["excision_missed_cleavage_score"].item()
            assert score == pytest.approx(0.0), peptide

    def test_internal_term_reaches_the_logit(self, model):
        clean = self._score(model, "SIINFEAAK", "trypsin")["excision_logit"].item()
        missed = self._score(model, "SKINKEKAK", "trypsin")["excision_logit"].item()
        assert missed < clean
