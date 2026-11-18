"""Scientific matching units, strong-tail control and multi-allelic confounding."""

import math
import pytest

from presto.data.affinity_matching import AffinityScore as S, MatchingPolicy as P, assess_match


def test_log_nm_caliper_is_a_ratio_not_a_rank_interval():
    p = P(metric="log10_nm", max_gap=0.1, profile="winner", max_log_rank_gap=None)
    a = {"I:A": S(100, 0.1)}
    assert assess_match(a, {"I:A": S(100 * 10**0.1, 9)}, p).accepted
    assert not assess_match(a, {"I:A": S(127, 0.1)}, p).accepted
    assert math.isclose(10**0.1, 1.2589254117941673)


def test_absolute_rank_points_and_relative_tail_guard():
    a, b = {"I:A": S(10, 0.01)}, {"I:A": S(100, 0.51)}
    assert assess_match(a, b, P(max_log_rank_gap=None)).accepted
    assert assess_match(a, b, P()).reason == "relative_rank_caliper"
    assert assess_match({"II:DR": S(200, 2)}, {"II:DR": S(250, 2.4)}, P()).accepted


def test_full_candidate_profile_catches_hidden_alternative_allele():
    a = {"I:A": S(50, 0.2), "I:B": S(500, 2)}
    b = {"I:A": S(51, 0.21), "I:B": S(10000, 30)}
    assert assess_match(a, b, P(profile="winner")).accepted
    result = assess_match(a, b, P(profile="all"))
    assert not result.accepted and result.max_rank_gap == 28


def test_same_best_rank_without_same_allele_is_not_a_match():
    a = {"I:A": S(50, 0.2), "I:B": S(500, 2)}
    b = {"I:A": S(500, 2), "I:B": S(50, 0.2)}
    assert assess_match(a, b, P()).reason == "different_winning_candidate"


def test_missing_candidate_scores_are_rejected_not_silently_intersected():
    with pytest.raises(ValueError, match="identical nonempty"):
        assess_match({"I:A": S(10, 1), "I:B": S(20, 2)}, {"I:A": S(10, 1)}, P())


def test_zero_printed_ranks_remain_finite_and_explicit():
    result = assess_match({"II:DP": S(1, 0)}, {"II:DP": S(1, 0)}, P())
    assert result.accepted and result.max_log_rank_gap == 0
    assert not assess_match({"II:DP": S(1, 0)}, {"II:DP": S(2, 0.1)}, P()).accepted


@pytest.mark.parametrize("score", [S(0, 1), S(float('nan'), 1), S(1, -1), S(1, 101)])
def test_invalid_affinity_values_fail(score):
    with pytest.raises(ValueError, match="BA"):
        assess_match({"I:A": score}, {"I:A": S(1, 1)}, P())


def test_el_rank_is_not_an_affinity_matching_option():
    with pytest.raises(ValueError, match="EL is excluded"):
        assess_match({"I:A": S(1, 1)}, {"I:A": S(1, 1)}, P(metric="el_rank"))
