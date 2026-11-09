"""Distribution and gradient contracts independent of biological supervision."""

import pytest
import torch

from presto.models.peptide_availability import AMINO_ACIDS, PeptideAvailabilityPolicy


def enumerate_all(policy, state, chunk_size):
    chunks = list(policy.enumerate(state, chunk_size=chunk_size))
    return tuple(torch.cat([chunk[index] for chunk in chunks]) for index in range(3))


@pytest.mark.parametrize("minimum,maximum", [(1, 1), (1, 3), (2, 7), (4, 4)])
def test_uniform_span_measure_and_exhaustive_support(minimum, maximum):
    policy = PeptideAvailabilityPolicy(min_length=minimum, max_length=maximum).double()
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
    state = policy.prepare("ACDE")
    starts, lengths, logp = enumerate_all(policy, state, 2)
    expected = [(s, n) for s in range(4) for n in range(minimum, maximum + 1) if s + n <= 4]
    assert list(zip(starts.tolist(), lengths.tolist())) == expected
    assert state.valid_span_count == len(expected)
    torch.testing.assert_close(logp.exp(), torch.full_like(logp, 1 / len(expected)))
    torch.testing.assert_close(logp.exp().sum(), torch.tensor(1.0, dtype=torch.double))


def test_features_preserve_exact_composition_and_boundary_symbols():
    policy = PeptideAvailabilityPolicy(max_length=4, context_size=1).double()
    state = policy.prepare("ACDE", [0.25])
    features, lengths, valid = policy.span_features(state, torch.tensor([0, 2, 3]))
    width = len(AMINO_ACIDS) + 1
    assert lengths.tolist() == [1, 2, 3, 4]
    assert valid.tolist() == [[True] * 4, [True, True, False, False], [True, False, False, False]]
    for row, start in enumerate([0, 2, 3]):
        for column, length in enumerate(lengths.tolist()):
            if not valid[row, column]:
                continue
            piece = "ACDE"[start : start + length]
            expected_residues = [
                piece[0],
                piece[-1],
                "ACDE"[start - 1] if start else "boundary",
                "ACDE"[start + length] if start + length < 4 else "boundary",
            ]
            for offset, amino_acid in enumerate(expected_residues):
                index = (
                    len(AMINO_ACIDS) if amino_acid == "boundary" else AMINO_ACIDS.index(amino_acid)
                )
                vector = features[row, column, offset * width : (offset + 1) * width]
                assert vector.argmax().item() == index
                assert vector.sum().item() == 1
            expected_mean = torch.tensor(
                [piece.count(amino_acid) / length for amino_acid in AMINO_ACIDS],
                dtype=torch.double,
            )
            torch.testing.assert_close(
                features[row, column, 4 * width : 4 * width + 22], expected_mean
            )
            assert features[row, column, -1].item() == 0.25


def test_lazy_selected_probabilities_and_gradients_equal_dense():
    torch.manual_seed(17)
    policy = PeptideAvailabilityPolicy(max_length=5, context_size=2).double()
    sequence = "ACDEFGHIKLMNPQRSTVWYUO"
    state = policy.prepare(sequence, [0.4, -0.2])
    starts, lengths, dense = enumerate_all(policy, state, 100)
    chunked = enumerate_all(policy, state, 3)
    torch.testing.assert_close(chunked[2], dense)
    torch.testing.assert_close(dense.exp().sum(), torch.ones((), dtype=torch.double))
    selected_starts, selected_lengths = torch.tensor([0, 0, 2, 5]), torch.tensor([1, 3, 2, 1])
    indices = [
        ((starts == s) & (lengths == n)).nonzero().item()
        for s, n in zip(selected_starts, selected_lengths)
    ]
    weights = torch.tensor([0.2, -1.3, 2.1, 0.5], dtype=torch.double)
    dense_gradients = torch.autograd.grad(
        (dense[indices] * weights).sum(), tuple(policy.parameters())
    )
    lazy = policy.log_prob(policy.prepare(sequence, [0.4, -0.2]), selected_starts, selected_lengths)
    torch.testing.assert_close(lazy, dense[indices])
    lazy_gradients = torch.autograd.grad((lazy * weights).sum(), tuple(policy.parameters()))
    for reference, candidate in zip(dense_gradients, lazy_gradients):
        assert torch.isfinite(candidate).all() and torch.count_nonzero(candidate) > 0
        torch.testing.assert_close(candidate, reference, atol=1e-12, rtol=1e-10)


def test_sampling_preserves_repeats_and_only_scores_sampled_starts():
    torch.manual_seed(31)
    policy = PeptideAvailabilityPolicy(max_length=7)
    state = policy.prepare(AMINO_ACIDS * 10)
    work = []

    def count_rows(_module, arguments):
        work.append(arguments[0].numel() // arguments[0].shape[-1])

    handle = policy.length_head[0].register_forward_pre_hook(count_rows)
    try:
        samples = policy.sample(state, 8, generator=torch.Generator().manual_seed(41))
    finally:
        handle.remove()
    assert len(samples.starts) == len(samples.lengths) == len(samples.log_probs) == 8
    assert sum(work) == samples.conditional_network_rows == len(samples.starts.unique()) * 7
    assert sum(work) < len(state.start_log_probs) * 7
    torch.testing.assert_close(
        samples.log_probs, policy.log_prob(state, samples.starts, samples.lengths)
    )
    tiny = policy.prepare("A")
    repeated = policy.sample(tiny, 20, generator=torch.Generator().manual_seed(41))
    assert repeated.starts.tolist() == [0] * 20
    assert repeated.lengths.tolist() == [1] * 20
    assert repeated.conditional_network_rows == 7


def test_sample_frequencies_agree_with_exact_small_distribution():
    torch.manual_seed(5)
    policy = PeptideAvailabilityPolicy(max_length=3)
    state = policy.prepare("ACDE")
    starts, lengths, logp = enumerate_all(policy, state, 2)
    samples = policy.sample(state, 15000, generator=torch.Generator().manual_seed(7))
    observed = torch.stack(
        [
            ((samples.starts == s) & (samples.lengths == n)).float().mean()
            for s, n in zip(starts, lengths)
        ]
    )
    torch.testing.assert_close(observed, logp.exp(), atol=0.012, rtol=0)


@pytest.mark.parametrize(
    "sequence,context",
    [("", [0.0]), ("ac", [0.0]), ("AX", [0.0]), ("AC", []), ("AC", [float("nan")])],
)
def test_invalid_inputs_are_rejected(sequence, context):
    policy = PeptideAvailabilityPolicy(context_size=1)
    with pytest.raises(ValueError, match="parent|context"):
        policy.prepare(sequence, context)


def test_invalid_support_selections_and_sampling_are_rejected():
    with pytest.raises(ValueError, match="invalid length support"):
        PeptideAvailabilityPolicy(min_length=0)
    policy = PeptideAvailabilityPolicy(min_length=2, max_length=3)
    with pytest.raises(ValueError, match="shorter than"):
        policy.prepare("A")
    state = policy.prepare("ACDE")
    for starts, lengths in [([-1], [2]), ([3], [2]), ([0], [4]), ([0.0], [2]), ([], [])]:
        with pytest.raises(ValueError, match="selected spans"):
            policy.log_prob(state, starts, lengths)
    with pytest.raises(ValueError, match="count must"):
        policy.sample(state, 0)
    with pytest.raises(ValueError, match="chunk_size"):
        list(policy.enumerate(state, chunk_size=0))
    with pytest.raises(ValueError, match="valid int64"):
        policy.conditional_log_probs(state, torch.tensor([-1]))
