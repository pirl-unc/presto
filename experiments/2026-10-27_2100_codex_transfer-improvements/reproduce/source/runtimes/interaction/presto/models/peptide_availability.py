"""Exact and lazily sampled distributions over variable-length protein spans.

This primitive predicts a product distribution, not calibrated concentration or
physical cleavage trajectories. Measurement selection and MHC binding belong to
downstream models. A state holds an autograd graph and must be rebuilt after each
optimizer update; it is not a reusable cached policy snapshot.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYUO"
_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
_BOUNDARY = len(AMINO_ACIDS)


@dataclass(frozen=True)
class SpanState:
    residues: torch.Tensor
    prefix: torch.Tensor
    context: torch.Tensor
    start_log_probs: torch.Tensor
    valid_length_counts: torch.Tensor
    min_length: int
    max_length: int

    @property
    def protein_length(self):
        return len(self.residues) - 1

    @property
    def valid_span_count(self):
        return int(self.valid_length_counts.sum().item())


@dataclass(frozen=True)
class SpanSamples:
    starts: torch.Tensor  # zero-based, with replacement
    lengths: torch.Tensor
    log_probs: torch.Tensor
    start_network_rows: int
    conditional_network_rows: int
    conditional_valid_spans: int


class PeptideAvailabilityPolicy(nn.Module):
    """p(start, length | parent, context) with exact conditional normalization.

    Lazy draws score all starts, then only the length distributions at sampled
    starts. Repeated draws survive; repeated conditional computation is shared.
    Zero learned scores give a uniform distribution over all valid spans.
    """

    def __init__(self, *, min_length=1, max_length=40, context_size=0, hidden_size=32):
        super().__init__()
        if (
            not isinstance(min_length, int)
            or not isinstance(max_length, int)
            or not 1 <= min_length <= max_length
            or not isinstance(context_size, int)
            or context_size < 0
            or not isinstance(hidden_size, int)
            or hidden_size < 1
        ):
            raise ValueError("invalid length support or policy dimensions")
        self.min_length = min_length
        self.max_length = max_length
        self.context_size = context_size
        self.width = max_length - min_length + 1
        flank_width = len(AMINO_ACIDS) + 1
        self.start_head = self._head(2 * flank_width + 2 + context_size, hidden_size)
        self.length_head = self._head(
            4 * flank_width + len(AMINO_ACIDS) + 3 + context_size, hidden_size
        )

    @staticmethod
    def _head(input_size, hidden_size):
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def prepare(self, sequence: str, context=()) -> SpanState:
        """Encode fixed features and compute current start probabilities once."""
        if not isinstance(sequence, str) or len(sequence) < self.min_length:
            raise ValueError("parent sequence is shorter than the minimum span")
        if any(amino_acid not in _INDEX for amino_acid in sequence):
            raise ValueError("parent must use literal uppercase amino-acid symbols")
        parameter = next(self.parameters())
        context = torch.as_tensor(context, device=parameter.device, dtype=parameter.dtype)
        if context.shape != (self.context_size,) or not torch.isfinite(context).all():
            raise ValueError("context must be a finite vector of the declared size")
        tokens = torch.tensor(
            [*(_INDEX[amino_acid] for amino_acid in sequence), _BOUNDARY],
            dtype=torch.long,
            device=parameter.device,
        )
        residues = F.one_hot(tokens, num_classes=_BOUNDARY + 1).to(parameter.dtype)
        prefix = torch.cat(
            [residues.new_zeros((1, _BOUNDARY)), residues[:-1, :_BOUNDARY].cumsum(0)]
        )
        size = len(sequence)
        starts = torch.arange(size - self.min_length + 1, device=parameter.device)
        counts = (size - starts).clamp(max=self.max_length) - self.min_length + 1
        previous = torch.where(starts > 0, starts - 1, size)
        features = torch.cat(
            [
                residues[previous],
                residues[starts],
                (starts.to(parameter.dtype) / max(size - 1, 1))[:, None],
                (counts.to(parameter.dtype) / self.width)[:, None],
                context.expand(len(starts), -1),
            ],
            dim=-1,
        )
        logits = self.start_head(features).squeeze(-1) + counts.to(parameter.dtype).log()
        return SpanState(
            residues,
            prefix,
            context,
            logits.log_softmax(0),
            counts,
            self.min_length,
            self.max_length,
        )

    def _check_state(self, state):
        if (
            state.min_length != self.min_length
            or state.max_length != self.max_length
            or state.context.shape != (self.context_size,)
        ):
            raise ValueError("state and policy support/context differ")

    def span_features(self, state: SpanState, starts: torch.Tensor):
        """Return padded conditional features, lengths and a validity mask.

        Invalid boundary-crossing lengths have finite placeholder features; their
        logits are excluded before normalization and they can never be sampled.
        """
        self._check_state(state)
        if (
            not isinstance(starts, torch.Tensor)
            or starts.dtype != torch.long
            or starts.ndim != 1
            or not len(starts)
            or bool((starts < 0).any())
            or bool((starts >= len(state.start_log_probs)).any())
        ):
            raise ValueError("starts must be a nonempty vector of valid int64 positions")
        size = state.protein_length
        lengths = torch.arange(self.min_length, self.max_length + 1, device=starts.device)
        ends = starts[:, None] + lengths[None, :]
        valid = ends <= size
        clipped_ends = ends.clamp(max=size)
        previous = torch.where(starts > 0, starts - 1, size)
        mean = (state.prefix[clipped_ends] - state.prefix[starts, None]) / (
            clipped_ends - starts[:, None]
        ).unsqueeze(-1)
        shape = (len(starts), self.width)
        dtype = state.residues.dtype
        features = torch.cat(
            [
                state.residues[starts, None].expand(*shape, -1),
                state.residues[clipped_ends - 1],
                state.residues[previous, None].expand(*shape, -1),
                state.residues[clipped_ends],
                mean,
                (lengths.to(dtype) / self.max_length)[None, :, None].expand(*shape, 1),
                (starts.to(dtype) / max(size - 1, 1))[:, None, None].expand(*shape, 1),
                (clipped_ends.to(dtype) / size)[:, :, None],
                state.context.expand(*shape, -1),
            ],
            dim=-1,
        )
        return features, lengths, valid

    def conditional_log_probs(self, state: SpanState, starts: torch.Tensor):
        features, lengths, valid = self.span_features(state, starts)
        logits = self.length_head(features).squeeze(-1).masked_fill(~valid, -torch.inf)
        return logits.log_softmax(-1), lengths, valid

    def enumerate(self, state: SpanState, *, chunk_size=256):
        """Yield (starts, lengths, joint log probabilities) for all valid spans.

        Chunking bounds inference working tensors. Retaining all differentiable
        outputs also retains their graphs; training memory is not automatically
        bounded by this iterator's chunk size.
        """
        self._check_state(state)
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        device = state.start_log_probs.device
        for offset in range(0, len(state.start_log_probs), chunk_size):
            starts = torch.arange(
                offset, min(offset + chunk_size, len(state.start_log_probs)), device=device
            )
            conditional, lengths, valid = self.conditional_log_probs(state, starts)
            joint = state.start_log_probs[starts, None] + conditional
            yield (
                starts[:, None].expand_as(valid)[valid],
                lengths[None, :].expand_as(valid)[valid],
                joint[valid],
            )

    def log_prob(self, state: SpanState, starts, lengths):
        """Exact selected log probabilities with shared repeated-start work."""
        self._check_state(state)
        device = state.start_log_probs.device
        starts = torch.as_tensor(starts, device=device)
        lengths = torch.as_tensor(lengths, device=device)
        if (
            starts.dtype != torch.long
            or lengths.dtype != torch.long
            or starts.ndim != 1
            or starts.shape != lengths.shape
            or not len(starts)
            or bool((starts < 0).any())
            or bool((starts >= len(state.start_log_probs)).any())
            or bool((lengths < self.min_length).any())
            or bool((lengths > self.max_length).any())
            or bool((starts + lengths > state.protein_length).any())
        ):
            raise ValueError("selected spans must be valid nonempty integer vectors")
        unique, inverse = starts.unique(sorted=True, return_inverse=True)
        conditional, _, _ = self.conditional_log_probs(state, unique)
        return state.start_log_probs[starts] + conditional[inverse, lengths - self.min_length]

    def sample(self, state: SpanState, count: int, *, generator=None) -> SpanSamples:
        """Sample with replacement using O(L + min(K,L)W) scored rows."""
        self._check_state(state)
        if not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        starts = torch.multinomial(
            state.start_log_probs.detach().exp(), count, replacement=True, generator=generator
        )
        unique, inverse = starts.unique(sorted=True, return_inverse=True)
        conditional, lengths, valid = self.conditional_log_probs(state, unique)
        indices = torch.multinomial(
            conditional.detach().exp()[inverse], 1, generator=generator
        ).squeeze(-1)
        return SpanSamples(
            starts,
            lengths[indices],
            state.start_log_probs[starts] + conditional[inverse, indices],
            start_network_rows=len(state.start_log_probs),
            conditional_network_rows=len(unique) * self.width,
            conditional_valid_spans=int(valid.sum().item()),
        )
