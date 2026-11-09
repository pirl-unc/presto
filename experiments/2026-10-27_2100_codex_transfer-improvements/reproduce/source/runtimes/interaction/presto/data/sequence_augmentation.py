"""Training-time corruption that mirrors what inference actually looks like.

Two independent augmentations, both about making the model robust to context it
will not always have.

**Flank dropout.** A peptide's source protein is often unknown at inference --
a caller with a peptide and an allele and nothing else is the common case, and
even in the corpus 0.70% of rows were never mapped. A model trained only on
rows with flanks learns to lean on them, then degrades unpredictably when they
are absent. Replacing both flanks with `?` on a subset of each batch forces it
to work either way, and makes "no flank" a state it has seen rather than a
distribution shift.

**Residue dropout.** Individual residues go unresolved in real proteomes; `X`
is the code for it. Corrupting scattered residues to `X` during training gives
that token a meaning grounded in context rather than leaving it to whatever the
45,992 initiator-residue rows happen to teach.

Both are applied to the *sequence strings*, before tokenization, so they cost
nothing at the tensor level and are visible in any dump of the batch.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from .vocab import ENCODABLE_RESIDUES

#: What an undetermined flank looks like on a sample: absent. The model
#: supplies the unknown-context embedding from the tensor side, padding the
#: window with `?` (`Presto._pad_for_side`), so no sentinel belongs in the
#: sequence string itself.
UNKNOWN_FLANK = None

#: What an unresolved residue looks like.
UNKNOWN_RESIDUE = "X"


@dataclass(frozen=True)
class AugmentationConfig:
    """How much to corrupt. All rates are per-row or per-residue probabilities."""

    #: Fraction of rows whose flanks are both replaced by `?`.
    flank_dropout_rate: float = 0.0
    #: Per-residue probability of becoming `X`, applied inside flanks.
    flank_residue_dropout_rate: float = 0.0
    #: Per-residue probability of becoming `X`, applied inside the peptide.
    peptide_residue_dropout_rate: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "flank_dropout_rate",
            "flank_residue_dropout_rate",
            "peptide_residue_dropout_rate",
        ):
            value = getattr(self, name)
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]; got {value!r}")

    @property
    def is_active(self) -> bool:
        return any(
            getattr(self, name) > 0.0
            for name in (
                "flank_dropout_rate",
                "flank_residue_dropout_rate",
                "peptide_residue_dropout_rate",
            )
        )


def corrupt_residues(sequence: str, rate: float, rng: random.Random) -> str:
    """Replace each residue with `X` independently with probability ``rate``.

    Only real residues are corrupted. A flank marker (`^`, `$`, `?`) already
    describes an absence, and turning one into "unresolved residue" would
    assert a residue exists where the point is that none does.
    """
    if rate <= 0.0 or not sequence:
        return sequence
    return "".join(
        UNKNOWN_RESIDUE if (character in ENCODABLE_RESIDUES and rng.random() < rate) else character
        for character in sequence
    )


def augment_sample_sequences(
    sample, config: AugmentationConfig, rng: Optional[random.Random] = None
) -> None:
    """Corrupt one sample's sequences in place.

    In place because the collator holds samples briefly and copying a record
    per batch is pure overhead; callers that need the original should copy
    first. Applied per-sample rather than per-batch so two rows in a batch see
    different corruption, which is the point.
    """
    if not config.is_active:
        return
    rng = rng or random

    if config.flank_dropout_rate > 0.0 and rng.random() < config.flank_dropout_rate:
        # Both sides together. Dropping one and keeping the other would teach
        # a correlation between the two that does not exist -- a peptide whose
        # source protein is unknown has neither flank, not one.
        sample.flank_n = UNKNOWN_FLANK
        sample.flank_c = UNKNOWN_FLANK
        # The terminus claim came from the mapping, and the mapping is what we
        # are pretending not to have.
        if hasattr(sample, "flank_n_is_terminus"):
            sample.flank_n_is_terminus = False
        if hasattr(sample, "flank_c_is_terminus"):
            sample.flank_c_is_terminus = False
        return

    if config.flank_residue_dropout_rate > 0.0:
        sample.flank_n = corrupt_residues(
            sample.flank_n or "", config.flank_residue_dropout_rate, rng
        )
        sample.flank_c = corrupt_residues(
            sample.flank_c or "", config.flank_residue_dropout_rate, rng
        )

    if config.peptide_residue_dropout_rate > 0.0:
        sample.peptide = corrupt_residues(
            sample.peptide or "", config.peptide_residue_dropout_rate, rng
        )
