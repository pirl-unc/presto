"""Source-relative ordered context without assumed numerical spacing."""

from dataclasses import dataclass, field
from typing import Any


def _label(value):
    return isinstance(value, str) and bool(value.strip()) and value != "unknown"


@dataclass(frozen=True)
class OrdinalScale:
    """An ordered vocabulary anchored to an explicit comparison/calibration frame."""

    levels: tuple[str, ...]
    reference: str

    def __post_init__(self):
        if (
            not isinstance(self.levels, tuple)
            or len(self.levels) < 2
            or any(not _label(level) for level in self.levels)
            or len(set(self.levels)) != len(self.levels)
        ):
            raise ValueError("ordinal scale needs at least two distinct ordered level labels")
        if not isinstance(self.reference, str) or not self.reference.strip():
            raise ValueError("ordinal scale requires an explicit reference")

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if not isinstance(value["levels"], (list, tuple)):
            raise ValueError("ordinal scale levels must be an ordered sequence of labels")
        value["levels"] = tuple(value["levels"])
        return cls(**value)


@dataclass(frozen=True)
class OrdinalValue:
    """A reported level; None is unknown, not the lowest category or absence."""

    level: str | None = None
    scale: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.level is None:
            if self.scale is not None:
                raise ValueError("unknown ordinal value cannot declare a known scale")
        elif not _label(self.level) or not isinstance(self.scale, str) or not self.scale.strip():
            raise ValueError("known ordinal value requires a level and scale reference")
        if not isinstance(self.evidence, dict):
            raise ValueError("ordinal evidence must be a mapping")
