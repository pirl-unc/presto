"""Typed output descriptors for fixed, assay-aware T-cell response tracks."""

from __future__ import annotations

from dataclasses import dataclass
import re


AXES = ("family", "response", "method")
INTENSITY_LEVELS = ("low", "intermediate", "high")
_TOKEN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


@dataclass(frozen=True)
class TCellTrack:
    """One declared counterfactual output, never an encoder input."""

    family: str
    response: str
    method: str

    def __post_init__(self):
        if any(not isinstance(value, str) or not _TOKEN.fullmatch(value) for value in self.values):
            raise ValueError("T-cell track axes require normalized lowercase tokens")

    @property
    def values(self) -> tuple[str, str, str]:
        return self.family, self.response, self.method

    @property
    def key(self) -> str:
        return "/".join(("tcell", *self.values))
