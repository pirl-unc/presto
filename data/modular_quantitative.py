"""Explicit measurement scales; no cross-study normalization is inferred."""

from dataclasses import dataclass
import math
import re


QUANTITY_PARTS = {
    "bulk_quantity": ("bulk", ("digest", "ms_response")),
    "elution_quantity": ("cellular", ("cellular", "ms_response")),
    "ms_quantity": ("detector", ("ms_response",)),
    "digest_quantity": ("digest", ("digest",)),
    "cellular_quantity": ("cellular", ("cellular",)),
}


def quantity_domain(endpoint: str) -> str | None:
    if ":" not in endpoint:
        return None
    family, domain = endpoint.split(":", 1)
    if family not in QUANTITY_PARTS or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", domain):
        raise ValueError("unsupported quantity endpoint or domain name")
    return domain


@dataclass(frozen=True)
class QuantitativeDomain:
    quantity: str
    unit: str
    transform: str
    normalization: str
    reference: str

    def __post_init__(self):
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.quantity, self.unit, self.normalization, self.reference)
        ):
            raise ValueError("quantity domain requires meaning, unit, normalization and reference")
        if self.transform not in ("identity", "log2", "log10"):
            raise ValueError("unsupported quantitative target transform")

    @property
    def target_space(self):
        return self.unit if self.transform == "identity" else f"{self.transform}({self.unit})"

    def convert(self, value: float, unit: str | None, encoding: str = "linear") -> float:
        if unit != self.unit:
            raise ValueError("quantitative measurement unit mismatch")
        if encoding not in ("linear", "log2", "log10") or not math.isfinite(value):
            raise ValueError("invalid quantitative value or encoding")
        if encoding == "linear":
            if self.transform == "identity":
                return float(value)
            if value <= 0:
                raise ValueError("log quantity target requires a positive measured value/bound")
            return math.log2(value) if self.transform == "log2" else math.log10(value)
        if self.transform == "identity":
            try:
                result = (2.0 if encoding == "log2" else 10.0) ** value
            except OverflowError as error:
                raise ValueError("quantity conversion exceeds floating-point range") from error
            if not math.isfinite(result) or result == 0:
                raise ValueError("quantity conversion exceeds floating-point range")
            return result
        factor = math.log(2) if encoding == "log2" else math.log(10)
        divisor = math.log(2) if self.transform == "log2" else math.log(10)
        result = value * (factor / divisor)
        if not math.isfinite(result):
            raise ValueError("nonfinite transformed quantity")
        return result

    def inverse(self, value: float) -> float | None:
        try:
            result = (
                value
                if self.transform == "identity"
                else (2.0 if self.transform == "log2" else 10.0) ** value
            )
        except OverflowError:
            return None
        return (
            result
            if math.isfinite(result) and (result != 0 or self.transform == "identity")
            else None
        )
