"""Output descriptors, never biological input features or study-ID embeddings."""

from dataclasses import dataclass

from .measurement_curation import assay_method_context


AXES = {
    "preparation": ("purified", "cellular", "lysate", "unknown"),
    "mechanism": ("competitive", "direct", "stabilization", "refolding", "functional", "unknown"),
    "detector": ("radioactivity", "fluorescence", "elisa", "luminescence", "unknown"),
}


@dataclass(frozen=True)
class AssayMethod:
    preparation: str = "unknown"
    mechanism: str = "unknown"
    detector: str = "unknown"
    review_status: str = "unknown"

    def __post_init__(self):
        for axis, vocabulary in AXES.items():
            if getattr(self, axis) not in vocabulary:
                raise ValueError(f"unsupported assay {axis}")
        if self.review_status not in (
            "unknown",
            "source_tag_unreviewed",
            "verified",
            "protocol_citation_conflict",
        ):
            raise ValueError("unsupported method review status")

    @property
    def usable(self):
        return self.review_status in ("source_tag_unreviewed", "verified") and all(
            getattr(self, axis) != "unknown" for axis in AXES
        )


@dataclass(frozen=True)
class AssayTrack:
    endpoint: str
    preparation: str
    mechanism: str
    detector: str

    def __post_init__(self):
        method = AssayMethod(self.preparation, self.mechanism, self.detector, "verified")
        if not method.usable or not self.endpoint:
            raise ValueError("assay tracks require a complete measurement definition")

    @property
    def key(self):
        return "/".join((self.endpoint, self.preparation, self.mechanism, self.detector))


def normalize_assay_method(raw):
    """Conservative projection of source tags, including known citation conflicts."""
    context = assay_method_context(raw)
    prep = {
        "purified MHC": "purified",
        "cellular MHC": "cellular",
        "cellular": "cellular",
        "lysate": "lysate",
    }.get(context["preparation"], "unknown")
    values = {"preparation": prep}
    for axis in ("mechanism", "detector"):
        value = context[axis].lower()
        values[axis] = value if value in AXES[axis] else "unknown"
    return AssayMethod(**values, review_status=context["review_status"])


def observation_method(row):
    """One normalizer for explicit observations and historical source exports."""
    return row.assay_method or normalize_assay_method(row.provenance.get("raw_record", {}))


def observation_track(row):
    method = observation_method(row)
    return (
        AssayTrack(row.endpoint, method.preparation, method.mechanism, method.detector)
        if method.usable
        else None
    )
