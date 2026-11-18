"""Primary-source conflicts keyed by measured-outcome identity, never by magnitude."""

from functools import lru_cache
from importlib.resources import files
import json
import re


@lru_cache(maxsize=1)
def _decisions():
    document = json.loads(
        files("presto.data").joinpath("assay_measurement_reviews.json").read_text()
    )
    if document["schema"] != "presto.assay-review.v1":
        raise ValueError("unsupported assay review registry")
    result = {}
    for entry in document["entries"]:
        identity = entry["assay_iri"]
        if identity in result or entry["status"] != "deferred":
            raise ValueError("invalid or duplicate assay review decision")
        result[identity] = entry["reason"]
    return result


def assay_review_reason(assay_iri):
    """Return a deferral reason, retaining future changed labels for explicit review."""
    if not isinstance(assay_iri, str):
        return None
    match = re.fullmatch(r"https?://www\.iedb\.org/assay/([0-9]+)", assay_iri.strip())
    if match is None:
        return None
    return _decisions().get("http://www.iedb.org/assay/" + match[1])
