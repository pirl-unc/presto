"""Known source conflicts cannot silently re-enter any purified-assay preparation."""

from importlib.resources import files
import json
import pytest

from presto.data.assay_reviews import assay_review_reason
from presto.data.modular_hitlist import AdmissionError, purified_assay_observation


REVIEWS = json.loads(files("presto.data").joinpath("assay_measurement_reviews.json").read_text())


@pytest.mark.parametrize("entry", REVIEWS["entries"], ids=lambda entry: entry["assay_iri"])
@pytest.mark.parametrize("source,scheme", [("iedb", "http"), ("cedar", "https")])
def test_every_reviewed_identity_is_deferred_without_reinterpreting_raw_labels(
    entry, source, scheme
):
    raw = {
        "source": source,
        "assay_iri": entry["assay_iri"].replace("http:", scheme + ":"),
        "response_measured": entry["observed_export"]["response_measured"],
        "quantitative_value": entry["observed_export"]["value"],
        "measurement_units": entry["observed_export"]["unit"],
        "measurement_inequality": entry["observed_export"]["qualifier"],
    }
    original = dict(raw)
    with pytest.raises(AdmissionError) as caught:
        purified_assay_observation(raw, None, source_sha256="a" * 64, ordinal=0)
    assert caught.value.disposition == "deferred"
    assert caught.value.reason == "source_review:" + entry["reason"]
    assert raw == original


def test_corrected_future_source_requires_explicit_review():
    raw = {
        "source": "iedb",
        "assay_iri": "http://www.iedb.org/assay/1821009",
        "response_measured": "on rate",
        "quantitative_value": 1.0,
        "measurement_units": "1/(M*s)",
    }
    with pytest.raises(AdmissionError, match="source_review:"):
        purified_assay_observation(raw, None, source_sha256="b" * 64, ordinal=0)


@pytest.mark.parametrize(
    "identity",
    [
        None,
        1821009,
        "https://example.org/assay/1821009",
        "http://www.iedb.org/assay/1820986",
        "http://www.iedb.org/assay/1820987",
        "http://www.iedb.org/assay/1820988",
        "http://www.iedb.org/assay/18210090",
    ],
)
def test_other_outcomes_and_neighboring_valid_half_lives_are_unaffected(identity):
    assert assay_review_reason(identity) is None


def test_registry_is_scoped_unique_and_cites_primary_evidence():
    entries = REVIEWS["entries"]
    assert len(entries) == len({entry["assay_iri"] for entry in entries}) == 20
    assert all(
        entry["reference"].startswith(
            ("https://pmc.ncbi.nlm.nih.gov/", "https://pubmed.ncbi.nlm.nih.gov/")
        )
        for entry in entries
    )
