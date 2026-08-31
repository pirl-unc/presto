"""Setup failures must not masquerade as ordinary data problems.

`normalize_allele_name` raises ValueError for an allele it cannot parse -- that
is expected data and skipping it is correct. It also raises RuntimeError when
mhcgnomes is unavailable, and a blanket `except Exception` treated the two
identically. The consequence of that conflation is severe and quiet: with
mhcgnomes missing, *every* allele fails normalization, the index degrades to
raw spellings, and nothing anywhere reports a problem.

This is the same shape as the sklearn ImportError that sat behind
`except Exception: auroc = 0.5` and kept CI red for months.
"""

import pytest

torch = pytest.importorskip("torch")


class TestRealUnparseableAllelesAreTolerated:
    """Drive the REAL function with REAL bad input.

    The first version of this test monkeypatched `normalize_allele_name` to
    raise a fabricated ValueError, so it passed while the actual path crashed:
    mhcgnomes signals an unparseable name with its own `ParseError`, which does
    NOT subclass ValueError, so narrowing `except Exception` to `except
    ValueError` turned tolerated bad data into a hard failure. A fabricated
    exception proves only that the handler catches what the test throws.
    """

    @pytest.mark.parametrize(
        "bad_allele",
        ["NOT-AN-ALLELE-ZZZ", "HLA-A*99:99:99zz", "???", "1234"],
    )
    def test_normalize_reports_bad_data_as_value_error(self, bad_allele):
        """The data failure mode must be ValueError, whatever mhcgnomes uses."""
        from presto.data.allele_resolver import normalize_allele_name

        with pytest.raises(ValueError):
            normalize_allele_name(bad_allele)

    def test_sequence_lookup_skips_an_unparseable_key(self):
        from presto.data.loaders import _normalize_mhc_sequence_lookup

        result = _normalize_mhc_sequence_lookup(
            {"NOT-AN-ALLELE-ZZZ": "AAAA", "HLA-A*02:01": "CCCC"}
        )
        assert "HLA-A*02:01" in result

    def test_index_token_falls_back_to_the_raw_spelling(self):
        from presto.data.mhc_index import _normalize_allele_token

        assert _normalize_allele_token("HLA-A*99:99:99zz") == "HLA-A*99:99:99zz"

    def test_good_alleles_still_normalize(self):
        from presto.data.allele_resolver import normalize_allele_name

        assert normalize_allele_name("A*02:01") == "HLA-A*02:01"


class TestSetupFailuresPropagate:
    """A missing dependency must surface, not degrade every allele silently."""

    def test_mhcgnomes_runtime_error_is_not_swallowed(self, monkeypatch):
        import presto.data.mhc_index as mhc_index

        def _explode(_name):
            raise RuntimeError("mhcgnomes is a required dependency")

        monkeypatch.setattr(mhc_index, "normalize_allele_name", _explode)
        with pytest.raises(RuntimeError, match="mhcgnomes"):
            mhc_index._normalize_allele_token("HLA-A*02:01")

    def test_parse_error_is_translated_not_leaked(self):
        """Callers must not need to import mhcgnomes to catch bad data."""
        from presto.data.allele_resolver import _mhcgnomes_parse_errors

        # Called, not read as a module constant: binding it at import time
        # pulled mhcgnomes into every process that touched the package, and
        # froze the result so a later install could never be picked up.
        resolved = _mhcgnomes_parse_errors()
        assert resolved, "ParseError type could not be resolved"
        for exc_type in resolved:
            assert not issubclass(exc_type, ValueError), (
                "if this ever subclasses ValueError the translation is "
                "redundant, but the call sites still rely on it"
            )


class TestNoSilentBlanketCatches:
    """No `except Exception` may be followed immediately by a bare pass/continue.

    Stated as a property over the source so a newly added one is caught, rather
    than as a list of known sites that goes stale.
    """

    def test_data_package_has_no_silent_blanket_catch(self):
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parent.parent / "data"
        offenders = []
        for path in sorted(root.rglob("*.py")):
            lines = path.read_text().splitlines()
            for i, line in enumerate(lines):
                if not re.match(r"\s*except Exception\s*:", line):
                    continue
                # Look at the next non-comment line.
                for follow in lines[i + 1 :]:
                    stripped = follow.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    if stripped in {"pass", "continue"}:
                        offenders.append(f"{path.name}:{i + 1}")
                    break
        assert offenders == [], (
            "these silently swallow every exception, including setup and "
            f"programming errors: {offenders}. Catch the specific type."
        )
