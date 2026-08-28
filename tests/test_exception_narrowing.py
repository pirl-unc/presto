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


class TestSetupFailuresPropagate:
    def test_mhcgnomes_runtime_error_is_not_swallowed(self, monkeypatch):
        """A missing dependency must surface, not degrade every allele."""
        import presto.data.mhc_index as mhc_index

        def _explode(_name):
            raise RuntimeError("mhcgnomes is a required dependency")

        monkeypatch.setattr(mhc_index, "normalize_allele_name", _explode)
        with pytest.raises(RuntimeError, match="mhcgnomes"):
            mhc_index._normalize_allele_token("HLA-A*02:01")

    def test_unparseable_allele_is_still_tolerated(self, monkeypatch):
        """The narrowing must not turn bad data into a crash."""
        import presto.data.mhc_index as mhc_index

        def _reject(name):
            raise ValueError(f"mhcgnomes failed to parse allele: {name!r}")

        monkeypatch.setattr(mhc_index, "normalize_allele_name", _reject)
        # Returns something rather than raising; the raw spelling survives.
        assert mhc_index._normalize_allele_token("HLA-A*02:01")


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
