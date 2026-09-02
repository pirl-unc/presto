"""A run that resolves no MHC sequences must refuse to train.

Zero resolved MHC is not a degraded pMHC model, it is a different model:
every pair collapses to peptide-only, the groove attends to padding, and the
loss curve looks entirely normal while meaning nothing.

This is not hypothetical. A remote run reported `resolved=0/88,797 (0.00%)`,
raised nothing, and trained for three epochs on empty MHC -- the mhcseqs
catalog had never been built on that machine. The coverage line was printed
and scrolled past.
"""

import pytest

from presto.scripts.train_iedb import MHCResolutionError  # noqa: E402


def _raise_if_zero(coverage):
    """The guard as `run()` applies it, kept in one place.

    Mirrors the check in `train_iedb.run()` rather than importing it, because
    the check lives inline after the strict-resolution branch -- it has to,
    since running before it would preempt the strict path's unresolved-allele
    report. `TestGuardMatchesTheImplementation` pins the two together.

    Reads `PRESTO_ALLOW_ZERO_MHC` from the real environment, exactly as `run()`
    does. It used to also accept an `allow_env=` parameter that nothing read
    and no caller passed -- a test written as `_raise_if_zero(cov,
    allow_env="1")` would have silently consulted the process environment
    instead and appeared to exercise the escape hatch while testing the
    opposite. Callers monkeypatch the variable instead.
    """
    import os

    overall = coverage.get("overall", {})
    rows = int(overall.get("rows_considered", 0) or 0)
    resolved = int(overall.get("resolved_rows", 0) or 0)
    allow = str(os.environ.get("PRESTO_ALLOW_ZERO_MHC", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if resolved == 0 and rows > 0 and not allow:
        raise MHCResolutionError(
            f"no MHC sequences resolved for any of {rows} rows. Training would "
            "silently reduce to a peptide-only model. Usual causes: the mhcseqs "
            "catalog was never built on this machine (run `mhcseqs build`, or "
            "copy ~/.cache/mhcseqs/mhc-full-seqs.csv), or --index-csv points "
            "nowhere. Set PRESTO_ALLOW_ZERO_MHC=1 only if a peptide-only run is "
            "genuinely what you want."
        )


def _coverage(rows: int, resolved: int):
    missing = rows - resolved
    return {
        "overall": {
            "rows_considered": rows,
            "resolved_rows": resolved,
            "missing_rows": missing,
            "resolved_fraction": (resolved / rows) if rows else 0.0,
            "missing_fraction": (missing / rows) if rows else 0.0,
        }
    }


class TestZeroResolutionIsFatal:
    def test_zero_resolved_raises(self):
        with pytest.raises(MHCResolutionError, match="no MHC sequences resolved"):
            _raise_if_zero(_coverage(88_797, 0))

    def test_the_error_names_the_usual_causes(self):
        """A bare failure here costs a GPU-hours-long debugging detour."""
        with pytest.raises(MHCResolutionError) as excinfo:
            _raise_if_zero(_coverage(100, 0))
        message = str(excinfo.value)
        assert "mhcseqs" in message
        assert "--index-csv" in message


class TestNormalCoverageIsUnaffected:
    @pytest.mark.parametrize("resolved", [1, 50, 75, 100])
    def test_any_nonzero_resolution_is_allowed(self, resolved):
        """Partial coverage is normal -- murine alleles resolve poorly."""
        _raise_if_zero(_coverage(100, resolved))

    def test_empty_corpus_does_not_raise(self):
        """Zero of zero is not a resolution failure."""
        _raise_if_zero(_coverage(0, 0))


class TestEscapeHatch:
    def test_peptide_only_runs_can_opt_in(self, monkeypatch):
        monkeypatch.setenv("PRESTO_ALLOW_ZERO_MHC", "1")
        _raise_if_zero(_coverage(100, 0))

    def test_opt_in_is_explicit(self, monkeypatch):
        """A stray or falsey value must not disable the guard."""
        for value in ("", "0", "false", "no"):
            monkeypatch.setenv("PRESTO_ALLOW_ZERO_MHC", value)
            with pytest.raises(MHCResolutionError):
                _raise_if_zero(_coverage(100, 0))


class TestGuardMatchesTheImplementation:
    """The helper above must not drift from the real check in run()."""

    def test_run_contains_the_guard(self):
        import inspect

        from presto.scripts import train_iedb

        source = inspect.getsource(train_iedb)
        assert "PRESTO_ALLOW_ZERO_MHC" in source
        assert "no MHC sequences resolved for any of" in source
        assert "MHCResolutionError" in source

    def test_guard_runs_after_the_strict_check(self):
        """Ordering is load-bearing.

        Placed earlier, it preempted the strict path and the
        unresolved-allele report was never written -- which is how the first
        version of this guard broke an existing test.
        """
        import inspect

        from presto.scripts import train_iedb

        from source_probe import unique_index

        source = inspect.getsource(train_iedb)
        strict_at = unique_index(source, "Unresolved MHC alleles are present", where="train_iedb")
        guard_at = unique_index(source, "no MHC sequences resolved for any of", where="train_iedb")
        assert strict_at < guard_at
