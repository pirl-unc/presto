"""The loader must refuse a hitlist whose column semantics it misreads.

`data/hitlist_source.py` reads `apm_genes_perturbed` as the **per-sample**
perturbation. That is only true from hitlist 1.46.0 (pirl-unc/hitlist#353).
Before then the column *was* the study-level roll-up: it ORed the parent
study's knockout panel onto every sample, so a WT control inside a KO study
carried the KO flag.

The requested column set resolves on 1.41 too, so an older install trains
happily and reports nothing -- 816,023 observations (18.4%) get the wrong
perturbation label, 716,992 of them genuinely unperturbed rows wearing their
study's panel. A silent wrong answer is the reason this is enforced at runtime
and not just declared in `pyproject.toml`: the dependency floor governs a fresh
resolve, while a long-lived environment can sit on a stale wheel indefinitely.
"""

import pytest

from presto.data.hitlist_source import (  # noqa: E402
    MINIMUM_HITLIST_VERSION,
    _parse_version,
    require_supported_hitlist,
)


class TestVersionParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1.46.0", (1, 46, 0)),
            ("1.53.1", (1, 53, 1)),
            ("2.0", (2, 0)),
            ("1.53.1.dev0", (1, 53, 1)),
            ("1.46.0rc1", (1, 46, 0)),
        ],
    )
    def test_leading_numeric_components(self, raw, expected):
        assert _parse_version(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "unknown", "dev"])
    def test_unreadable_versions_yield_none(self, raw):
        assert _parse_version(raw) is None


class TestFloorEnforcement:
    @pytest.mark.parametrize("raw", ["1.46.0", "1.53.1", "1.54.0", "2.0.0"])
    def test_supported_versions_pass(self, raw):
        require_supported_hitlist(raw)

    @pytest.mark.parametrize("raw", ["1.41", "1.45.0", "1.30.16", "0.9"])
    def test_versions_below_the_floor_raise(self, raw):
        with pytest.raises(RuntimeError, match="too old"):
            require_supported_hitlist(raw)

    def test_the_error_explains_the_silent_failure(self):
        """A version error that does not say what goes wrong invites a pin bump."""
        with pytest.raises(RuntimeError) as excinfo:
            require_supported_hitlist("1.45.0")
        message = str(excinfo.value)
        assert "apm_genes_perturbed" in message
        assert "1.46.0" in message

    @pytest.mark.parametrize("raw", [None, "", "unknown"])
    def test_unreadable_version_is_allowed_through(self, raw):
        """Editable installs and source checkouts often have no `__version__`.

        Refusing to run there would be worse than the risk it guards against.
        """
        require_supported_hitlist(raw)

    def test_the_floor_matches_what_the_module_documents(self):
        assert MINIMUM_HITLIST_VERSION == (1, 46, 0)


class TestDeclaredDependencyAgrees:
    """The runtime guard and the packaging floor must not drift apart."""

    def test_pyproject_floor_matches_the_runtime_floor(self):
        from pathlib import Path

        wanted = ".".join(str(part) for part in MINIMUM_HITLIST_VERSION)
        text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
        assert f'"hitlist>={wanted}"' in text

    def test_the_remote_launcher_installs_the_same_floor(self):
        """A worker built from an older wheel would train on wrong labels."""
        from pathlib import Path

        wanted = ".".join(str(part) for part in MINIMUM_HITLIST_VERSION)
        launcher = (
            Path(__file__).resolve().parents[1] / "scripts" / "train_remote.py"
        ).read_text()
        assert f'"hitlist>={wanted}"' in launcher
