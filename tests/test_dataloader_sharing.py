"""Worker->parent tensor sharing must not scale with fields per batch.

A `PrestoBatch` carries ~50 tensor fields. Under torch's default
`file_descriptor` sharing strategy every batch passes that many descriptors
from each worker, multiplied by prefetch depth, and the process runs out. The
symptom names neither descriptors nor the cause:

    RuntimeError: received 0 items of ancdata
    Error: Pin memory thread exited unexpectedly

Observed on a 48-core box with `--num-workers 16`, dying on the first batch of
a run that was otherwise correctly configured.
"""

import pytest

torch = pytest.importorskip("torch")

from presto.data.loaders import (  # noqa: E402
    _raise_open_file_limit,
    _use_file_system_sharing,
)


class TestSharingStrategy:
    def test_file_descriptor_strategy_is_replaced(self):
        import torch.multiprocessing as mp

        available = mp.get_all_sharing_strategies()
        if "file_descriptor" not in available:
            pytest.skip("platform offers only file_system sharing")
        original = mp.get_sharing_strategy()
        try:
            mp.set_sharing_strategy("file_descriptor")
            _use_file_system_sharing()
            assert mp.get_sharing_strategy() == "file_system"
        finally:
            mp.set_sharing_strategy(original)

    def test_an_explicit_choice_is_respected(self):
        """Only the default is overridden, not a deliberate setting."""
        import torch.multiprocessing as mp

        if "file_system" not in mp.get_all_sharing_strategies():
            pytest.skip("file_system unavailable")
        original = mp.get_sharing_strategy()
        try:
            mp.set_sharing_strategy("file_system")
            _use_file_system_sharing()
            assert mp.get_sharing_strategy() == "file_system"
        finally:
            mp.set_sharing_strategy(original)

    def test_is_idempotent(self):
        _use_file_system_sharing()
        _use_file_system_sharing()


class TestOpenFileLimit:
    def test_soft_limit_is_not_lowered(self):
        resource = pytest.importorskip("resource")
        before_soft, before_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        _raise_open_file_limit()
        after_soft, after_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        assert after_soft >= before_soft
        assert after_hard == before_hard

    def test_never_exceeds_the_hard_limit(self):
        """Raising past the administrator's ceiling would just raise OSError."""
        resource = pytest.importorskip("resource")
        _raise_open_file_limit()
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard != resource.RLIM_INFINITY:
            assert soft <= hard

    def test_is_idempotent(self):
        _raise_open_file_limit()
        _raise_open_file_limit()


class TestAppliedWhereItMatters:
    def test_multiprocess_loaders_get_the_fix(self):
        """Single-process loading needs neither, so it must not pay for them."""
        import inspect

        from presto.data import loaders

        from source_probe import unique_index

        # Assert the call sits *inside* a worker guard, rather than merely
        # after the first one. `if num_workers > 0:` occurs twice in this
        # function -- the original ordering check compared against whichever
        # came first and would have passed even if the sharing call had
        # escaped its guard entirely.
        source = inspect.getsource(loaders.create_dataloader)
        unique_index(source, "_use_file_system_sharing()", where="create_dataloader")
        lines = source.splitlines()
        call_lines = [i for i, line in enumerate(lines) if "_use_file_system_sharing()" in line]
        assert len(call_lines) == 1, call_lines
        index = call_lines[0]
        preceding = lines[index - 1].strip()
        assert preceding == "if num_workers > 0:", (
            "the file-system sharing call is no longer directly inside a "
            f"worker guard; the line above it is {preceding!r}. Single-process "
            "loading would pay for a fix it does not need."
        )
        assert lines[index].startswith(" " * (len(lines[index - 1]) - len(preceding) + 4)), (
            "the sharing call is not indented under the guard"
        )
