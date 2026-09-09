"""Exercise the pinned Modal client's transport without a cloud call."""

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest


def test_remote_entry_round_trips_without_importing_local_launcher(tmp_path):
    modal = pytest.importorskip("modal")
    assert modal.__version__ == "1.1.4"
    from modal._utils.function_utils import FunctionInfo

    launcher = runpy.run_path(str(Path(__file__).with_name("launch.py")))
    function = launcher["remote_census"]
    info = FunctionInfo(function, serialized=True)
    assert info.module_name is None
    payload = tmp_path / "remote_entry.pickle"
    payload.write_bytes(info.serialized_function())
    receiver = """
import hashlib
import inspect
import json
import sys
from pathlib import Path
from modal._serialization import deserialize
function = deserialize(Path(sys.argv[1]).read_bytes(), None)
assert {'ROOT', 'EXPERIMENT', 'RAW'}.isdisjoint(function.__globals__)
assert 'launch' not in sys.modules
print(json.dumps({
    'name': function.__name__,
    'parameters': list(inspect.signature(function).parameters),
    'code_sha256': hashlib.sha256(function.__code__.co_code).hexdigest(),
}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", receiver, str(payload)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "name": "remote_census",
        "parameters": ["condition", "attempt", "run_dir"],
        "code_sha256": hashlib.sha256(function.__code__.co_code).hexdigest(),
    }
