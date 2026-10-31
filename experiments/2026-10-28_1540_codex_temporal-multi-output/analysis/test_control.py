"""Transport retries must not hide real remote execution failures."""

from pathlib import Path
import sys
import pytest
from modal.exception import FunctionTimeoutError
from modal.exception import ConnectionError as ModalConnectionError
from grpclib.exceptions import StreamTerminatedError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from control import get_result


class Call:
    def __init__(self, replies):
        self.replies = iter(replies)

    def get(self, *, timeout):
        value = next(self.replies)
        if isinstance(value, Exception):
            raise value
        return value


def test_dns_and_poll_timeout_recover_without_resubmitting(monkeypatch):
    monkeypatch.setattr('control.time.sleep', lambda _: None)
    result = get_result(
        Call([OSError(8, 'nodename nor servname provided'), TimeoutError(), {'done': True}])
    )
    assert result == {'done': True}


@pytest.mark.parametrize(
    'error',
    [
        FunctionTimeoutError('remote deadline'),
        ValueError('source hash mismatch'),
        OSError(28, 'disk full'),
    ],
)
def test_actual_failures_are_not_retried(error):
    with pytest.raises(type(error)):
        get_result(Call([error]))


def test_nonblocking_timeout_remains_pending():
    with pytest.raises(TimeoutError):
        get_result(Call([TimeoutError()]), wait=False)


def test_modal_sdk_connection_error_recovers_without_resubmitting(monkeypatch):
    monkeypatch.setattr('control.time.sleep', lambda _: None)
    assert get_result(Call([ModalConnectionError('Deadline exceeded'), {'done': True}])) == {
        'done': True
    }


def test_transport_retries_are_bounded(monkeypatch):
    monkeypatch.setattr('control.time.sleep', lambda _: None)
    with pytest.raises(ConnectionError):
        get_result(Call([ConnectionError('temporary')] * 10))


def test_closed_transport_stream_recovers_same_call(monkeypatch):
    monkeypatch.setattr('control.time.sleep', lambda _: None)
    assert get_result(Call([StreamTerminatedError('Connection lost'), {'done': True}])) == {
        'done': True
    }
