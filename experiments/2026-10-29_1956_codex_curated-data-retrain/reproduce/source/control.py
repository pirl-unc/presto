"""Retry only transient local transport failures while retaining remote call IDs."""

import time
from modal.exception import FunctionTimeoutError
from modal.exception import ConnectionError as ModalConnectionError
from grpclib.exceptions import StreamTerminatedError


def get_result(call, *, wait=True):
    failures = 0
    while True:
        try:
            return call.get(timeout=30 if wait else 0)
        except FunctionTimeoutError:
            raise  # A real remote execution timeout is not a polling timeout.
        except TimeoutError:
            if not wait:
                raise
            failures = 0
        except Exception as exc:
            message = str(exc).lower()
            transport = isinstance(
                exc, (ConnectionError, ModalConnectionError, StreamTerminatedError)
            ) or any(
                token in message
                for token in (
                    'nodename nor servname provided',
                    'temporary failure in name resolution',
                    'name or service not known',
                )
            )
            if not transport or failures >= 8:
                raise
            failures += 1
            print(
                'Retrying saved-call transport',
                type(exc).__name__,
                repr(exc),
                'attempt',
                failures,
                flush=True,
            )
            time.sleep(5)
