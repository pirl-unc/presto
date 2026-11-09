"""Surface terminal coordinator errors instead of leaving a stale preparing badge."""

import json
from pathlib import Path
import time

import modal
from modal.exception import FunctionTimeoutError

F = Path(__file__).resolve().parents[1]
identifier = json.loads((F / 'reproduce/coordinator-call.json').read_text())['call_id']
call = modal.FunctionCall.from_id(identifier)
live = modal.Dict.from_name('presto-cross-class-live')
while True:
    try:
        result = call.get(timeout=30)
    except FunctionTimeoutError as error:
        failure = repr(error)
    except TimeoutError:
        continue
    except Exception as error:
        failure = repr(error)
    else:
        (F / 'results/coordinator-result.json').write_text(json.dumps(result, indent=2) + '\n')
        break
    state = live.get(F.name + ':coordinator', {})
    state.update(
        status='coordinator_error_requires_review', error=failure, updated_unix=time.time()
    )
    live.put(F.name + ':coordinator', state)
    (F / 'results/coordinator-error.json').write_text(json.dumps(state, indent=2) + '\n')
    print(failure, flush=True)
    break
