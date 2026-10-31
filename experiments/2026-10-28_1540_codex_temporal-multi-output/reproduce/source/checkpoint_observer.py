"""Atomically bind experiment diagnostics to the canonical resumable snapshot.

The common trainer alone owns model, optimizer, RNG, order and evaluation state.
An extra top-level field is ignored by its checkpoint loader and does not propagate
into selected checkpoints. This keeps observer rollback at exactly the same step.
"""

from pathlib import Path
import os
import torch

KEY = 'presto_experiment_observer_v1'


def save(original, path, args, kwargs, observer, binding, commit):
    path = Path(path)
    common = path.with_suffix('.common-staging.pt')
    temporary = path.with_suffix('.observer-staging.pt')
    original(common, *args, **kwargs)
    payload = torch.load(common, map_location='cpu', weights_only=True)
    steps = payload['metadata']['resume']['state']['steps']
    state = observer()
    assert state['steps'] == steps, 'observer and canonical snapshot steps differ'
    payload[KEY] = {'binding': binding, 'state': state}
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    common.unlink()
    commit()


def restore(path, binding):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    observer = payload.get(KEY)
    if not observer or observer.get('binding') != binding:
        raise ValueError('missing observer state or changed experiment recovery contract')
    state = observer['state']
    if state['steps'] != payload['metadata']['resume']['state']['steps']:
        raise ValueError('observer and canonical snapshot steps differ')
    return state
