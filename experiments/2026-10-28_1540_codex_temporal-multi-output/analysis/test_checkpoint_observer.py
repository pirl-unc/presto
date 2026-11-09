"""Observer rollback preserves the common trainer's exact dropout/RNG trajectory."""

from pathlib import Path
import json
import sys
import pytest
import torch

F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F / 'code'))
sys.path.insert(0, str(F.parents[1] / 'tests'))
from checkpoint_observer import KEY, save, restore  # noqa: E402
from test_modular_resume import manifest_fixture, assert_same  # noqa: E402
from presto.training import modular_run as runner  # noqa: E402
from presto.data.modular_sources import ObservationIndex  # noqa: E402


@pytest.mark.parametrize('before_step', [2, 6])
def test_instrumented_interruption_matches_uninstrumented_common_fit(
    tmp_path, monkeypatch, before_step
):
    manifest = manifest_fixture(tmp_path, replay=False)
    value = json.loads(manifest.read_text())
    value['model'].update(d_model=32, n_layers=1, n_heads=4, d_ff=64)
    value['training']['validation_every_steps'] = 2
    manifest.write_text(json.dumps(value))
    index = ObservationIndex(manifest)
    baseline, resumed = tmp_path / 'baseline', tmp_path / 'resumed'
    reference = runner.fit_index(index, baseline)
    step, snapshot = runner.train_step, runner.save_resume
    state = {'steps': 0, 'ids': []}
    binding = {'manifest': index.manifest_sha256, 'observer': 'fixture'}

    def observed_step(*args, **kwargs):
        if state['steps'] + 1 == before_step:
            raise RuntimeError('interrupted for fixture')
        result = step(*args, **kwargs)
        state['steps'] += 1
        state['ids'].extend(row.observation_id for row in args[2])
        return result

    def observed_snapshot(path, *args, **kwargs):
        save(snapshot, path, args, kwargs, lambda: state, binding, lambda: None)

    monkeypatch.setattr(runner, 'train_step', observed_step)
    monkeypatch.setattr(runner, 'save_resume', observed_snapshot)
    with pytest.raises(RuntimeError, match='interrupted for fixture'):
        runner.fit_index(index, resumed)
    state = restore(resumed / 'resume.pt', binding)
    assert state['steps'] == (0 if before_step == 2 else 4)
    assert len(state['ids']) == state['steps'] * 2
    before_step = 100
    actual = runner.fit_index(index, resumed, resume=True)
    assert actual['steps'] == reference['steps'] == state['steps'] == 8
    assert actual['metrics'] == reference['metrics']
    for name in ('last.pt', 'selected.pt'):
        left = torch.load(baseline / name, weights_only=True)
        right = torch.load(resumed / name, weights_only=True)
        for key in ('state_dict', 'optimizer', 'element_coverage', 'support'):
            assert_same(left[key], right[key])
        assert KEY not in right and KEY not in right['metadata']
    with pytest.raises(ValueError, match='recovery contract'):
        restore(resumed / 'resume.pt', {'manifest': 'changed'})

    def no_second_evaluation(*args, **kwargs):
        raise AssertionError('completed fit attempted another evaluation')

    monkeypatch.setattr(runner, 'evaluate', no_second_evaluation)
    assert runner.fit_index(index, resumed, resume=True)['metrics'] == reference['metrics']


def test_mismatched_observer_cannot_replace_last_coherent_snapshot(tmp_path):
    path = tmp_path / 'resume.pt'
    original_bytes = b'previous complete snapshot'
    path.write_bytes(original_bytes)

    def producer(destination):
        torch.save({'metadata': {'resume': {'state': {'steps': 4}}}}, destination)

    with pytest.raises(AssertionError, match='steps differ'):
        save(producer, path, (), {}, lambda: {'steps': 5}, {}, lambda: None)
    assert path.read_bytes() == original_bytes
