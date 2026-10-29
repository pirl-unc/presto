"""Source-review revisions must not relax the pre-optimization admission gate."""

from pathlib import Path
import copy
import json
import sys
import pytest

F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F / 'code'))
from freeze_contract import validate_preflight  # noqa: E402
from revise_cohort import verify_delta  # noqa: E402


def contract():
    p = {
        'review_metadata': {
            'preflight.json': {'status': 'pending_in_trainers', 'expected_scales': {'fixture': 1}}
        },
        'expected_scales': {'fixture': 1},
        'source_review': {
            'revision': 'kinetic-review-v2',
            'removed_rows': 20,
            'policy_sha256': 'policy',
            'runtime_source': 'commit',
        },
    }
    audit = {
        'review_revision': 'kinetic-review-v2',
        'removed_rows': 20,
        'exact_reviewed_row_removal': True,
        'retained_source_bytes_and_order_verified': True,
        'unchanged_holdouts_and_backgrounds': True,
        'inherited_full_verification': {'status': 'passed'},
        'canonical_preflight': 'pending_in_trainers',
        'input_receipts': {
            'hitlist-v4-iedb-train.jsonl': {'before_sha256': 'a', 'after_sha256': 'b'},
            'test.jsonl': {'before_sha256': 'c', 'after_sha256': 'c'},
        },
    }
    pins = {'git_commit': 'commit', 'files': {'data/assay_measurement_reviews.json': 'policy'}}
    return p, audit, pins


def test_declared_pending_preflight_requires_specific_verified_revision():
    p, audit, pins = contract()
    original = copy.deepcopy((p, audit, pins))
    validate_preflight(p, audit, pins)
    assert (p, audit, pins) == original  # Pending status must not be relabeled passed.
    for field in (
        'exact_reviewed_row_removal',
        'retained_source_bytes_and_order_verified',
        'unchanged_holdouts_and_backgrounds',
    ):
        invalid = {**audit, field: False}
        with pytest.raises(AssertionError):
            validate_preflight(p, invalid, pins)
    audit['input_receipts']['test.jsonl']['after_sha256'] = 'changed'
    with pytest.raises(AssertionError):
        validate_preflight(p, audit, pins)


def test_runtime_registry_and_scale_changes_fail_closed():
    p, audit, pins = contract()
    bad = copy.deepcopy(p)
    bad['expected_scales']['fixture'] = 2
    with pytest.raises(AssertionError):
        validate_preflight(bad, audit, pins)
    for field, value in [('git_commit', 'other'), ('files', {})]:
        with pytest.raises((AssertionError, KeyError)):
            validate_preflight(p, audit, {**pins, field: value})


def test_row_removal_verifier_rejects_any_unreviewed_mutation(tmp_path):
    original, revised = tmp_path / 'original', tmp_path / 'revised'
    lines = [json.dumps({'observation_id': str(i), 'value': i}) + '\n' for i in range(4)]
    original.write_text(''.join(lines))
    revised.write_text(''.join(lines[i] for i in (0, 2, 3)))
    verify_delta(original, revised, {'1'})
    for content in (
        ''.join(lines[i] for i in (2, 0, 3)),
        ''.join(lines[i] for i in (0, 3)),
        ''.join(lines[i] for i in (0, 2, 3, 3)),
        ''.join(lines[i] for i in (0, 2, 3)).replace('"value": 2', '"value": 99'),
    ):
        revised.write_text(content)
        with pytest.raises(AssertionError):
            verify_delta(original, revised, {'1'})
