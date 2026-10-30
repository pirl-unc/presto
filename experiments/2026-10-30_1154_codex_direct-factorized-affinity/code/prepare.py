"""Reuse immutable observations; declare all measured affinity configurations."""

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare(parent, prepared, destination, source_volume_root):
    from presto.data.assay_measurement import AssayMethod, AssayTrack
    from presto.models.assay_measurement import AFFINITY_ENDPOINTS
    from presto.models.presto import PrestoConfig

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    tracks, support, all_methods = {}, Counter(), []
    for key, count in prepared['methods'].items():
        split, endpoint, cls, descriptor, review = key.split(':')
        if endpoint not in AFFINITY_ENDPOINTS:
            continue
        preparation, mechanism, detector = descriptor.split('/')
        method = AssayMethod(preparation, mechanism, detector, review)
        if not method.usable:
            continue
        track = AssayTrack(endpoint, preparation, mechanism, detector)
        all_methods.append(track.key)
        if split == 'train':
            tracks[track.key] = track
            support[track.key] += count
    assert not set(all_methods) - tracks.keys(), 'held-out assay missing TRAIN registry'
    model = {
        **parent['model'],
        'assay_mode': 'factorized',
        'assay_tracks': [asdict(tracks[key]) for key in sorted(tracks)],
    }
    PrestoConfig.from_dict(model)
    sources = [{**s, 'path': str(destination / Path(s['path']).name)} for s in prepared['sources']]
    manifests = {}
    for seed in (42, 43):
        condition = f'direct-s{seed}'
        manifest = {
            **parent,
            'model': model,
            'sources': sources,
            'training': {**prepared['training'], 'seed': seed},
        }
        assert manifest['training']['max_steps'] == 4000
        path = destination / (condition + '.json')
        write(path, manifest)
        manifests[condition] = {'path': str(path), 'sha256': sha(path)}
    result = {
        'status': 'prepared',
        'source_volume_root': str(source_volume_root),
        'sources': sources,
        'manifests': manifests,
        'counts': prepared['counts'],
        'methods': prepared['methods'],
        'disposition': prepared['disposition'],
        'assay_tracks_train_counts': dict(support),
        'parent_metadata_sha256': hashlib.sha256(
            json.dumps(prepared, sort_keys=True).encode()
        ).hexdigest(),
        'observation_changes': 'none: only source locations and model configuration change',
        'training': manifest['training'],
    }
    write(destination / 'prepared.json', result)
    return result
