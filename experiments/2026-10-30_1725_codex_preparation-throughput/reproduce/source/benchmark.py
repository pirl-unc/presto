"""Time actual common preparation phases in an isolated frozen runtime."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import sys
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('runtime', 'manifest', 'output', 'condition'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.runtime)
    import psutil
    import torch
    from presto.data import modular_sources as sources
    from presto.models.modular_dependencies import row_has_trainable_loss
    from presto.training import modular_run as runner
    from presto.training.modular_sampling import epoch_order

    torch.set_num_threads(4)
    normalized = sources.normalized_observation
    calls = 0

    def counted(*a, **kw):
        nonlocal calls
        calls += 1
        return normalized(*a, **kw)

    sources.normalized_observation = counted
    phases = {}

    def measure(name, function):
        before = calls
        start = time.perf_counter()
        result = function()
        phases[name] = {
            'seconds': time.perf_counter() - start,
            'normalized_rows': calls - before,
            'rss_bytes': psutil.Process().memory_info().rss,
        }
        return result

    index = measure('admission', lambda: sources.ObservationIndex(args.manifest))
    config, training, scales, admission = measure('preflight', lambda: runner.preflight(index))
    model, _ = runner.initialized_model(config, index)

    def filter_rows():
        if hasattr(runner, 'trainable_indices'):
            return runner.trainable_indices(index, model)
        return tuple(
            [
                i
                for i in index.partitions.get(f'train:{family}', [])
                if row_has_trainable_loss(model, index[i])
            ]
            for family in ('assay', 'identity')
        )

    trainable = measure('trainable_rows', filter_rows)
    rng = random.Random(42)
    order = measure(
        'epoch_order',
        lambda: epoch_order(
            index,
            trainable[0] or trainable[1],
            training['balance_by'],
            rng,
            training['stratum_weights'],
        ),
    )
    results = {
        'condition': args.condition,
        'hardware': {
            'platform': platform.platform(),
            'python': sys.version,
            'processor': platform.processor(),
            'cpu_count': os.cpu_count(),
            'torch': torch.__version__,
            'cpuinfo': Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0]
            if Path('/proc/cpuinfo').exists()
            else platform.processor(),
            'memory_total': psutil.virtual_memory().total,
        },
        'phases': phases,
        'total_seconds': sum(x['seconds'] for x in phases.values()),
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == 'darwin' else 1024),
        'rows': len(index),
        'order_length': len(order),
        'metadata_code_bytes': (index._metadata_codes.itemsize * len(index._metadata_codes))
        if hasattr(index, '_metadata_codes')
        else 0,
        'metadata_categories': len(index._metadata_values)
        if hasattr(index, '_metadata_values')
        else 0,
        'invariants': {
            'scales': {key: asdict(value) for key, value in scales.items()},
            'admission': admission,
            'partitions': digest(index.partitions),
            'trainable_indices': digest(trainable),
            'order': digest(order),
            'rng': digest(rng.getstate()),
        },
    }
    Path(args.output).write_text(json.dumps(results, indent=2) + '\n')
    print(
        json.dumps({key: results[key] for key in ('condition', 'rows', 'total_seconds', 'phases')})
    )


if __name__ == '__main__':
    main()
