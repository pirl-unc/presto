"""Materialize the frozen shared recipe and report only prospective support."""

import hashlib
import json
import resource
import shutil
import time
from pathlib import Path

from hitlist.evidence_recipe import materialize_recipe, verify_recipe_package
from hitlist.evidence_store import EvidenceStore

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    start = time.perf_counter()
    config = json.loads((ROOT / 'reproduce/launch.json').read_text())
    assert not (ROOT / 'results/admission.json').exists()
    assert shutil.disk_usage(ROOT).free / 2**30 >= config['limits']['minimum_free_disk_gib']
    for path, expected in config['hitlist_source_hashes'].items():
        assert sha(Path(config['hitlist_path']) / path) == expected
    assert sha(Path(config['source_directory'])/'manifest.json') == config['source_manifest_sha256']
    shutil.copy2(__file__, ROOT/'reproduce/source/admit.py')
    store = EvidenceStore.open(config['source_directory'])
    recipe = {'schema_version':1,'name':'lymph_dc_observed_composition','registered_recipe':config['recipe_id']}
    materialize_recipe(store, recipe, config['recipe_directory'])
    manifest = verify_recipe_package(config['recipe_directory'], source_package=config['source_directory'])
    summary = manifest['coverage']
    admitted = {}
    for block in sorted({row['block_id'] for row in summary['coverage']}):
        counts = {row['split']:row['n_components'] for row in summary['coverage'] if row['block_id']==block}
        admitted[block] = {'n_components':counts,'supported':all(counts[split]>=minimum for split,minimum in config['pilot_min_components'].items())}
    result = {'status':'completed','recipe_manifest_sha256':sha(Path(config['recipe_directory'])/'manifest.json'),'coverage':summary,'cohort_support':admitted,'runtime_seconds':time.perf_counter()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30}
    assert result['runtime_seconds'] < config['limits']['stage_seconds']
    assert result['peak_rss_gib'] < config['limits']['rss_gib']
    (ROOT/'results/admission.json').write_text(json.dumps(result,indent=2)+'\n')
    shutil.copy2(Path(config['recipe_directory'])/'manifest.json',ROOT/'results/recipe-manifest.json')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
