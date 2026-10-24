"""Nested sequence features over the unchanged analytical digestion model."""
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch
from presto.models.independent_cut import log_fragment_expectation

FAMILY = Path(__file__).resolve().parents[1]
AA = 'ACDEFGHIKLMNPQRSTVWY'
LN10 = math.log(10)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def center(x, w):
    return x - (w[:, None] * x).sum(0) / w.sum() if x.ndim == 2 else x - w @ x / w.sum()


def raw_rate(sequence):
    rows = []
    for i in range(len(sequence) - 1):
        row = []
        for delta in [-2, -1, 2, 3]:
            j = i + delta
            residue = sequence[j] if 0 <= j < len(sequence) else '_'
            row.extend([float(residue == a) for a in AA[1:] + '_'])
        rows.append(row)
    return np.array(rows)


def raw_response(g, instruments):
    out = np.zeros((len(g['rows']), len(instruments) * 38))
    offset = instruments.index(g['instrument']) * 38
    for i, r in enumerate(g['rows']):
        peptide = r['peptide_sequence']
        out[i, offset:offset + 38] = [float(residue == a) for residue in [peptide[0], peptide[-1]] for a in AA[1:]]
    return out


def extend(groups, key, rawkey, names, response):
    train = [g for g in groups if g['split'] == 'train']
    xw, ew = [], []
    for g in train:
        x, e = g[key].numpy(), g[rawkey]
        w = g['w'].numpy() if response else np.full(len(x), float(g['w'].sum()) / len(x))
        if response:
            x, e = center(x, w), center(e, w)
        xw.append(x * np.sqrt(w)[:, None]); ew.append(e * np.sqrt(w)[:, None])
    xw, ew = np.concatenate(xw), np.concatenate(ew)
    regression = np.linalg.lstsq(xw, ew, rcond=1e-12)[0]
    residual = ew - xw @ regression
    gram = residual.T @ residual
    rms = np.sqrt(np.maximum(np.diag(gram), 0))
    supported = rms > 1e-10
    scale = np.where(supported, rms, 1.)
    standardized = gram / scale[:, None] / scale[None, :]
    eigen, vectors = np.linalg.eigh(standardized)
    keep = eigen > max(float(eigen[-1]), 1.) * 1e-10
    basis = vectors[:, keep]
    for i in range(basis.shape[1]):
        nz = np.flatnonzero(np.abs(basis[:, i]) > 1e-10)
        if basis[nz[0], i] < 0: basis[:, i] *= -1
    extrapolation = {}
    for g in groups:
        r = (g[rawkey] - g[key].numpy() @ regression) / scale
        projected = r @ basis
        g[key + '_extended'] = torch.cat([g[key], torch.tensor(projected)], dim=1)
        if response: r = center(r, g['w'].numpy())
        off = r - (r @ basis) @ basis.T
        extrapolation[g['condition']] = dict(rows=len(r), outside_span_rows=int((np.linalg.norm(off, axis=1) > 1e-6).sum()), maximum_residual_norm=float(np.linalg.norm(off, axis=1).max()))
    return dict(names=names, regression=regression.tolist(), scale=scale.tolist(), basis=basis.tolist(), eigenvalues=eigen.tolist(), added_dimensions=int(keep.sum()), absent_raw_columns=[n for n, s in zip(names, supported) if not s], extrapolation=extrapolation, training_orthogonality_error=float(np.max(np.abs(xw.T @ residual))))


def load():
    torch.set_num_threads(1)
    cfg = json.loads((FAMILY / 'reproduce/launch.json').read_text())
    for p, h in cfg['source_hashes'].items(): assert sha(p) == h, p
    path = Path(cfg['observation_family']) / 'code/observation.py'
    spec = importlib.util.spec_from_file_location('fixed_observation', path)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    _, groups, feature, initial = old.load_data()
    instruments = feature['instruments']
    for g in groups:
        g['raw_rate_extra'] = raw_rate(g['sequence'])
        g['raw_response_extra'] = raw_response(g, instruments)
    rate_names = [f'bond_offset{d}:{a}' for d in [-2, -1, 2, 3] for a in AA[1:] + '_']
    response_names = [f'{ins}:{pos}:{a}' for ins in instruments for pos in ['first', 'last'] for a in AA[1:]]
    features = dict(old_response=feature, rate=extend(groups, 'x', 'raw_rate_extra', rate_names, False), response=extend(groups, 'z', 'raw_response_extra', response_names, True))
    for g in groups:
        g['templates'] = []
        for t in g['t'].unique(sorted=True).tolist():
            mask = g['t'] == t
            spans, inverse = torch.stack([g['a'][mask], g['b'][mask]], 1).unique(dim=0, sorted=True, return_inverse=True)
            g['templates'].append(dict(time=t, mask=mask, spans=spans, inverse=inverse))
    return cfg, old, groups, features, initial


def variant(groups, initial, arm):
    rate = arm in ['rate_context', 'both', 'both_hybrid']
    response = arm in ['response_termini', 'both', 'both_hybrid']
    result = [dict(g, x=g['x_extended'] if rate else g['x'], z=g['z_extended'] if response else g['z']) for g in groups]
    nx, nz = result[0]['x'].shape[1], result[0]['z'].shape[1]
    theta = torch.cat([initial, initial.new_zeros(nx - 40)])
    return result, theta, initial.new_zeros(nz)


def amount(theta, g):
    return log_fragment_expectation(g['x'] @ theta, g['a'], g['b'], g['t'], validate=False) / LN10


def objective(theta, gamma, groups):
    data = theta.new_zeros(())
    for g in groups:
        if g['split'] != 'train': continue
        r = amount(theta, g) + g['z'] @ gamma - g['y']
        r = r - (g['w'] * r).sum() / g['w'].sum()
        data += (g['w'] * r.square()).sum()
    return data + .005 * (theta[1:].square().sum() + gamma.square().sum()), data


def independent(theta, gamma, groups):
    """NumPy analytical expectation and full parameter Jacobian."""
    loss = 0.; grad = np.zeros(len(theta) + len(gamma)); amounts = []
    for g in groups:
        x, z, t, a, b = [g[k].numpy() for k in ['x', 'z', 't', 'a', 'b']]
        rate = np.exp(x @ theta)
        prefix = np.r_[0., np.cumsum(rate)]
        xp = np.vstack([np.zeros((1, len(theta))), np.cumsum(rate[:, None] * x, axis=0)])
        lv = -t * (prefix[b-1] - prefix[a]); jac = -t[:, None] * (xp[b-1] - xp[a])
        for mask, idx in [(a > 0, a-1), (b <= len(rate), b-1)]:
            ii = np.flatnonzero(mask); k = idx[ii]; h = rate[k] * t[ii]
            lv[ii] += np.log(-np.expm1(-h))
            jac[ii] += (h * np.exp(-h) / (-np.expm1(-h)))[:, None] * x[k]
        lv /= LN10; jac /= LN10; amounts.append(lv)
        if g['split'] != 'train': continue
        w, y = g['w'].numpy(), g['y'].numpy()
        r = center(lv + z @ gamma - y, w)
        loss += w @ r**2
        grad[:len(theta)] += 2 * (w*r) @ jac
        grad[len(theta):] += 2 * (w*r) @ z
    loss += .005 * (theta[1:] @ theta[1:] + gamma @ gamma)
    grad[1:len(theta)] += .01 * theta[1:]; grad[len(theta):] += .01 * gamma
    return loss, grad, amounts


def prepare(theta, gamma, groups):
    records = []
    for g in groups:
        if g['split'] != 'train': continue
        logs = g['x'] @ theta
        error = amount(theta, g) + g['z'] @ gamma - g['y']
        error -= (g['w'] * error).sum() / g['w'].sum()
        coeff = 2 * g['w'] * error / LN10
        for template in g['templates']:
            a, b = template['spans'].T; t = template['time']
            combined = torch.zeros(len(a), dtype=torch.float64)
            combined.scatter_add_(0, template['inverse'], coeff[template['mask']])
            mass = log_fragment_expectation(logs, a, b, torch.full((len(a),), t, dtype=torch.float64)).exp()
            assert bool((torch.isfinite(mass) & (mass > 0)).all())
            h = logs.exp() * t
            cut = h * torch.exp(-h) / (-torch.expm1(-h))
            prefix = torch.cat([torch.zeros((1, len(theta)), dtype=torch.float64), (h[:, None] * g['x']).cumsum(0)])
            score = -(prefix[b-1] - prefix[a])
            left, right = a > 0, b <= len(logs)
            score[left] += cut[a[left]-1, None] * g['x'][a[left]-1]
            score[right] += cut[b[right]-1, None] * g['x'][b[right]-1]
            records.append(dict(logs=logs, time=t, starts=a, ends=b, mass=mass, coefficient=combined, conditional=score))
    return records


def decomposition(rows):
    by_curve = {}
    for r in rows: by_curve.setdefault(r['curve_id'], []).append(r)
    persistent = temporal = 0.
    for rr in by_curve.values():
        w = np.array([r['weight'] for r in rr]); e = np.array([r['centered_prediction']-r['centered_target'] for r in rr])
        mean = w @ e / w.sum(); persistent += w.sum() * mean**2; temporal += w @ (e-mean)**2
    return dict(persistent_mse=float(persistent), temporal_mse=float(temporal), persistent_fraction=float(persistent / (persistent+temporal)))


def evaluate(old, theta, gamma, groups, out):
    with torch.no_grad():
        amounts = [amount(theta, g) for g in groups]
        responses = [g['z'] @ gamma for g in groups]
        rows = old.predictions(groups, [a+b for a,b in zip(amounts,responses)], amounts, responses)
    with gzip.open(out/'predictions.jsonl.gz', 'wt') as f:
        for r in rows: f.write(json.dumps(r, allow_nan=False)+'\n')
    result = {}
    for split in ['train', 'val']:
        rr = [r for r in rows if r['split'] == split]
        result[split] = dict(**old.metrics(rr), **decomposition(rr))
    per_parent = {}
    for parent in sorted({r['construct_id'] for r in rows}):
        rr = [dict(r) for r in rows if r['construct_id'] == parent]
        w = sum(r['weight'] for r in rr)
        for r in rr: r['weight'] /= w
        per_parent[parent] = dict(split=rr[0]['split'], **old.metrics(rr), **decomposition(rr))
    return dict(metrics=result, per_parent=per_parent)
