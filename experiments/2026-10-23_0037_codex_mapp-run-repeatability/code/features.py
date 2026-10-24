"""Train-only rank selection for identifiable observed-set sequence scores."""

import numpy as np
from scipy.linalg import qr

AA = "ACDEFGHIKLMNPQRSTVWY"


def raw_features(sequence, candidates):
    names = ["length/40"] + ["composition:" + a for a in AA]
    for role, alphabet in (
        ("first", AA),
        ("last", AA),
        ("previous", AA + "-"),
        ("next", AA + "-"),
    ):
        names.extend(role + ":" + a for a in alphabet)
    names.extend(["start/(L-1)", "end/L"])
    values = []
    for c in candidates:
        assert len(c["start_positions_0based"]) == 1
        start = c["start_positions_0based"][0]
        peptide = c["sequence"]
        end = start + len(peptide)
        assert sequence[start:end] == peptide and set(sequence) <= set(AA)
        x = [len(peptide) / 40] + [peptide.count(a) / len(peptide) for a in AA]
        for residue, alphabet in (
            (peptide[0], AA),
            (peptide[-1], AA),
            (sequence[start - 1] if start else "-", AA + "-"),
            (sequence[end] if end < len(sequence) else "-", AA + "-"),
        ):
            x.extend(float(residue == a) for a in alphabet)
        x.extend([start / max(len(sequence) - 1, 1), end / len(sequence)])
        values.append(x)
    values = np.array(values, dtype=np.float64)
    return values - values.mean(axis=0), names


def design(examples, parents, architecture):
    raw = []
    for e in examples:
        x, names = raw_features(parents[e["parent_sequence_sha256"]], e["candidates"])
        if architecture == "length":
            x, names = x[:, :1], names[:1]
        elif architecture == "interaction":
            x = np.column_stack([x, e["context"][0] * x])
            names = names + ["TNF*" + n for n in names]
        else:
            assert architecture == "sequence"
        raw.append(x)
    train = np.vstack([x for x, e in zip(raw, examples) if e["split"] == "train"])
    rms = np.sqrt(np.mean(train**2, axis=0))
    eligible = np.flatnonzero(rms > 1e-12)
    scales = np.where(rms > 1e-12, rms, 1.0)
    _, r, pivots = qr(
        train[:, eligible] / scales[eligible], mode="economic", pivoting=True
    )
    diagonal = np.abs(np.diag(r))
    rank = int(np.sum(diagonal > diagonal.max() * 1e-10))
    selected = np.sort(eligible[pivots[:rank]])
    transform = np.linalg.lstsq(
        train[:, selected] / scales[selected], train / scales, rcond=1e-10
    )[0]
    audits = {}
    for split in ("train", "validation", "test"):
        x = np.vstack([x for x, e in zip(raw, examples) if e["split"] == split])
        residual = x / scales - x[:, selected] / scales[selected] @ transform
        audits[split] = {
            "max_abs_residual": float(np.abs(residual).max()),
            "n_rows_residual_gt_1e_minus6": int(
                np.any(np.abs(residual) > 1e-6, axis=1).sum()
            ),
        }
    contract = {
        "architecture": architecture,
        "raw_names": names,
        "train_rms": rms.tolist(),
        "scales": scales.tolist(),
        "eligible_columns": eligible.tolist(),
        "qr_pivots": pivots.tolist(),
        "qr_abs_diagonal": diagonal.tolist(),
        "rank": rank,
        "selected_columns": selected.tolist(),
        "selected_names": [names[i] for i in selected],
        "omitted_names": [n for i, n in enumerate(names) if i not in selected],
        "exposure_audit": audits,
        "training_projection": transform.tolist(),
    }
    matrices = [x[:, selected] / scales[selected] for x in raw]
    return matrices, contract
