"""Pure raw feature function extracted from family0143; source SHA pinned."""

import math

AA = "ACDEFGHIKLMNPQRSTVWY"


def raw_features(sequence, representation):
    assert sequence and set(sequence) <= set(AA)
    features = [math.log(len(sequence))]
    if representation == "sequence":
        features += [sequence.count(a) / len(sequence) for a in AA]
        features += [float(sequence[0] == a) for a in AA]
        features += [float(sequence[-1] == a) for a in AA]
    else:
        assert representation == "length"
    return features
