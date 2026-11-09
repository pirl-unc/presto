"""Explicit probability/log reductions used by the registered full-parent gate."""

import math
import numpy as np
from scipy.special import logsumexp


def probability_errors(log_means, reference):
    means = np.asarray(log_means)
    truth = np.asarray(reference)
    differences = means - truth
    return (
        np.expm1(logsumexp(means, axis=0) - math.log(len(means)) - truth),
        differences.mean(0),
        np.abs(differences).mean(0),
    )


def probability_gate(relative, absolute_log_error, gates):
    return bool(
        np.max(np.abs(relative)) <= gates["maximum_mean_relative_probability_error"]
        and np.max(absolute_log_error) <= gates["maximum_mean_abs_log_error"]
    )
