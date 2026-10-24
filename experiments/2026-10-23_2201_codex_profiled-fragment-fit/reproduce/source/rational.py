"""Experimental full-support triangular resolvent and IPF CRAM exponential.

CRAM numerical coefficients/formulation: OpenMC official implementation,
https://docs.openmc.org/en/stable/_modules/openmc/deplete/cram.html
and Pusa (2016), https://doi.org/10.13182/NSE15-26.
This is a numerical candidate, not an error-certified or biological model.
"""

import torch


ALPHA16 = [
    5.464930576870210e3 - 3.797983575308356e4j,
    9.045112476907548e1 - 1.115537522430261e3j,
    2.344818070467641e2 - 4.228020157070496e2j,
    9.453304067358312e1 - 2.951294291446048e2j,
    7.283792954673409e2 - 1.205646080220011e5j,
    3.648229059594851e1 - 1.155509621409682e2j,
    2.547321630156819e1 - 2.639500283021502e1j,
    2.394538338734709e1 - 5.650522971778156e0j,
]
THETA16 = [
    3.509103608414918 + 8.436198985884374j,
    5.948152268951177 + 3.587457362018322j,
    -5.264971343442647 + 16.22022147316793j,
    1.419375897185666 + 10.92536348449672j,
    6.416177699099435 + 1.194122393370139j,
    4.993174737717997 + 5.996881713603942j,
    -1.413928462488886 + 13.49772569889275j,
    -10.84391707869699 + 19.27744616718165j,
]
ALPHA48 = [
    complex(a, b)
    for a, b in zip(
        [
            6.387380733878774e2,
            1.909896179065730e2,
            4.236195226571914e2,
            4.645770595258726e2,
            7.765163276752433e2,
            1.907115136768522e3,
            2.909892685603256e3,
            1.944772206620450e2,
            1.382799786972332e5,
            5.628442079602433e3,
            2.151681283794220e2,
            1.324720240514420e3,
            1.617548476343347e4,
            1.112729040439685e2,
            1.074624783191125e2,
            8.835727765158191e1,
            9.354078136054179e1,
            9.418142823531573e1,
            1.040012390717851e2,
            6.861882624343235e1,
            8.766654491283722e1,
            1.056007619389650e2,
            7.738987569039419e1,
            1.041366366475571e2,
        ],
        [
            -6.743912502859256e2,
            -3.973203432721332e2,
            -2.041233768918671e3,
            -1.652917287299683e3,
            -1.783617639907328e4,
            -5.887068595142284e4,
            -9.953255345514560e3,
            -1.427131226068449e3,
            -3.256885197214938e6,
            -2.924284515884309e4,
            -1.121774011188224e3,
            -6.370088443140973e4,
            -1.008798413156542e6,
            -8.837109731680418e1,
            -1.457246116408180e2,
            -6.388286188419360e1,
            -2.195424319460237e2,
            -6.719055740098035e2,
            -1.693747595553868e2,
            -1.177598523430493e1,
            -4.596464999363902e3,
            -1.738294585524067e3,
            -4.311715386228984e1,
            -2.777743732451969e2,
        ],
    )
]
THETA48 = [
    complex(a, b)
    for a, b in zip(
        [
            -4.465731934165702e1,
            -5.284616241568964,
            -8.867715667624458,
            3.493013124279215,
            1.564102508858634e1,
            1.742097597385893e1,
            -2.834466755180654e1,
            1.661569367939544e1,
            8.011836167974721,
            -2.056267541998229,
            1.449208170441839e1,
            1.853807176907916e1,
            9.932562704505182,
            -2.244223871767187e1,
            8.590014121680897e-1,
            -1.286192925744479e1,
            1.164596909542055e1,
            1.806076684783089e1,
            5.870672154659249,
            -3.542938819659747e1,
            1.901323489060250e1,
            1.885508331552577e1,
            -1.734689708174982e1,
            1.316284237125190e1,
        ],
        [
            6.233225190695437e1,
            4.057499381311059e1,
            4.325515754166724e1,
            3.281615453173585e1,
            1.558061616372237e1,
            1.076629305714420e1,
            5.492841024648724e1,
            1.316994930024688e1,
            2.780232111309410e1,
            3.794824788914354e1,
            1.799988210051809e1,
            5.974332563100539,
            2.532823409972962e1,
            5.179633600312162e1,
            3.536456194294350e1,
            4.600304902833652e1,
            2.287153304140217e1,
            8.368200580099821,
            3.029700159040121e1,
            5.834381701800013e1,
            1.194282058271408,
            3.583428564427879,
            4.883941101108207e1,
            2.042951874827759e1,
        ],
    )
]


def shifted_solve(log_rates, eta, rhs, shift, dt=1.0, anchor=20.0):
    """Solve (shift I - dt A)x=rhs by decreasing fragment length, O(L²)."""
    size = log_rates.numel() + 1
    if rhs.shape != (size, size) or log_rates.dtype != torch.float64:
        raise ValueError("Expected full square RHS and float64 bond log-rates")
    rates = log_rates.exp()
    prefix = torch.cat((rates.new_zeros(1), rates.cumsum(0)))
    left = torch.cat((rates.new_zeros(1), rates))
    right = torch.cat((rates, rates.new_zeros(1)))
    row = torch.zeros(size, dtype=torch.complex128, device=rhs.device)
    column = row.clone()
    values, indices = [], []
    for length in range(size, 0, -1):
        starts = torch.arange(size - length + 1, device=rhs.device)
        ends = starts + length - 1
        multiplier = torch.exp(eta * eta.new_tensor(length / anchor).log())
        exit_rates = multiplier * (prefix[ends] - prefix[starts])
        gains = right[ends] * row[starts] + left[starts] * column[ends]
        value = (rhs[starts, ends] + dt * gains) / (shift + dt * exit_rates)
        weighted = multiplier * value
        n = len(starts)
        row = torch.cat((row[:n] + weighted, row[n:]))
        column = torch.cat((column[: length - 1], column[length - 1 :] + weighted))
        values.append(value)
        indices.append(starts * size + ends)
    return (
        row.new_zeros(size * size)
        .scatter(0, torch.cat(indices), torch.cat(values))
        .reshape(size, size)
    )


def cram_step(log_rates, eta, rhs, dt, *, order=48):
    if order == 16:
        alphas, poles, final = ALPHA16, THETA16, 2.124853710495224e-16
    elif order == 48:
        alphas, poles, final = ALPHA48, THETA48, 2.258038182743983e-47
    else:
        raise ValueError("Only fixed orders 16 and 48 are registered")
    if dt == 0:
        return rhs + 0 * (log_rates.sum() + eta)
    y = rhs
    for alpha, pole in zip(alphas, poles):
        y = y - 2 * (alpha * shifted_solve(log_rates, eta, y, pole, dt)).real
    return final * y


def expectations(log_rates, eta, time, *, order=48, substeps=1):
    size = log_rates.numel() + 1
    y = log_rates.new_zeros((size, size))
    y[0, -1] = 1
    for _ in range(substeps):
        y = cram_step(log_rates, eta, y, time / substeps, order=order)
    return y
