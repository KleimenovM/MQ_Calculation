import numpy as np


def log_likelihood_measurements(flux, f_l, f_p, upper_indices, model_flux):
    """

    :param flux: [erg cm-2 s-1] measured flux values
    :param f_l: [erg cm-2 s-1] lower error
    :param f_p: [erg cm-2 s-1] upper error
    :param upper_indices: upper limit indices
    :param model_flux: [erg cm-2 s-1] model flux values
    :return:
    """
    # upper limits
    delta_ul = flux[upper_indices] - model_flux[upper_indices]
    if np.sum(delta_ul < 0) > 0:
        return -np.inf

    # points
    delta_points = (model_flux - flux)[~upper_indices]
    f_l1, f_p1 = f_l[~upper_indices], f_p[~upper_indices]

    log_likelihood_minus = np.heaviside(-delta_points, 0.5) * (delta_points / f_l1)**2
    log_likelihood_plus = np.heaviside(delta_points, 0.5) * (delta_points / f_p1)**2

    return -np.sum(log_likelihood_minus + log_likelihood_plus)


def log_norm(x, mu, sigma):
    """
    Calculate the log pdf of a normal distribution.
    :param x: Value
    :param mu: normal distribution center
    :param sigma: normal distribution variance
    :return: parabolic_function - const(sigma)
    """
    return -(x - mu) ** 2 / (2 * sigma ** 2)


def full_log_norm(x, mu, sigma):
    return -(x - mu) ** 2 / (2 * sigma ** 2) - 0.5 * np.log(2 * np.pi * sigma ** 2)


def log_uniform(x, center, half_width):
    """
    Calculate the uniform distribution log pdf
    :param x: value
    :param center: center of a uniform distribution
    :param half_width: half-width of a uniform distribution
    :return: log(1/2hw) if x is in the uniform distribution, -np.inf otherwise
    """
    return 0 if abs(x - center) < half_width else - np.inf


def full_log_uniform(x, center, half_width):
    return np.log(1 / (2 * half_width)) if abs(x - center) < half_width else -np.inf
