"""
accrual.py -- the real-world accrual factor of a premium paid at t until retirement T.

    A(t, T) = E^P_t[ exp( int_t^T (r_u + s) du ) ]

in closed form, as a function of the short rate r_t at payment (so a reward that
uses it is known when it is paid). For a Gaussian short rate the integral
I = int_t^T r_u du is normal, and with tau = T - t, B(tau) = (1 - e^{-kappa tau}) / kappa:

    var(I)  = (sigma^2 / kappa^2) (tau - B(tau)) - (sigma^2 / (2 kappa)) B(tau)^2
    A(t, T) = exp(mean(I) + var(I) / 2 + s tau)

    vasicek_short   r reverts to theta_P under P:
                    mean(I) = theta_P tau + (r_t - theta_P) B(tau)
    hull_white_p    r = x + alpha(t), x reverts to m = sigma phi / kappa under P:
                    mean(I) = int_t^T alpha(u) du + m tau + (x_t - m) B(tau)
                    int alpha = ln P^M(0,t) - ln P^M(0,T)
                                + sigma^2/(2 kappa^2) int_t^T (1 - e^{-kappa u})^2 du

    vasicek         the modelled rate is the 10Y y (theta, kappa, sigma under P); the
                    short rate is r = y - spread, so with I_y = int_t^T y du:
                    mean(I_y) = theta tau + (y_t - theta) B(tau),
                    A(t, T) = exp(mean(I_y) + var(I_y)/2 + (s - spread) tau)

hull_white has none: its short rate is risk-neutral (use hull_white_p).
"""
import numpy as np


def _B(kappa, tau):
    return (1 - np.exp(-kappa * tau)) / kappa


def _var_integral(kappa, sigma, tau):
    B = _B(kappa, tau)
    return (sigma**2 / kappa**2) * (tau - B) - (sigma**2 / (2 * kappa)) * B**2


def hull_white_alpha(t, kappa, sigma, curve):
    """alpha(t) = f(0,t) + sigma^2/(2 kappa^2) (1 - e^{-kappa t})^2."""
    from pension.rates.calibration import nss_forward
    t = np.asarray(t, float)
    return nss_forward(np.maximum(t, 0.0), *curve["params"]) \
        + (sigma**2 / (2 * kappa**2)) * (1 - np.exp(-kappa * t))**2


def hull_white_alpha_integral(t, T, kappa, sigma, curve):
    """int_t^T alpha(u) du, analytic."""
    from pension.rates.pricing import _Pmarket
    prm = curve["params"]
    fwd = np.log(_Pmarket(t, prm)) - np.log(_Pmarket(T, prm))
    sq = (T - t) - 2 * (np.exp(-kappa * t) - np.exp(-kappa * T)) / kappa \
        + (np.exp(-2 * kappa * t) - np.exp(-2 * kappa * T)) / (2 * kappa)
    return fwd + (sigma**2 / (2 * kappa**2)) * sq


def closed_form_accrual(t, r_t, p, s=0.0):
    """A(t, T) for p.RATE_MODEL in {"vasicek", "vasicek_short", "hull_white_p"};
    r_t (the SHORT rate at t) scalar or array (one per path). T = p.T; the
    financing spread s is added per year."""
    from pension.economy import hull_white_p_params, vasicek_params, vasicek_short_params
    tau = p.T - t
    r_t = np.asarray(r_t, float)
    if p.RATE_MODEL == "vasicek":
        vp = vasicek_params(p)
        kappa, sigma, theta = vp["kappa"], vp["sigma"], vp["theta"]
        y_t = r_t + vp["spread"]
        mean = theta * tau + (y_t - theta) * _B(kappa, tau)
        return np.exp(mean + _var_integral(kappa, sigma, tau) / 2 + (s - vp["spread"]) * tau)
    if p.RATE_MODEL == "vasicek_short":
        vp = vasicek_short_params(p)
        kappa, sigma = vp["kappa"], vp["sigma"]
        mean = vp["theta_P"] * tau + (r_t - vp["theta_P"]) * _B(kappa, tau)
    elif p.RATE_MODEL == "hull_white_p":
        hp = hull_white_p_params(p)
        kappa, sigma, m = hp["kappa"], hp["sigma"], hp["m"]
        x_t = r_t - hull_white_alpha(t, kappa, sigma, hp["curve"])
        mean = hull_white_alpha_integral(t, p.T, kappa, sigma, hp["curve"]) \
            + m * tau + (x_t - m) * _B(kappa, tau)
    else:
        raise ValueError(f"no real-world accrual for RATE_MODEL={p.RATE_MODEL!r}: use "
                         f"'vasicek' (canonical) or 'hull_white_p'; hull_white's short rate "
                         f"is risk-neutral (hull_white_p with LONG_RATE_P=None is its phi = 0 case)")
    return np.exp(mean + _var_integral(kappa, sigma, tau) / 2 + s * tau)
