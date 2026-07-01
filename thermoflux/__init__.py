"""Small heat-transfer helpers: conduction, convection, radiation, and
LMTD-based heat-exchanger sizing. All quantities are SI (W, m, m^2, K, mol, Pa)."""

import math

SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant, W/(m^2*K^4)
R_GAS = 8.314462618  # universal gas constant, J/(mol*K)


def conduction(k, area, dT, thickness):
    """Conductive heat rate (W) through a slab via Fourier's law.

    k: thermal conductivity W/(m*K); area in m^2; dT temperature drop in K;
    thickness in m.
    """
    return k * area * dT / thickness


def convection(h, area, dT):
    """Convective heat rate (W) from Newton's law of cooling.

    h: heat-transfer coefficient W/(m^2*K); area in m^2; dT in K.
    """
    return h * area * dT


def radiation(emissivity, area, T):
    """Radiated heat rate (W) for a grey surface at absolute temperature T (K).

    emissivity is dimensionless (0-1); area in m^2.
    """
    return emissivity * SIGMA * area * T ** 4


def ideal_gas_pressure(n, T, V):
    """Pressure (Pa) from the ideal-gas law p = nRT/V.

    n: amount in mol; T in K; V volume in m^3.
    """
    return n * R_GAS * T / V


def lmtd(t_hot_in, t_hot_out, t_cold_in, t_cold_out, counterflow=True):
    """Log-mean temperature difference (K) for a two-stream exchanger.

    Temperatures in K (or consistent degrees). Set counterflow=False for a
    parallel-flow arrangement. When the two end differences are equal the
    LMTD reduces to that common value.
    """
    if counterflow:
        d1 = t_hot_in - t_cold_out
        d2 = t_hot_out - t_cold_in
    else:
        d1 = t_hot_in - t_cold_in
        d2 = t_hot_out - t_cold_out
    if d1 == d2:
        return d1
    return (d1 - d2) / math.log(d1 / d2)


def exchanger_area(Q, U, lmtd_value):
    """Required heat-transfer area (m^2) for duty Q (W).

    U: overall heat-transfer coefficient W/(m^2*K); lmtd_value in K.
    """
    return Q / (U * lmtd_value)


def fin_efficiency(h, k, thickness, length):
    """Efficiency (dimensionless, 0-1) of a straight rectangular fin.

    h: heat-transfer coefficient W/(m^2*K); k thermal conductivity W/(m*K);
    thickness and length in m. As the fin parameter m*length approaches zero
    (e.g. zero length or no convection) the efficiency tends to 1.
    """
    m = math.sqrt(2 * h / (k * thickness))
    mL = m * length
    if mL == 0:
        return 1.0
    return math.tanh(mL) / mL
