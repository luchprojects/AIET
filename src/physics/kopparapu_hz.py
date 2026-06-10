"""
Kopparapu et al. (2013) habitable-zone boundaries (conservative pair).

Reference: Kopparapu et al. 2013, ApJ, 765, 131.
https://doi.org/10.1088/0004-637X/765/2/131

Uses Table 3 coefficients with Eq. (2)-(3):
  T* = T_eff - 5780 K
  S_eff = S_eff_sun + a*T* + b*T*^2 + c*T*^3 + d*T*^4
  r_AU = sqrt(L/L_sun / S_eff)

Conservative limits implemented here:
  - Inner: Recent Venus (water-loss / recent Venus limit)
  - Outer: Early Mars (maximum CO2 greenhouse in K13 nomenclature for outer edge)

Valid for 2600 K <= T_eff <= 7200 K per the paper; values are clamped to that range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

# Solar effective temperature used in Eq. (2) of Kopparapu et al. (2013).
T_EFF_SUN_K = 5780.0
T_EFF_MIN_K = 2600.0
T_EFF_MAX_K = 7200.0


@dataclass(frozen=True)
class _K13BoundaryCoeffs:
    """Polynomial coefficients for one HZ edge (Table 3)."""
    s_eff_sun: float
    a: float
    b: float
    c: float
    d: float


# Table 3 — Recent Venus (inner, conservative)
_RECENT_VENUS = _K13BoundaryCoeffs(
    s_eff_sun=1.7753,
    a=1.4316e-4,
    b=2.9875e-9,
    c=-7.5702e-12,
    d=-1.1635e-15,
)

# Table 3 — Early Mars (outer, conservative)
_EARLY_MARS = _K13BoundaryCoeffs(
    s_eff_sun=0.3179,
    a=5.4513e-5,
    b=1.5313e-9,
    c=-2.7786e-12,
    d=-4.8997e-16,
)


def _clamp_teff(t_eff_k: float) -> float:
    if not math.isfinite(t_eff_k):
        return T_EFF_SUN_K
    return max(T_EFF_MIN_K, min(T_EFF_MAX_K, float(t_eff_k)))


def kopparapu_s_eff(t_eff_k: float, boundary: _K13BoundaryCoeffs) -> float:
    """
    Effective insolation limit S_eff (in units of Earth's current flux) for a given T_eff.
    """
    t_star = _clamp_teff(t_eff_k) - T_EFF_SUN_K
    s = (
        boundary.s_eff_sun
        + boundary.a * t_star
        + boundary.b * t_star**2
        + boundary.c * t_star**3
        + boundary.d * t_star**4
    )
    # Guard against numerical edge cases; HZ requires positive flux.
    return max(s, 1e-6)


def kopparapu_hz_boundaries_au(
    luminosity_solar: float,
    t_eff_k: float,
) -> Tuple[float, float, float, float]:
    """
    Compute conservative Kopparapu (2013) HZ radii in AU.

    Args:
        luminosity_solar: Stellar luminosity in solar units (L/L_sun).
        t_eff_k: Stellar effective temperature in Kelvin.

    Returns:
        (inner_au, outer_au, s_eff_inner, s_eff_outer)
    """
    l_star = max(float(luminosity_solar), 1e-12)
    s_inner = kopparapu_s_eff(t_eff_k, _RECENT_VENUS)
    s_outer = kopparapu_s_eff(t_eff_k, _EARLY_MARS)
    inner_au = math.sqrt(l_star / s_inner)
    outer_au = math.sqrt(l_star / s_outer)
    if outer_au < inner_au:
        outer_au = inner_au
    return inner_au, outer_au, s_inner, s_outer


def planet_in_kopparapu_hz(
    semi_major_axis_au: float,
    luminosity_solar: float,
    t_eff_k: float,
) -> bool:
    """True if orbital distance lies inside the conservative K13 HZ."""
    inner_au, outer_au, _, _ = kopparapu_hz_boundaries_au(luminosity_solar, t_eff_k)
    a = float(semi_major_axis_au)
    return inner_au <= a <= outer_au
