"""
Exoplanet detection-method observables for the Detection panel.

Pure functions only; no pygame, no simulation state. Everything is derived from
the parameters a planet and its host star already carry in the sandbox
(planet mass / radius / semi-major axis / period / eccentricity, star mass /
radius). The panel uses these numbers for *visualization and education*; they
never feed back into the simulation.

Provenance
----------
- Radial-velocity semi-amplitude: standard two-body result
  K = (2*pi*G / P)^(1/3) * M_p sin i / (M_* + M_p)^(2/3) / sqrt(1 - e^2)
  (e.g. Lovis & Fischer 2010, "Radial Velocity Techniques for Exoplanets").
- Keplerian RV curve: v_r = K [cos(nu + omega) + e cos(omega)].
- Doppler shift: exact special-relativistic formula, ported from the
  Hubble Doppler Sonifier (lambda_obs = lambda_rest * sqrt((1+beta)/(1-beta))).
- Transit depth / probability / duration / impact parameter: Winn (2010),
  "Transits and Occultations", eqs. 7-14; Seager & Mallen-Ornelas (2003).
- Uniform-source transit light curve: Mandel & Agol (2002) small-body-agnostic
  overlap of two discs (no limb darkening).
- Projection geometry (sky-plane position vs. inclination): ported from the
  Orbital Inclination widget, re-expressed in the astronomical convention where
  i = 90 deg is edge-on.
- Instrument thresholds are order-of-magnitude reference points quoted in the
  panel as such (HARPS ~1 m/s, ESPRESSO ~0.1 m/s, Kepler ~20 ppm, TESS ~60 ppm).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Constants (SI)
# ---------------------------------------------------------------------------

G_SI = 6.67430e-11
M_SUN_KG = 1.98847e30
M_EARTH_KG = 5.9722e24
M_JUP_KG = 1.89813e27
R_SUN_M = 6.957e8
R_EARTH_M = 6.371e6
R_JUP_M = 7.1492e7
AU_M = 1.495978707e11
C_M_S = 2.99792458e8
DAY_S = 86400.0
PC_M = 3.0856775814913673e16

# Reference lines used for the Doppler strip (rest wavelengths, nm). The Sonifier's
# emission set plus the Fraunhofer lines already used by the Star Data spectrum tab.
DOPPLER_LINES: Sequence[Tuple[float, str]] = (
    (393.4, "Ca K"),
    (434.0, "Hγ"),
    (486.1, "Hβ"),
    (517.0, "Mg b"),
    (589.0, "Na D"),
    (656.3, "Hα"),
)

H_ALPHA_NM = 656.3


# ---------------------------------------------------------------------------
# Orbital mechanics helpers
# ---------------------------------------------------------------------------

def kepler_period_days(a_au: float, m_star_solar: float, m_planet_earth: float = 0.0) -> float:
    """Orbital period from Kepler's third law (days)."""
    a = max(1e-6, float(a_au)) * AU_M
    m = max(1e-6, float(m_star_solar)) * M_SUN_KG + max(0.0, float(m_planet_earth)) * M_EARTH_KG
    return 2.0 * math.pi * math.sqrt(a ** 3 / (G_SI * m)) / DAY_S


def kepler_semi_major_axis_au(period_days: float, m_star_solar: float, m_planet_earth: float = 0.0) -> float:
    p = max(1e-6, float(period_days)) * DAY_S
    m = max(1e-6, float(m_star_solar)) * M_SUN_KG + max(0.0, float(m_planet_earth)) * M_EARTH_KG
    return (G_SI * m * p * p / (4.0 * math.pi * math.pi)) ** (1.0 / 3.0) / AU_M


def solve_kepler(mean_anomaly: float, e: float) -> float:
    """Eccentric anomaly E from mean anomaly M (radians) by Newton iteration."""
    e = max(0.0, min(0.95, float(e)))
    m = math.fmod(mean_anomaly, 2.0 * math.pi)
    if m < 0:
        m += 2.0 * math.pi
    ecc_anom = m if e < 0.8 else math.pi
    for _ in range(30):
        f = ecc_anom - e * math.sin(ecc_anom) - m
        fp = 1.0 - e * math.cos(ecc_anom)
        step = f / max(fp, 1e-12)
        ecc_anom -= step
        if abs(step) < 1e-12:
            break
    return ecc_anom


def true_anomaly(mean_anomaly: float, e: float) -> float:
    ecc_anom = solve_kepler(mean_anomaly, e)
    e = max(0.0, min(0.95, float(e)))
    return 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(ecc_anom / 2.0),
                            math.sqrt(1.0 - e) * math.cos(ecc_anom / 2.0))


def orbital_radius(a: float, e: float, nu: float) -> float:
    e = max(0.0, min(0.95, float(e)))
    return a * (1.0 - e * e) / (1.0 + e * math.cos(nu))


# ---------------------------------------------------------------------------
# Radial velocity
# ---------------------------------------------------------------------------

def rv_semi_amplitude_m_s(m_planet_earth: float, m_star_solar: float, period_days: float,
                          e: float = 0.0, inc_deg: float = 90.0) -> float:
    """Stellar reflex-velocity semi-amplitude K in m/s."""
    mp = max(0.0, float(m_planet_earth)) * M_EARTH_KG
    ms = max(1e-6, float(m_star_solar)) * M_SUN_KG
    p = max(1e-3, float(period_days)) * DAY_S
    e = max(0.0, min(0.95, float(e)))
    sin_i = math.sin(math.radians(max(0.0, min(90.0, inc_deg))))
    return (2.0 * math.pi * G_SI / p) ** (1.0 / 3.0) * mp * sin_i / (ms + mp) ** (2.0 / 3.0) / math.sqrt(1.0 - e * e)


def rv_at_mean_anomaly(mean_anomaly: float, k_m_s: float, e: float = 0.0, omega_deg: float = 90.0) -> float:
    """Keplerian radial velocity (m/s, positive = receding/redshift) at mean anomaly M."""
    nu = true_anomaly(mean_anomaly, e)
    w = math.radians(omega_deg)
    e = max(0.0, min(0.95, float(e)))
    return k_m_s * (math.cos(nu + w) + e * math.cos(w))


def rv_curve(k_m_s: float, e: float = 0.0, omega_deg: float = 90.0, n: int = 240) -> List[Tuple[float, float]]:
    """[(orbital phase 0..1 from periastron, v_r m/s)]."""
    n = max(8, int(n))
    return [(i / (n - 1), rv_at_mean_anomaly(2.0 * math.pi * i / (n - 1), k_m_s, e, omega_deg)) for i in range(n)]


def minimum_mass_earth(k_m_s: float, m_star_solar: float, period_days: float, e: float = 0.0) -> float:
    """M_p sin i implied by a measured K (inverse of rv_semi_amplitude for M_p << M_*)."""
    ms = max(1e-6, float(m_star_solar)) * M_SUN_KG
    p = max(1e-3, float(period_days)) * DAY_S
    e = max(0.0, min(0.95, float(e)))
    mp = k_m_s * math.sqrt(1.0 - e * e) * ms ** (2.0 / 3.0) / (2.0 * math.pi * G_SI / p) ** (1.0 / 3.0)
    return mp / M_EARTH_KG


# ---------------------------------------------------------------------------
# Doppler shift
# ---------------------------------------------------------------------------

def doppler_wavelength_nm(lambda_rest_nm: float, v_m_s: float) -> float:
    """Exact relativistic Doppler shift (radial motion). Positive v = receding = redshift."""
    beta = max(-0.999, min(0.999, float(v_m_s) / C_M_S))
    return float(lambda_rest_nm) * math.sqrt((1.0 + beta) / (1.0 - beta))


def doppler_shift_nm(lambda_rest_nm: float, v_m_s: float) -> float:
    return doppler_wavelength_nm(lambda_rest_nm, v_m_s) - float(lambda_rest_nm)


def redshift_z(v_m_s: float) -> float:
    return doppler_wavelength_nm(1.0, v_m_s) - 1.0


# ---------------------------------------------------------------------------
# Transits
# ---------------------------------------------------------------------------

def radius_ratio(r_planet_earth: float, r_star_solar: float) -> float:
    return max(0.0, float(r_planet_earth)) * R_EARTH_M / (max(1e-6, float(r_star_solar)) * R_SUN_M)


def transit_depth(r_planet_earth: float, r_star_solar: float) -> float:
    """Fractional flux drop (R_p / R_*)^2 for a uniform stellar disc."""
    k = radius_ratio(r_planet_earth, r_star_solar)
    return k * k


def a_over_rstar(a_au: float, r_star_solar: float) -> float:
    return max(1e-6, float(a_au)) * AU_M / (max(1e-6, float(r_star_solar)) * R_SUN_M)


def impact_parameter(a_au: float, r_star_solar: float, inc_deg: float, e: float = 0.0, omega_deg: float = 90.0) -> float:
    """b = (a cos i / R_*) * (1 - e^2) / (1 + e sin omega)  (Winn 2010, eq. 7)."""
    e = max(0.0, min(0.95, float(e)))
    w = math.radians(omega_deg)
    return a_over_rstar(a_au, r_star_solar) * math.cos(math.radians(inc_deg)) * (1.0 - e * e) / (1.0 + e * math.sin(w))


def transit_probability(r_star_solar: float, a_au: float, r_planet_earth: float = 0.0, e: float = 0.0) -> float:
    """Geometric probability that a randomly oriented orbit transits (Winn 2010, eq. 9)."""
    e = max(0.0, min(0.95, float(e)))
    rs = max(1e-6, float(r_star_solar)) * R_SUN_M
    rp = max(0.0, float(r_planet_earth)) * R_EARTH_M
    a = max(1e-6, float(a_au)) * AU_M
    return max(0.0, min(1.0, (rs + rp) / a / (1.0 - e * e)))


def min_transit_inclination_deg(r_star_solar: float, a_au: float, r_planet_earth: float = 0.0) -> float:
    """Smallest inclination (deg) that still yields a grazing transit for a circular orbit."""
    rs = max(1e-6, float(r_star_solar)) * R_SUN_M
    rp = max(0.0, float(r_planet_earth)) * R_EARTH_M
    a = max(1e-6, float(a_au)) * AU_M
    ratio = min(1.0, (rs + rp) / a)
    return math.degrees(math.acos(ratio))


def transit_duration_hours(period_days: float, a_au: float, r_star_solar: float, r_planet_earth: float,
                           inc_deg: float, e: float = 0.0, omega_deg: float = 90.0) -> float:
    """Total (first-to-fourth contact) duration in hours; 0 if no transit (Winn 2010, eq. 14)."""
    k = radius_ratio(r_planet_earth, r_star_solar)
    b = abs(impact_parameter(a_au, r_star_solar, inc_deg, e, omega_deg))
    if b >= 1.0 + k:
        return 0.0
    ar = a_over_rstar(a_au, r_star_solar)
    sin_i = max(1e-9, math.sin(math.radians(inc_deg)))
    arg = math.sqrt(max(0.0, (1.0 + k) ** 2 - b * b)) / (ar * sin_i)
    arg = min(1.0, arg)
    e = max(0.0, min(0.95, float(e)))
    w = math.radians(omega_deg)
    ecc_factor = math.sqrt(1.0 - e * e) / (1.0 + e * math.sin(w))
    return float(period_days) * 24.0 / math.pi * math.asin(arg) * ecc_factor


def uniform_disc_blocked_fraction(d_over_rstar: float, k: float) -> float:
    """Fraction of a uniform stellar disc hidden by a planet of radius k (in R_*) whose centre
    is d (in R_*) from the star's centre. Mandel & Agol (2002), uniform source."""
    d = abs(float(d_over_rstar))
    k = max(0.0, float(k))
    if d >= 1.0 + k:
        return 0.0
    if d <= 1.0 - k:
        return k * k
    if k >= 1.0 and d <= k - 1.0:
        return 1.0
    # partial overlap: lens area of two circles (radii 1 and k, separation d)
    k0 = math.acos(max(-1.0, min(1.0, (k * k + d * d - 1.0) / (2.0 * k * d))))
    k1 = math.acos(max(-1.0, min(1.0, (1.0 - k * k + d * d) / (2.0 * d))))
    root = math.sqrt(max(0.0, (4.0 * d * d - (1.0 + d * d - k * k) ** 2) / 4.0))
    area = k * k * k0 + k1 - root
    return max(0.0, min(1.0, area / math.pi))


def sky_position(mean_anomaly: float, a_over_rs: float, inc_deg: float, e: float = 0.0,
                 omega_deg: float = 90.0) -> Tuple[float, float, float, float]:
    """Planet position in stellar radii for an observer on +z.

    Returns (x, y, z, nu): x along the sky (line of nodes), y up the sky, z toward the
    observer (positive = in front of the star), nu = true anomaly. Inclination follows
    the astronomical convention (90 deg = edge-on). With omega = 90 deg, mid-transit is
    at periastron (mean anomaly 0)."""
    e = max(0.0, min(0.95, float(e)))
    nu = true_anomaly(mean_anomaly, e)
    r = orbital_radius(a_over_rs, e, nu)
    w = math.radians(omega_deg)
    ang = nu + w
    inc = math.radians(inc_deg)
    x_orb = r * math.cos(ang)
    y_orb = r * math.sin(ang)
    # Rotate the orbital plane about the x (nodes) axis by the inclination.
    return x_orb, y_orb * math.cos(inc), y_orb * math.sin(inc), nu


def transit_light_curve_hours(period_days: float, a_au: float, r_star_solar: float, r_planet_earth: float,
                              inc_deg: float, e: float = 0.0, omega_deg: float = 90.0,
                              half_window_hours: Optional[float] = None, n: int = 241) -> Tuple[List[Tuple[float, float]], float]:
    """Relative flux vs time (hours from mid-transit) around the transit. Returns (points, half_window)."""
    k = radius_ratio(r_planet_earth, r_star_solar)
    ar = a_over_rstar(a_au, r_star_solar)
    dur = transit_duration_hours(period_days, a_au, r_star_solar, r_planet_earth, inc_deg, e, omega_deg)
    if half_window_hours is None:
        # Fall back to the edge-on duration so a non-transiting geometry still shows a flat curve
        ref = dur if dur > 0 else transit_duration_hours(period_days, a_au, r_star_solar, r_planet_earth, 90.0, e, omega_deg)
        half_window_hours = max(0.5, 1.6 * max(ref, 0.25))
    p_hours = max(1e-6, float(period_days)) * 24.0
    pts: List[Tuple[float, float]] = []
    n = max(9, int(n))
    for i in range(n):
        t = -half_window_hours + 2.0 * half_window_hours * i / (n - 1)
        mean_anom = 2.0 * math.pi * t / p_hours
        x, y, z, _ = sky_position(mean_anom, ar, inc_deg, e, omega_deg)
        if z > 0:
            d = math.hypot(x, y)
            flux = 1.0 - uniform_disc_blocked_fraction(d, k)
        else:
            flux = 1.0
        pts.append((t, flux))
    return pts, half_window_hours


# ---------------------------------------------------------------------------
# Other methods (single-number readouts)
# ---------------------------------------------------------------------------

def astrometric_wobble_uas(m_planet_earth: float, m_star_solar: float, a_au: float, distance_pc: float = 10.0) -> float:
    """Angular semi-amplitude of the star's reflex orbit in micro-arcseconds."""
    mp = max(0.0, float(m_planet_earth)) * M_EARTH_KG
    ms = max(1e-6, float(m_star_solar)) * M_SUN_KG
    a_star_au = float(a_au) * mp / (ms + mp)
    return a_star_au / max(1e-3, float(distance_pc)) * 1e6  # 1 AU at 1 pc = 1 arcsec


def reflected_light_contrast(r_planet_earth: float, a_au: float, geometric_albedo: float = 0.3) -> float:
    """Planet/star flux ratio at full phase: A_g (R_p / a)^2."""
    rp = max(0.0, float(r_planet_earth)) * R_EARTH_M
    a = max(1e-6, float(a_au)) * AU_M
    return max(0.0, float(geometric_albedo)) * (rp / a) ** 2


def angular_separation_mas(a_au: float, distance_pc: float = 10.0) -> float:
    return float(a_au) / max(1e-3, float(distance_pc)) * 1e3


def light_travel_time_s(distance_au: float) -> float:
    return float(distance_au) * AU_M / C_M_S


# ---------------------------------------------------------------------------
# Reference thresholds (order-of-magnitude, quoted as such in the UI)
# ---------------------------------------------------------------------------

RV_THRESHOLDS_M_S: Sequence[Tuple[float, str]] = (
    (0.1, "ESPRESSO-class (~0.1 m/s goal)"),
    (1.0, "HARPS-class (~1 m/s)"),
    (3.0, "HIRES-class (~3 m/s)"),
    (15.0, "1990s precision (~15 m/s)"),
)

TRANSIT_THRESHOLDS_PPM: Sequence[Tuple[float, str]] = (
    (20.0, "Kepler-class (~20 ppm)"),
    (60.0, "TESS-class (~60 ppm)"),
    (1000.0, "Ground-based (~1000 ppm)"),
)

# Textbook comparison systems (Sun-hosted unless noted).
RV_REFERENCES: Sequence[Tuple[str, float]] = (
    ("Sun · Jupiter", 12.5),
    ("Sun · Earth", 0.089),
    ("51 Peg b (1995)", 56.0),
)

TRANSIT_REFERENCES_PPM: Sequence[Tuple[str, float]] = (
    ("Sun · Jupiter", 10_500.0),
    ("Sun · Earth", 84.0),
    ("TRAPPIST-1 e", 7_000.0),
)


def rv_detectability(k_m_s: float) -> Tuple[str, str]:
    """(label, tone) where tone in {'good','ok','hard'} — descriptive only."""
    if k_m_s >= 3.0:
        return "Detectable with most modern spectrographs", "good"
    if k_m_s >= 1.0:
        return "Needs HARPS-class stability (~1 m/s)", "good"
    if k_m_s >= 0.1:
        return "Needs ESPRESSO-class precision (~0.1 m/s)", "ok"
    return "Below current RV precision (stellar jitter dominates)", "hard"


def transit_detectability(depth_ppm: float) -> Tuple[str, str]:
    if depth_ppm >= 1000.0:
        return "Detectable from the ground (>1 mmag)", "good"
    if depth_ppm >= 60.0:
        return "Detectable by TESS-class space photometry", "good"
    if depth_ppm >= 20.0:
        return "Needs Kepler-class precision (~20 ppm)", "ok"
    return "Below Kepler-class precision", "hard"


# ---------------------------------------------------------------------------
# Convenience bundle for a planet + host star pair
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectionSummary:
    name: str
    mass_earth: float
    radius_earth: float
    a_au: float
    period_days: float
    eccentricity: float
    inc_deg: float
    k_m_s: float
    k_edge_on_m_s: float
    depth: float
    depth_ppm: float
    transit_prob: float
    min_inc_deg: float
    impact_b: float
    duration_hours: float
    transits: bool
    astrometry_uas_10pc: float
    reflected_contrast: float
    separation_mas_10pc: float


def _f(v, default: float) -> float:
    try:
        if v is None:
            return default
        out = float(v)
        if not math.isfinite(out):
            return default
        return out
    except Exception:
        return default


def summarize_planet(planet: Dict, star: Dict, inc_deg: float = 90.0, omega_deg: float = 90.0) -> DetectionSummary:
    m_star = max(1e-3, _f(star.get("mass"), 1.0))
    r_star = max(1e-3, _f(star.get("radius"), 1.0))
    mp = max(0.0, _f(planet.get("mass"), 1.0))
    rp = max(1e-3, _f(planet.get("actual_radius", planet.get("radius")), 1.0))
    a_au = _f(planet.get("semiMajorAxis"), 0.0)
    p_days = _f(planet.get("orbital_period"), 0.0)
    if a_au <= 0 and p_days > 0:
        a_au = kepler_semi_major_axis_au(p_days, m_star, mp)
    if a_au <= 0:
        a_au = 1.0
    if p_days <= 0:
        p_days = kepler_period_days(a_au, m_star, mp)
    e = max(0.0, min(0.95, _f(planet.get("eccentricity"), 0.0)))
    inc = max(0.0, min(90.0, float(inc_deg)))

    k = rv_semi_amplitude_m_s(mp, m_star, p_days, e, inc)
    k90 = rv_semi_amplitude_m_s(mp, m_star, p_days, e, 90.0)
    depth = transit_depth(rp, r_star)
    b = impact_parameter(a_au, r_star, inc, e, omega_deg)
    kk = radius_ratio(rp, r_star)
    dur = transit_duration_hours(p_days, a_au, r_star, rp, inc, e, omega_deg)
    return DetectionSummary(
        name=str(planet.get("display_name") or planet.get("name") or "Planet"),
        mass_earth=mp, radius_earth=rp, a_au=a_au, period_days=p_days, eccentricity=e, inc_deg=inc,
        k_m_s=k, k_edge_on_m_s=k90, depth=depth, depth_ppm=depth * 1e6,
        transit_prob=transit_probability(r_star, a_au, rp, e),
        min_inc_deg=min_transit_inclination_deg(r_star, a_au, rp),
        impact_b=b, duration_hours=dur, transits=abs(b) < 1.0 + kk and dur > 0,
        astrometry_uas_10pc=astrometric_wobble_uas(mp, m_star, a_au, 10.0),
        reflected_contrast=reflected_light_contrast(rp, a_au),
        separation_mas_10pc=angular_separation_mas(a_au, 10.0),
    )


def format_velocity(v_m_s: float) -> str:
    a = abs(v_m_s)
    if a >= 1000.0:
        return f"{v_m_s / 1000.0:,.2f} km/s"
    if a >= 10.0:
        return f"{v_m_s:,.1f} m/s"
    if a >= 0.1:
        return f"{v_m_s:.2f} m/s"
    return f"{v_m_s * 100.0:.2f} cm/s"


def format_depth(depth: float) -> str:
    ppm = depth * 1e6
    if ppm >= 10_000:
        return f"{depth * 100:.2f} %"
    if ppm >= 1000:
        return f"{ppm / 1000:.2f} ppt"
    return f"{ppm:,.0f} ppm"


def format_duration_hours(h: float) -> str:
    if h <= 0:
        return "no transit"
    if h < 1.0:
        return f"{h * 60:.0f} min"
    if h < 48.0:
        return f"{h:.1f} h"
    return f"{h / 24.0:.1f} d"
