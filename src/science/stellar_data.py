"""
Stellar data model for the Star Data panel (spectra, H-R diagram, evolution).

Pure functions only; no pygame, no simulation state. Everything here is
derived from a star's basic parameters (mass, radius, T_eff, luminosity, age)
and is used for *visualization and education*, not to drive the simulation.

Provenance of the models
------------------------
- Planck / Wien: standard blackbody physics (idealized photosphere).
- Spectral class boundaries: Harvard sequence temperature cuts (Gray & Corbally).
- T_eff -> RGB: piecewise approximation of blackbody chromaticity (display only).
- Main-sequence locus: canonical (T_eff, log L) points (Sun = 5778 K, 0.0).
- Evolutionary anchors: schematic MIST/Padova-inspired checkpoints
  (Choi et al. 2016; Bressan et al. 2012) ported from the Stellar Nursery
  lifecycle lab. They are interpolated in log mass and are *illustrative*,
  which the UI states explicitly.
- Kopparapu HZ: reuses AIET's existing physics implementation (read-only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from src.physics.kopparapu_hz import kopparapu_hz_boundaries_au
except Exception:  # pragma: no cover - keeps the panel usable if physics import fails
    kopparapu_hz_boundaries_au = None  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

H_PLANCK = 6.62607015e-34      # J s
C_LIGHT = 2.99792458e8         # m/s
K_BOLTZMANN = 1.380649e-23     # J/K
WIEN_B_NM_K = 2.897771955e6    # nm K
T_SUN_K = 5778.0
R_SUN_KM = 695_700.0
LOG_G_SUN_CGS = 4.438          # log10(cm s^-2)
RHO_SUN_G_CM3 = 1.408
V_ESC_SUN_KM_S = 617.7
M_BOL_SUN = 4.74

# Phase boundaries on the normalized 0..1 lifecycle timeline.
PHASE_PROTO_END = 0.05
PHASE_MS_END = 0.60
PHASE_GIANT_END = 0.95


# ---------------------------------------------------------------------------
# Blackbody spectrum
# ---------------------------------------------------------------------------

def planck_spectral_radiance(wavelength_nm: float, t_eff_k: float) -> float:
    """Planck B_lambda in W sr^-1 m^-3 for wavelength in nm and temperature in K."""
    lam = max(wavelength_nm, 1e-3) * 1e-9
    t = max(t_eff_k, 1.0)
    x = (H_PLANCK * C_LIGHT) / (lam * K_BOLTZMANN * t)
    if x > 700.0:
        return 0.0
    return (2.0 * H_PLANCK * C_LIGHT * C_LIGHT) / (lam ** 5 * (math.exp(x) - 1.0))


def wien_peak_nm(t_eff_k: float) -> float:
    """Wavelength of peak spectral radiance (Wien displacement law), in nm."""
    return WIEN_B_NM_K / max(t_eff_k, 1.0)


def spectrum_samples(
    t_eff_k: float,
    lam_min_nm: float = 100.0,
    lam_max_nm: float = 1300.0,
    n: int = 280,
    normalize: bool = True,
) -> List[Tuple[float, float]]:
    """Sample the Planck curve. Returns [(lambda_nm, value)], optionally peak-normalized."""
    if n < 2:
        n = 2
    pts = []
    step = (lam_max_nm - lam_min_nm) / (n - 1)
    for i in range(n):
        lam = lam_min_nm + i * step
        pts.append((lam, planck_spectral_radiance(lam, t_eff_k)))
    if normalize:
        peak = max(v for _, v in pts) or 1.0
        pts = [(lam, v / peak) for lam, v in pts]
    return pts


def band_fraction(t_eff_k: float, lam_lo_nm: float, lam_hi_nm: float) -> float:
    """Fraction of total blackbody flux emitted between two wavelengths (numerical)."""
    lo = max(lam_lo_nm, 1.0)
    hi = max(lam_hi_nm, lo + 1e-6)
    # Integrate on a log grid spanning essentially all of the emission.
    def integrate(a: float, b: float, steps: int = 600) -> float:
        la, lb = math.log(a), math.log(b)
        total = 0.0
        prev_lam = math.exp(la)
        prev_v = planck_spectral_radiance(prev_lam, t_eff_k)
        for i in range(1, steps + 1):
            lam = math.exp(la + (lb - la) * i / steps)
            v = planck_spectral_radiance(lam, t_eff_k)
            total += 0.5 * (v + prev_v) * (lam - prev_lam)
            prev_lam, prev_v = lam, v
        return total
    total = integrate(10.0, 2.0e5)
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, integrate(lo, hi) / total))


# Reference absorption features (rest wavelengths, nm). Positions only; the
# panel does not model line depths.
SPECTRAL_LINES: Sequence[Tuple[float, str, str]] = (
    (121.6, "Lyα", "H I"),
    (393.4, "Ca K", "Ca II"),
    (396.8, "Ca H", "Ca II"),
    (434.0, "Hγ", "H I"),
    (486.1, "Hβ", "H I"),
    (517.0, "Mg b", "Mg I"),
    (589.0, "Na D", "Na I"),
    (656.3, "Hα", "H I"),
    (854.2, "Ca IR", "Ca II"),
)

BANDS: Sequence[Tuple[float, float, str]] = (
    (100.0, 380.0, "UV"),
    (380.0, 750.0, "Visible"),
    (750.0, 1300.0, "Near-IR"),
)


# ---------------------------------------------------------------------------
# Classification and color
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralClassInfo:
    letter: str
    t_min_k: float
    t_max_k: Optional[float]
    signature: str


SPECTRAL_CLASSES: Sequence[SpectralClassInfo] = (
    SpectralClassInfo("O", 30000.0, None, "Ionised He; intense UV continuum"),
    SpectralClassInfo("B", 10000.0, 30000.0, "Neutral He; strong Balmer lines"),
    SpectralClassInfo("A", 7500.0, 10000.0, "Balmer lines at maximum strength"),
    SpectralClassInfo("F", 6000.0, 7500.0, "Ca II H&K rising; H lines weakening"),
    SpectralClassInfo("G", 5200.0, 6000.0, "Ca II H&K, neutral metals (Fe I) — solar-like"),
    SpectralClassInfo("K", 3700.0, 5200.0, "Strong Ca I, CH and CN molecular bands"),
    SpectralClassInfo("M", 0.0, 3700.0, "TiO / VO molecular bands dominate"),
)


def spectral_class_from_temperature(t_eff_k: float) -> SpectralClassInfo:
    for info in SPECTRAL_CLASSES:
        if t_eff_k > info.t_min_k:
            return info
    return SPECTRAL_CLASSES[-1]


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def temperature_to_rgb(t_eff_k: float) -> Tuple[int, int, int]:
    """Approximate photospheric color for display. Clamped to 1500–40000 K."""
    t = max(1500.0, min(40000.0, float(t_eff_k)))
    if t < 3500:
        f = (t - 1500) / 2000
        r, g, b = 255, _lerp(40, 120, f), _lerp(0, 30, f)
    elif t < 5000:
        f = (t - 3500) / 1500
        r, g, b = 255, _lerp(120, 200, f), _lerp(30, 120, f)
    elif t < 7000:
        f = (t - 5000) / 2000
        r, g, b = 255, _lerp(200, 240, f), _lerp(120, 200, f)
    elif t < 10000:
        f = (t - 7000) / 3000
        r, g, b = 255, _lerp(240, 250, f), _lerp(200, 255, f)
    elif t < 20000:
        f = (t - 10000) / 10000
        r, g, b = _lerp(255, 195, f), _lerp(250, 210, f), 255
    else:
        f = (t - 20000) / 20000
        r, g, b = _lerp(195, 148, f), _lerp(210, 172, f), 255
    return int(round(r)), int(round(g)), int(round(b))


# ---------------------------------------------------------------------------
# H-R diagram reference data
# ---------------------------------------------------------------------------

# Zero-age-ish main-sequence locus: (T_eff K, log10 L/Lsun)
MAIN_SEQUENCE_LOCUS: Sequence[Tuple[float, float]] = (
    (50000, 5.8), (40000, 5.5), (30000, 4.9), (22000, 4.2),
    (15000, 3.5), (10000, 2.3), (8000, 1.5), (7000, 0.9),
    (6500, 0.5), (5778, 0.0), (5200, -0.3), (4500, -0.8),
    (4000, -1.2), (3500, -1.9), (3000, -2.7), (2500, -3.6),
)

# Region labels for orientation (T_eff K, log L). Text only.
HR_REGION_LABELS: Sequence[Tuple[str, float, float]] = (
    ("Supergiants", 9000, 5.0),
    ("Giants", 4300, 2.2),
    ("Main sequence", 12000, 1.4),
    ("White dwarfs", 14000, -2.6),
)


# ---------------------------------------------------------------------------
# Schematic evolution model (log-mass interpolated anchors)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvolutionAnchor:
    name: str
    t_proto: float
    t_ms: float
    t_giant: float
    t_total: float
    r_ms: float
    r_giant: float
    r_remnant: float
    temp_ms: float
    temp_giant: float
    temp_remnant: float
    logL_ms: float
    logL_giant: float
    logL_remnant: float
    endpoint: str


EVOLUTION_ANCHORS: Dict[float, EvolutionAnchor] = {
    0.08: EvolutionAnchor("Sub-stellar", 10.0e6, 1.00e12, 1.0e6, 1.00e12,
                          0.11, 0.11, 0.10, 2500, 2500, 2500, -3.8, -3.8, -3.8, "Brown Dwarf"),
    0.5: EvolutionAnchor("Low mass", 30.0e6, 56.0e9, 2.0e9, 58.05e9,
                         0.40, 50.0, 0.014, 3800, 3000, 28000, -1.5, 2.3, -1.5, "White Dwarf"),
    1.0: EvolutionAnchor("Solar mass", 50.0e6, 10.0e9, 1.5e9, 11.55e9,
                         1.00, 150.0, 0.015, 5778, 3500, 50000, 0.0, 3.5, -1.0, "White Dwarf"),
    15.0: EvolutionAnchor("Massive", 0.30e6, 11.0e6, 0.80e6, 12.1e6,
                          7.00, 800.0, 0.000014, 32000, 3300, 600000, 4.3, 5.1, 3.5, "Neutron Star"),
    25.0: EvolutionAnchor("Hypermassive", 0.10e6, 6.5e6, 0.40e6, 7.0e6,
                          10.0, 900.0, 0.0, 40000, 8000, 0, 5.2, 5.8, 0.0, "Black Hole"),
    100.0: EvolutionAnchor("Ultra-massive", 0.05e6, 2.9e6, 0.10e6, 3.05e6,
                           20.0, 60.0, 0.0, 55000, 28000, 0, 6.0, 6.4, 0.0, "Black Hole"),
}
ANCHOR_MASSES: Sequence[float] = tuple(sorted(EVOLUTION_ANCHORS.keys()))


def _log_lerp(a: float, b: float, t: float) -> float:
    la = math.log(max(a, 1e-30))
    lb = math.log(max(b, 1e-30))
    return math.exp(la + (lb - la) * t)


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


# Canonical initial-mass thresholds for the final state (textbook values;
# the exact NS/BH boundary depends on metallicity and mass loss).
HYDROGEN_BURNING_LIMIT_MSUN = 0.08
WHITE_DWARF_MAX_MSUN = 8.0
NO_GIANT_BRANCH_BELOW_MSUN = 0.25
NEUTRON_STAR_MAX_MSUN = 25.0


def end_state_for_mass(mass_solar: float) -> str:
    m = float(mass_solar)
    if m < HYDROGEN_BURNING_LIMIT_MSUN:
        return "Brown Dwarf"
    if m < WHITE_DWARF_MAX_MSUN:
        return "White Dwarf"
    if m < NEUTRON_STAR_MAX_MSUN:
        return "Neutron Star"
    return "Black Hole"


def mass_regime_name(mass_solar: float) -> str:
    m = float(mass_solar)
    if m < HYDROGEN_BURNING_LIMIT_MSUN:
        return "Sub-stellar"
    if m < 0.5:
        return "Very low mass"
    if m < 1.5:
        return "Solar-type"
    if m < 8.0:
        return "Intermediate mass"
    if m < 25.0:
        return "Massive"
    return "Very massive"


def interpolate_evolution(mass_solar: float) -> EvolutionAnchor:
    """Blend anchor tracks in log mass (smoothstepped)."""
    m = float(mass_solar)
    if m <= ANCHOR_MASSES[0] or m >= ANCHOR_MASSES[-1]:
        base = EVOLUTION_ANCHORS[ANCHOR_MASSES[0] if m <= ANCHOR_MASSES[0] else ANCHOR_MASSES[-1]]
        return EvolutionAnchor(**{**base.__dict__, "name": mass_regime_name(m), "endpoint": end_state_for_mass(m)})
    lo, hi = ANCHOR_MASSES[0], ANCHOR_MASSES[1]
    for i in range(len(ANCHOR_MASSES) - 1):
        if ANCHOR_MASSES[i] <= m <= ANCHOR_MASSES[i + 1]:
            lo, hi = ANCHOR_MASSES[i], ANCHOR_MASSES[i + 1]
            break
    a, b = EVOLUTION_ANCHORS[lo], EVOLUTION_ANCHORS[hi]
    t = (math.log(m) - math.log(lo)) / (math.log(hi) - math.log(lo))
    ts = _smoothstep(t)
    result = EvolutionAnchor(
        name=mass_regime_name(m),
        endpoint=end_state_for_mass(m),
        t_proto=_log_lerp(a.t_proto, b.t_proto, ts),
        t_ms=_log_lerp(a.t_ms, b.t_ms, ts),
        t_giant=_log_lerp(a.t_giant + 1, b.t_giant + 1, ts),
        t_total=_log_lerp(a.t_total, b.t_total, ts),
        r_ms=_lerp(a.r_ms, b.r_ms, ts),
        r_giant=_lerp(a.r_giant, b.r_giant, ts),
        r_remnant=_lerp(a.r_remnant, b.r_remnant, ts),
        temp_ms=_lerp(a.temp_ms, b.temp_ms, ts),
        temp_giant=_lerp(a.temp_giant, b.temp_giant, ts),
        temp_remnant=_lerp(a.temp_remnant, b.temp_remnant, ts),
        logL_ms=_lerp(a.logL_ms, b.logL_ms, ts),
        logL_giant=_lerp(a.logL_giant, b.logL_giant, ts),
        logL_remnant=_lerp(a.logL_remnant, b.logL_remnant, ts),
    )
    if HYDROGEN_BURNING_LIMIT_MSUN <= m < NO_GIANT_BRANCH_BELOW_MSUN:
        # Stars below ~0.25 Msun never ignite helium or climb the giant branch; they
        # contract directly into helium white dwarfs (Laughlin, Bodenheimer & Adams 1997).
        result = EvolutionAnchor(**{**result.__dict__,
                                    "r_giant": result.r_ms, "temp_giant": result.temp_ms,
                                    "logL_giant": result.logL_ms})
    return result


def has_giant_branch(sd: EvolutionAnchor) -> bool:
    return sd.r_giant > 1.5 * sd.r_ms


@dataclass(frozen=True)
class StarState:
    phase: float
    temp_k: float
    log_l: float
    radius_solar: float
    age_years: float


def compute_star_state(phase: float, sd: EvolutionAnchor) -> StarState:
    """Schematic (T, log L, R, age) along the normalized lifecycle timeline."""
    p = max(0.0, min(1.0, phase))
    if p < PHASE_PROTO_END:
        t = p / PHASE_PROTO_END
        s = _smoothstep(t)
        temp = _lerp(800, sd.temp_ms * 0.97, s)
        radius = _lerp(sd.r_ms * 5, sd.r_ms, s)
        age = t * sd.t_proto
    elif p < PHASE_MS_END:
        t = (p - PHASE_PROTO_END) / (PHASE_MS_END - PHASE_PROTO_END)
        temp = _lerp(sd.temp_ms * 0.97, sd.temp_ms * 1.06, t)
        radius = _lerp(sd.r_ms, sd.r_ms * 1.15, t)
        age = sd.t_proto + t * sd.t_ms
    elif p < PHASE_GIANT_END:
        t = (p - PHASE_MS_END) / (PHASE_GIANT_END - PHASE_MS_END)
        s = _smoothstep(t)
        temp = _lerp(sd.temp_ms * 1.06, sd.temp_giant, s)
        radius = _lerp(sd.r_ms * 1.15, sd.r_giant, s)
        age = sd.t_proto + sd.t_ms + t * sd.t_giant
    else:
        t = (p - PHASE_GIANT_END) / (1.0 - PHASE_GIANT_END)
        s = _smoothstep(t)
        age = sd.t_proto + sd.t_ms + sd.t_giant + s * (sd.t_total - sd.t_proto - sd.t_ms - sd.t_giant)
        if sd.endpoint in ("Neutron Star", "Black Hole"):
            # Core collapse: the visible track ends at the supergiant point (supernova).
            temp, radius = sd.temp_giant, sd.r_giant
        elif sd.endpoint == "Brown Dwarf":
            temp = _lerp(sd.temp_ms, max(sd.temp_remnant * 0.6, 300.0), s)
            radius = _lerp(sd.r_ms, sd.r_remnant, s)
        else:
            # Post-AGB → white dwarf: luminosity stays near the giant value while the
            # exposed core heats up (horizontal move to the blue), then the remnant
            # fades at roughly constant temperature (vertical drop). Radius follows
            # from Stefan-Boltzmann so the track stays self-consistent.
            t_hot = max(sd.temp_remnant, sd.temp_giant)
            if s < 0.55:
                f = _smoothstep(s / 0.55)
                log_t = _lerp(math.log10(sd.temp_giant), math.log10(t_hot), f)
                temp = 10 ** log_t
                log_l_target = sd.logL_giant
            else:
                f = _smoothstep((s - 0.55) / 0.45)
                temp = t_hot
                log_l_target = _lerp(sd.logL_giant, sd.logL_remnant - 1.5, f)
            radius = math.sqrt(10 ** log_l_target) / (max(temp, 500.0) / T_SUN_K) ** 2
    safe_r = max(1e-6, radius)
    safe_t = max(500.0, temp)
    # Stefan-Boltzmann: L/Lsun = (R/Rsun)^2 (T/Tsun)^4
    log_l = 2.0 * math.log10(max(0.001, safe_r)) + 4.0 * math.log10(safe_t / T_SUN_K)
    return StarState(p, safe_t, log_l, safe_r, age)


def sample_evolution_track(sd: EvolutionAnchor, n_per_phase: int = 90) -> List[StarState]:
    specs = (
        (0.0, PHASE_PROTO_END, max(20, n_per_phase // 3)),
        (PHASE_PROTO_END, PHASE_MS_END, n_per_phase),
        (PHASE_MS_END, PHASE_GIANT_END, n_per_phase),
        (PHASE_GIANT_END, 1.0, max(20, n_per_phase // 2)),
    )
    out: List[StarState] = []
    for a, b, n in specs:
        for i in range(n):
            out.append(compute_star_state(a + (b - a) * i / max(1, n - 1), sd))
    return out


def phase_from_age(age_years: float, sd: EvolutionAnchor) -> float:
    """Inverse of the age mapping in compute_star_state (piecewise)."""
    a = max(0.0, float(age_years))
    if a <= sd.t_proto:
        return PHASE_PROTO_END * (a / max(sd.t_proto, 1e-9))
    a -= sd.t_proto
    if a <= sd.t_ms:
        return PHASE_PROTO_END + (PHASE_MS_END - PHASE_PROTO_END) * (a / max(sd.t_ms, 1e-9))
    a -= sd.t_ms
    if a <= sd.t_giant:
        return PHASE_MS_END + (PHASE_GIANT_END - PHASE_MS_END) * (a / max(sd.t_giant, 1e-9))
    a -= sd.t_giant
    rem = max(sd.t_total - sd.t_proto - sd.t_ms - sd.t_giant, 1e-9)
    # smoothstep is monotonic on [0,1]; invert by bisection
    target = max(0.0, min(1.0, a / rem))
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _smoothstep(mid) < target:
            lo = mid
        else:
            hi = mid
    return PHASE_GIANT_END + (1.0 - PHASE_GIANT_END) * 0.5 * (lo + hi)


def phase_label(phase: float, endpoint: str, giant_branch: bool = True) -> str:
    if endpoint == "Brown Dwarf":
        if phase < PHASE_PROTO_END:
            return "Pre-main sequence"
        if phase < PHASE_GIANT_END:
            return "Cooling / contraction"
        return "Sub-stellar remnant"
    if phase < PHASE_PROTO_END:
        return "Pre-main sequence"
    if phase < PHASE_MS_END:
        return "Main sequence (core H fusion)"
    if phase < PHASE_GIANT_END:
        return "Giant branch (shell fusion)" if giant_branch else "Contraction (no giant phase)"
    return "Stellar remnant"


PHASE_BLURBS: Dict[str, str] = {
    "proto": "Gravitational contraction heats the core toward ~10 MK, the threshold for hydrogen fusion. No sustained nuclear burning yet.",
    "ms": "Hydrostatic equilibrium: the core fuses hydrogen into helium at 10–15 MK (CNO cycle dominates above ~1.3 Msun).",
    "nogiant": "Below ~0.25 Msun the star is fully convective and burns nearly all of its hydrogen; it never ignites helium and contracts directly toward a helium white dwarf.",
    "giant": "Core hydrogen is exhausted. The helium core contracts and heats while a shell above it fuses hydrogen; the envelope expands and cools.",
    "Brown Dwarf": "Below ~0.08 Msun the core never sustains hydrogen fusion; the object radiates away leftover gravitational heat.",
    "White Dwarf": "Electron degeneracy pressure supports the exposed C–O core. Fusion has ceased; the remnant cools over billions of years.",
    "Neutron Star": "Neutron degeneracy supports matter at nuclear density: more than a solar mass within ~20 km.",
    "Black Hole": "Core collapse continues past the neutron-star limit; nothing halts infall inside the event horizon.",
}


@dataclass(frozen=True)
class InteriorLayer:
    frac_radius: float
    label: str
    kind: str   # "core", "radiative", "convective", "shell", "envelope", "surface"


def interior_structure(mass_solar: float, phase: float, endpoint: str) -> Tuple[List[InteriorLayer], str]:
    """Schematic interior shells (fractional radius) with a one-line note."""
    m = max(0.08, float(mass_solar))
    if phase >= PHASE_GIANT_END:
        if endpoint == "Black Hole":
            return [InteriorLayer(1.0, "Event horizon", "core")], "No internal structure is observable beyond the horizon."
        if endpoint == "Neutron Star":
            return ([InteriorLayer(0.35, "Core (schematic)", "core"),
                     InteriorLayer(0.72, "Neutron-rich mantle", "radiative"),
                     InteriorLayer(1.0, "Solid crust", "surface")],
                    "Nuclear-density matter supported by neutron degeneracy pressure.")
        if endpoint == "Brown Dwarf":
            return ([InteriorLayer(0.65, "Degenerate interior", "core"),
                     InteriorLayer(1.0, "Cool atmosphere", "surface")],
                    "No sustained fusion; slowly cooling.")
        return ([InteriorLayer(0.65, "Degenerate C–O interior", "core"),
                 InteriorLayer(1.0, "Thin H/He atmosphere", "surface")],
                "Electron-degenerate remnant; all fusion has ceased.")
    if phase < PHASE_PROTO_END:
        return ([InteriorLayer(0.3, "Contracting core", "core"),
                 InteriorLayer(1.0, "Accreting envelope", "convective")],
                "Pre-main-sequence contraction; core not yet hot enough for sustained H fusion.")
    if phase < PHASE_MS_END:
        if m < 0.35:
            return ([InteriorLayer(0.18, "H-fusion core", "core"),
                     InteriorLayer(1.0, "Fully convective envelope", "convective")],
                    "Fully convective — no separate radiative zone (typical M dwarf).")
        if m < 1.5:
            return ([InteriorLayer(0.25, "H-fusion core", "core"),
                     InteriorLayer(0.715, "Radiative zone", "radiative"),
                     InteriorLayer(1.0, "Convective zone", "convective")],
                    "Solar-type: radiative interior with an outer convection zone (helioseismic boundary at 0.713 Rsun).")
        if m < 8.0:
            core_f = 0.10 + 0.05 * math.log10(m)
            return ([InteriorLayer(core_f, "Convective H core", "core"),
                     InteriorLayer(1.0, "Radiative envelope", "radiative")],
                    "Intermediate mass: convective core beneath a radiative envelope.")
        core_f = min(0.30, 0.16 + 0.035 * math.log10(m))
        return ([InteriorLayer(core_f, "Convective CNO core", "core"),
                 InteriorLayer(1.0, "Radiative envelope", "radiative")],
                "Massive O/B star: extended convective core, CNO cycle dominated.")
    t = (phase - PHASE_MS_END) / (PHASE_GIANT_END - PHASE_MS_END)
    return ([InteriorLayer(0.03 + t * 0.04, "Degenerate He / C–O core", "core"),
             InteriorLayer(0.06 + t * 0.05, "He-burning shell", "shell"),
             InteriorLayer(0.11 + t * 0.06, "H-burning shell", "shell"),
             InteriorLayer(0.80 + t * 0.12, "Convective envelope", "convective"),
             InteriorLayer(1.0, "Photosphere", "surface")],
            "Giant: compact core and thin fusion shells inside a vast convective envelope.")


# ---------------------------------------------------------------------------
# Derived quantities from the star's own parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DerivedStarQuantities:
    mass_solar: float
    radius_solar: float
    t_eff_k: float
    luminosity_solar: float
    age_gyr: float
    metallicity_feh: Optional[float]
    luminosity_sb_solar: float        # from R and T
    radius_sb_solar: float            # from L and T
    log_g_cgs: float
    mean_density_g_cm3: float
    escape_velocity_km_s: float
    abs_bol_magnitude: float
    peak_wavelength_nm: float
    ms_lifetime_gyr: float            # tau ~ 10 Gyr * M^-2.5 (same heuristic AIET already displays)
    hz_inner_au: Optional[float]
    hz_outer_au: Optional[float]
    class_from_temperature: str
    color_rgb: Tuple[int, int, int]


def derive_star_quantities(star: Dict) -> DerivedStarQuantities:
    m = max(1e-3, float(star.get("mass", 1.0) or 1.0))
    r = max(1e-4, float(star.get("radius", 1.0) or 1.0))
    t = max(500.0, float(star.get("temperature", T_SUN_K) or T_SUN_K))
    l = max(1e-9, float(star.get("luminosity", 1.0) or 1.0))
    age = float(star.get("age", 4.6) or 0.0)
    feh = star.get("metallicity", None)
    feh_val = float(feh) if feh is not None else None

    l_sb = (r ** 2) * (t / T_SUN_K) ** 4
    r_sb = math.sqrt(l) / (t / T_SUN_K) ** 2
    log_g = LOG_G_SUN_CGS + math.log10(m / (r ** 2))
    rho = RHO_SUN_G_CM3 * m / (r ** 3)
    v_esc = V_ESC_SUN_KM_S * math.sqrt(m / r)
    m_bol = M_BOL_SUN - 2.5 * math.log10(l)
    tau_ms = 10.0 * (m ** -2.5)

    hz_in = hz_out = None
    if kopparapu_hz_boundaries_au is not None:
        try:
            hz_in, hz_out, _, _ = kopparapu_hz_boundaries_au(l, t)
        except Exception:
            hz_in = hz_out = None

    return DerivedStarQuantities(
        mass_solar=m, radius_solar=r, t_eff_k=t, luminosity_solar=l, age_gyr=age,
        metallicity_feh=feh_val,
        luminosity_sb_solar=l_sb, radius_sb_solar=r_sb, log_g_cgs=log_g,
        mean_density_g_cm3=rho, escape_velocity_km_s=v_esc, abs_bol_magnitude=m_bol,
        peak_wavelength_nm=wien_peak_nm(t), ms_lifetime_gyr=tau_ms,
        hz_inner_au=hz_in, hz_outer_au=hz_out,
        class_from_temperature=spectral_class_from_temperature(t).letter,
        color_rgb=temperature_to_rgb(t),
    )


def format_years(years: float) -> str:
    y = float(years)
    if y >= 1e9:
        return f"{y / 1e9:.2f} Gyr" if y < 1e11 else f"{y / 1e9:.0f} Gyr"
    if y >= 1e6:
        return f"{y / 1e6:.1f} Myr"
    if y >= 1e3:
        return f"{y / 1e3:.0f} kyr"
    return f"{y:.0f} yr"


# ---------------------------------------------------------------------------
# Habitable zone over the star's lifetime
# ---------------------------------------------------------------------------
#
# No new physics: this evaluates AIET's existing Kopparapu et al. (2013)
# conservative boundaries (src/physics/kopparapu_hz.py) along the schematic
# L(t), T_eff(t) track defined above. Kopparapu's polynomial fits are only valid
# for 2600 K <= T_eff <= 7200 K; outside that range the physics module clamps
# T_eff, and the samples are flagged so the UI can hatch those segments.

KOPPARAPU_T_MIN_K = 2600.0
KOPPARAPU_T_MAX_K = 7200.0


@dataclass(frozen=True)
class HZTimeSample:
    age_years: float
    phase: float
    inner_au: float
    outer_au: float
    t_eff_k: float
    log_l: float
    teff_in_range: bool     # False when Kopparapu's T_eff validity range was exceeded (clamped)


def hz_over_lifetime(sd: EvolutionAnchor, n: int = 360) -> List[HZTimeSample]:
    """Sample the conservative HZ along the schematic track (uniform in lifecycle phase so the
    short pre-main-sequence and giant transitions are resolved). Empty if the HZ module is missing."""
    if kopparapu_hz_boundaries_au is None:
        return []
    n = max(16, int(n))
    out: List[HZTimeSample] = []
    for i in range(n):
        phase = i / (n - 1)
        st = compute_star_state(phase, sd)
        lum = 10.0 ** st.log_l
        try:
            inner, outer, _, _ = kopparapu_hz_boundaries_au(lum, st.temp_k)
        except Exception:
            continue
        out.append(HZTimeSample(
            age_years=st.age_years, phase=phase, inner_au=inner, outer_au=outer,
            t_eff_k=st.temp_k, log_l=st.log_l,
            teff_in_range=KOPPARAPU_T_MIN_K <= st.temp_k <= KOPPARAPU_T_MAX_K,
        ))
    return out


def hz_intervals_for_orbit(samples: Sequence[HZTimeSample], a_au: float,
                           min_duration_years: float = 0.0) -> List[Tuple[float, float]]:
    """Age intervals [(t_enter, t_exit), ...] during which an orbit at a_au lies inside the HZ band.
    Boundaries are linearly interpolated between samples; intervals shorter than
    min_duration_years (e.g. the instant the fading remnant's HZ sweeps past) are dropped."""
    a = float(a_au)
    if not samples or a <= 0:
        return []

    def inside(s: HZTimeSample) -> bool:
        return s.inner_au <= a <= s.outer_au

    def crossing_age(s0: HZTimeSample, s1: HZTimeSample) -> float:
        # Find where a crosses whichever boundary changed state; bisection on the interpolated band.
        lo, hi = 0.0, 1.0
        in0 = inside(s0)
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            inner = s0.inner_au + (s1.inner_au - s0.inner_au) * mid
            outer = s0.outer_au + (s1.outer_au - s0.outer_au) * mid
            if (inner <= a <= outer) == in0:
                lo = mid
            else:
                hi = mid
        f = 0.5 * (lo + hi)
        return s0.age_years + (s1.age_years - s0.age_years) * f

    intervals: List[Tuple[float, float]] = []
    start: Optional[float] = samples[0].age_years if inside(samples[0]) else None
    for s0, s1 in zip(samples, samples[1:]):
        i0, i1 = inside(s0), inside(s1)
        if i0 == i1:
            continue
        t = crossing_age(s0, s1)
        if i1 and start is None:
            start = t
        elif i0 and start is not None:
            intervals.append((start, t))
            start = None
    if start is not None:
        intervals.append((start, samples[-1].age_years))
    return [(t0, t1) for t0, t1 in intervals if t1 - t0 >= min_duration_years]
