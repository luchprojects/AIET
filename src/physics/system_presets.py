"""
System presets for the AIET sandbox.

These helpers return lightweight body specification dictionaries that can be
consumed by the UI layer (via `place_object`) to construct full simulation
systems. They intentionally avoid any dependency on the UI so they can also
be reused by other entry points if needed.

Units:
- Star masses in solar masses (M_sun)
- Planet/moon masses in Earth masses (M_earth)
- Distances in AU where applicable
"""

from typing import Any, Dict, List


def get_blank_system() -> List[Dict[str, Any]]:
    """
    New Spacetime Grid preset.

    Returns a minimal system containing a single Sun-like star at the origin.
    The UI will take these specs and route them through `place_object`, which
    applies the canonical defaults for stellar parameters.
    """
    return [
        {
            "type": "star",
            "name": "Sun",
        }
    ]


def get_earth_moon_sun_system() -> List[Dict[str, Any]]:
    """
    Default Sun–Earth–Moon system preset.

    The values mirror the existing default used by the sandbox:
    - Sun at the center
    - Earth at 1 AU with the frozen DEFAULT_SYSTEM_EARTH_PRESET
    - Moon at 0.00257 AU from Earth

    The dictionaries are specifications that the UI passes to `place_object`
    to build the full body state (IDs, colors, derived properties, etc.).
    """
    return [
        {
            "type": "star",
            "name": "Sun",
        },
        {
            "type": "planet",
            "name": "Earth",
            "semi_major_axis": 1.0,
            "preset_type": "Earth",
            "density": 5.51,  # Earth's density (g/cm³)
            "use_default_preset": True,
        },
        {
            "type": "moon",
            "name": "Moon",
            "semi_major_axis": 0.00257,
        },
    ]


def get_solar_system() -> List[Dict[str, Any]]:
    """
    Full Solar System preset.

    High-level description of the canonical Solar System used by the sandbox:
    - Sun at the center
    - Eight planets at their canonical semi-major axes
    - Earth's Moon as a satellite of Earth

    The UI uses these entries as placement intents and then applies the
    corresponding SOLAR_SYSTEM_PLANET_PRESETS so that all physical parameters
    (mass, radius, temperature, etc.) match the default planet presets.
    """
    planets = [
        "Mercury",
        "Venus",
        "Earth",
        "Mars",
        "Jupiter",
        "Saturn",
        "Uranus",
        "Neptune",
    ]

    bodies: List[Dict[str, Any]] = [
        {"type": "star", "name": "Sun"},
    ]

    for name in planets:
        bodies.append(
            {
                "type": "planet",
                "name": name,
                "preset_type": name,
            }
        )

    # Earth's Moon
    bodies.append(
        {
            "type": "moon",
            "name": "Moon",
            "semi_major_axis": 0.00257,
            "parent": "Earth",
        }
    )

    return bodies


def get_alpha_centauri_system() -> List[Dict[str, Any]]:
    """
    Alpha Centauri neighbor system (triple-star + planets).

    Stars:
    - Alpha Centauri A (G2V)
    - Alpha Centauri B (K1V)
    - Proxima Centauri (M5.5V)

    Planets:
    - Alpha Centauri Ab (candidate, gas giant around A)
    - Proxima d (rocky)
    - Proxima b (rocky, temperate)
    - Proxima c (candidate, cold super-Earth)

    Positions are specified in AU in a simple barycentric layout for the AB pair
    and a wide Proxima companion. The true AB–Proxima separation is ~10^4 AU (~0.06 ly);
    for sandbox viewing the companion is placed at a **compressed** separation so the
    whole triple fits on screen while keeping AB physics and planet orbits realistic.
    """
    bodies: List[Dict[str, Any]] = []

    # Orbital velocities (AU/yr) so the triple does not collapse. G_AU = 39.478.
    # A–B binary: separation 23.5 AU, total mass 2.01 M_sun → tangent speeds in barycenter frame.
    M_A, M_B = 1.10, 0.91
    M_AB = M_A + M_B
    G_AU = 39.478
    r_AB = 23.5  # AU
    v_rel_AB = (G_AU * M_AB / r_AB) ** 0.5  # relative speed for circular orbit at this separation
    v_A = (M_B / M_AB) * v_rel_AB  # A's speed (tangent, +y when A at +x)
    v_B = (M_A / M_AB) * v_rel_AB  # B's speed (opposite)

    # Proxima: physically ~13 000 AU from AB; presentation distance is heavily
    # compressed so the triple fits on screen. Two stability constraints:
    #   1. Holman-Wiegert (circumbinary): a_Prox >= ~2.3 * a_AB to escape the
    #      chaotic zone of a circular binary. a_AB = 23.5 AU -> a_Prox > ~54 AU.
    #   2. Velocity must be the circular velocity around the *actual* AB
    #      barycenter (offset from origin because A is heavier than B), not
    #      around the origin -- otherwise Proxima starts on an eccentric orbit
    #      that grazes the binary at periapsis and eventually ejects.
    # We place Proxima at 75 AU (~3.2 * a_AB) to give a comfortable margin and
    # long-term stability for time-scaled viewing (e.g. 10 yr/sec).
    M_Prox = 0.122
    com_AB_x = (M_A * 11.75 + M_B * (-11.75)) / M_AB  # ~+1.111 AU
    a_Prox = 75.0  # AU from origin; ~3.2 * a_AB, well outside the chaotic zone
    r_from_AB_com = a_Prox - com_AB_x  # ~73.89 AU; what Proxima actually orbits
    # Two-body relative speed: sqrt(G * (M_AB + M_Prox) / r). Using the relative
    # mass keeps the orbit nearly circular once we subtract the system COM velocity
    # in the N-body init; remaining e <~0.01.
    v_Prox = (G_AU * (M_AB + M_Prox) / r_from_AB_com) ** 0.5

    # Alpha Centauri A
    bodies.append(
        {
            "type": "star",
            "name": "Alpha Centauri A",
            "mass": 1.10,  # M_sun
            "radius": 1.22,  # R_sun
            "temperature": 5790.0,  # K
            "luminosity": 1.52,  # L_sun
            "age": 5.3,  # Gyr
            "metallicity": 0.23,  # [Fe/H] dex
            "spectral_class": "G2V",
            "activity": "Quiet",
            "binary_semi_major_axis_au": 23.5,
            "binary_eccentricity": 0.518,
            "position_au": [11.75, 0.0],
            "velocity_au": [0.0, v_A],  # tangent so A–B orbit is stable
        }
    )

    # Alpha Centauri B
    bodies.append(
        {
            "type": "star",
            "name": "Alpha Centauri B",
            "mass": 0.91,
            "radius": 0.86,
            "temperature": 5260.0,
            "luminosity": 0.50,
            "age": 5.3,
            "metallicity": 0.23,
            "spectral_class": "K1V",
            "activity": "Quiet",
            "binary_semi_major_axis_au": 23.5,
            "binary_eccentricity": 0.518,
            "position_au": [-11.75, 0.0],
            "velocity_au": [0.0, -v_B],  # opposite to A so momentum cancels
        }
    )

    # Proxima Centauri
    bodies.append(
        {
            "type": "star",
            "name": "Proxima Centauri",
            "mass": 0.122,
            "radius": 0.154,
            "temperature": 3040.0,
            "luminosity": 0.0017,
            "age": 4.8,
            "metallicity": 0.0,
            "spectral_class": "M5.5V",
            "activity": "Active",
            "system_semi_major_axis_au": a_Prox,
            "position_au": [a_Prox, 0.0],
            "velocity_au": [0.0, v_Prox],  # tangent so wide orbit is stable
        }
    )

    # Alpha Centauri Ab (candidate, gas giant around A)
    bodies.append(
        {
            "type": "planet",
            "name": "Alpha Centauri Ab",
            "host_star": "Alpha Centauri A",
            "preset_type": "Alpha Centauri Ab",
            "classification": "Gas Giant",
            # Physical
            "mass": 95.0,  # M_earth
            "radius": 9.1,  # R_earth
            "density": 0.7,  # g/cm^3
            # Orbit
            "semi_major_axis": 1.5,  # AU
            "orbital_period": 650.0,  # days (approx)
            "eccentricity": 0.30,
            # Flux/temperature
            "stellarFlux": 0.75,  # S_earth
            "equilibrium_temperature": 225.0,  # K
            "greenhouse_offset": 40.0,  # K
            "temperature": 265.0,  # K
            # Rotation / atmosphere
            "rotation_period_days": 10.0 / 24.0,  # ~10 hours
            "atmosphere_type": "Dense (H/He)",
        }
    )

    # Proxima d (rocky, inner planet)
    bodies.append(
        {
            "type": "planet",
            "name": "Proxima d",
            "host_star": "Proxima Centauri",
            "preset_type": "Proxima d",
            "classification": "Rocky",
            "mass": 0.26,
            "radius": 0.70,
            "density": 5.5,
            "semi_major_axis": 0.029,
            "orbital_period": 5.1,
            "eccentricity": 0.05,
            "stellarFlux": 1.90,
            "equilibrium_temperature": 360.0,
            "greenhouse_offset": 10.0,
            "temperature": 370.0,
            "rotation_period_days": 5.0,
            "atmosphere_type": "Thin",
        }
    )

    # Proxima b (temperate rocky)
    bodies.append(
        {
            "type": "planet",
            "name": "Proxima b",
            "host_star": "Proxima Centauri",
            "preset_type": "Proxima b",
            "classification": "Rocky",
            "mass": 1.07,
            "radius": 1.03,
            "density": 5.5,
            "semi_major_axis": 0.048,
            "orbital_period": 11.2,
            "eccentricity": 0.05,
            "stellarFlux": 0.65,
            "equilibrium_temperature": 234.0,
            "greenhouse_offset": 30.0,
            "temperature": 264.0,
            "rotation_period_days": 11.0,
            "atmosphere_type": "Earth-like",
        }
    )

    # Proxima c (candidate, cold super-Earth)
    bodies.append(
        {
            "type": "planet",
            "name": "Proxima c",
            "host_star": "Proxima Centauri",
            "preset_type": "Proxima c",
            "classification": "Cold Super-Earth",
            "mass": 7.0,
            "radius": 1.9,
            "density": 4.0,
            "semi_major_axis": 1.48,
            "orbital_period": 1900.0,
            "eccentricity": 0.10,
            "stellarFlux": 0.001,
            "equilibrium_temperature": 40.0,
            "greenhouse_offset": 5.0,
            "temperature": 45.0,
            "rotation_period_days": 1.0,
            "atmosphere_type": "Thin",
        }
    )

    return bodies


def get_trappist1_system() -> List[Dict[str, Any]]:
    """
    TRAPPIST-1 compact multi-planet system.

    Design goals:
    - Keep the representation lightweight and UI-friendly (like other presets).
    - Clearly separate "measured" / tightly constrained quantities from
      "assumed" or illustrative parameters.
    - Use AU, Earth masses, Earth radii, etc., consistent with the header.

    Conventions below:
    - Comments starting with "Measured:" (or "Well-constrained:") reflect values
      that are close to literature numbers.
    - Comments starting with "Assumed:" are reasonable but not strictly
      observationally derived (e.g., greenhouse offsets, atmosphere labels).
    """
    bodies: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Host star: TRAPPIST-1
    # ------------------------------------------------------------------
    bodies.append(
        {
            "type": "star",
            "name": "TRAPPIST-1",
            # Measured / well-constrained stellar properties
            "mass": 0.089,  # M_sun  (ultra-cool M dwarf)
            "radius": 0.121,  # R_sun
            "temperature": 2550.0,  # K (effective temperature)
            "luminosity": 0.00055,  # L_sun
            "age": 7.6,  # Gyr (order-of-magnitude; literature varies)
            "metallicity": 0.04,  # [Fe/H] dex (roughly solar to slightly sub-solar)
            # Descriptive tags
            "spectral_class": "M8V",
            "activity": "Active",
        }
    )

    # Helper for planets to keep dictionary creation readable
    def add_planet(
        name: str,
        semi_major_axis_au: float,
        mass_earth: float,
        radius_earth: float,
        equilibrium_temperature_k: float,
        stellar_flux_se: float,
        orbital_period_days: float,
        greenhouse_offset_k: float,
        classification: str,
        atmosphere_type: str,
        eccentricity: float = 0.01,
    ) -> None:
        bodies.append(
            {
                "type": "planet",
                "name": name,
                "host_star": "TRAPPIST-1",
                "preset_type": name,
                "classification": classification,
                # Measured / well-constrained
                "mass": mass_earth,  # M_earth
                "radius": radius_earth,  # R_earth
                "semi_major_axis": semi_major_axis_au,  # AU (NASA archive)
                "orbital_period": orbital_period_days,  # days
                "eccentricity": eccentricity,
                "stellarFlux": stellar_flux_se,  # S_earth
                "equilibrium_temperature": equilibrium_temperature_k,  # K
                # Assumed / model-like knobs
                "greenhouse_offset": greenhouse_offset_k,  # K
                "temperature": equilibrium_temperature_k + greenhouse_offset_k,
                "rotation_period_days": orbital_period_days,  # Assumed: near-tidally locked
                "atmosphere_type": atmosphere_type,
            }
        )

    # Orbital semi-major axes below are NASA archive values (AU). Visual orbit spread
    # when inner planets would overlap the host star is handled uniformly in the UI
    # via per-host _orbit_visual_scale (same law for every preset).

    # TRAPPIST-1 b – hot inner world
    add_planet(
        name="TRAPPIST-1 b",
        semi_major_axis_au=0.0111,
        mass_earth=1.37,
        radius_earth=1.12,
        equilibrium_temperature_k=400.0,
        stellar_flux_se=4.2,
        orbital_period_days=1.51,
        greenhouse_offset_k=50.0,
        classification="Rocky",
        atmosphere_type="Very thick, hot",
    )

    # TRAPPIST-1 c
    add_planet(
        name="TRAPPIST-1 c",
        semi_major_axis_au=0.0152,
        mass_earth=1.31,
        radius_earth=1.10,
        equilibrium_temperature_k=342.0,
        stellar_flux_se=2.3,
        orbital_period_days=2.42,
        greenhouse_offset_k=40.0,
        classification="Rocky",
        atmosphere_type="Thick, hot",
    )

    # TRAPPIST-1 d
    add_planet(
        name="TRAPPIST-1 d",
        semi_major_axis_au=0.0211,
        mass_earth=0.39,
        radius_earth=0.78,
        equilibrium_temperature_k=288.0,
        stellar_flux_se=1.1,
        orbital_period_days=4.05,
        greenhouse_offset_k=25.0,
        classification="Rocky",
        atmosphere_type="Thin / marginal",
    )

    # TRAPPIST-1 e – near the classical habitable zone
    add_planet(
        name="TRAPPIST-1 e",
        semi_major_axis_au=0.0282,
        mass_earth=0.69,
        radius_earth=0.92,
        equilibrium_temperature_k=251.0,
        stellar_flux_se=0.66,
        orbital_period_days=6.10,
        greenhouse_offset_k=35.0,
        classification="Rocky (temperate)",
        atmosphere_type="Temperate, possible surface water",
    )

    # TRAPPIST-1 f
    add_planet(
        name="TRAPPIST-1 f",
        semi_major_axis_au=0.0371,
        mass_earth=1.04,
        radius_earth=1.05,
        equilibrium_temperature_k=219.0,
        stellar_flux_se=0.38,
        orbital_period_days=9.21,
        greenhouse_offset_k=40.0,
        classification="Cold rocky / water-rich",
        atmosphere_type="Thick, possibly icy",
    )

    # TRAPPIST-1 g
    add_planet(
        name="TRAPPIST-1 g",
        semi_major_axis_au=0.0451,
        mass_earth=1.34,
        radius_earth=1.15,
        equilibrium_temperature_k=198.0,
        stellar_flux_se=0.26,
        orbital_period_days=12.35,
        greenhouse_offset_k=35.0,
        classification="Cold rocky / water-rich",
        atmosphere_type="Thick, cold",
    )

    # TRAPPIST-1 h – outermost, low flux
    add_planet(
        name="TRAPPIST-1 h",
        semi_major_axis_au=0.0619,
        mass_earth=0.77,
        radius_earth=0.77,
        equilibrium_temperature_k=173.0,
        stellar_flux_se=0.13,
        orbital_period_days=18.77,
        greenhouse_offset_k=30.0,
        classification="Icy / outer rocky",
        atmosphere_type="Thin or tenuous",
    )

    return bodies


def get_kepler62_system() -> List[Dict[str, Any]]:
    """
    Kepler-62 five-planet system (NASA Exoplanet Archive, default_flag=1 rows).

    Stellar parameters (K2 V host):
    - T_eff = 4925 K, M = 0.69 M_sun, R = 0.64 R_sun, [Fe/H] = -0.37, age ~7 Gyr
    - Luminosity from archive log10(L/L_sun) = -0.678 → ~0.21 L_sun

    Planet orbital/archival values (period, semi-major axis, radius, mass, T_eq)
    are taken from the consolidated default solutions. Stellar flux is computed as
    L / a² in Earth insolation units. Greenhouse offsets and atmosphere labels
    are illustrative (surface temperatures are not directly observed).
    """
    bodies: List[Dict[str, Any]] = []

    # log10(L/L_sun) = -0.678 from NASA Exoplanet Archive stellar_hosts
    luminosity = 10.0 ** (-0.678)

    bodies.append(
        {
            "type": "star",
            "name": "Kepler-62",
            "mass": 0.69,
            "radius": 0.64,
            "temperature": 4925.0,
            "luminosity": luminosity,
            "age": 7.0,
            "metallicity": -0.37,
            "spectral_class": "K2V",
            "activity": "Quiet",
        }
    )

    def add_planet(
        name: str,
        semi_major_axis_au: float,
        mass_earth: float,
        radius_earth: float,
        equilibrium_temperature_k: float,
        orbital_period_days: float,
        greenhouse_offset_k: float,
        classification: str,
        atmosphere_type: str,
        eccentricity: float = 0.0,
    ) -> None:
        stellar_flux_se = luminosity / (semi_major_axis_au ** 2)
        bodies.append(
            {
                "type": "planet",
                "name": name,
                "host_star": "Kepler-62",
                "preset_type": name,
                "classification": classification,
                "mass": mass_earth,
                "radius": radius_earth,
                "semi_major_axis": semi_major_axis_au,
                "orbital_period": orbital_period_days,
                "eccentricity": eccentricity,
                "stellarFlux": stellar_flux_se,
                "equilibrium_temperature": equilibrium_temperature_k,
                "greenhouse_offset": greenhouse_offset_k,
                "temperature": equilibrium_temperature_k + greenhouse_offset_k,
                "rotation_period_days": orbital_period_days,
                "atmosphere_type": atmosphere_type,
            }
        )

    add_planet(
        name="Kepler-62 b",
        semi_major_axis_au=0.0553,
        mass_earth=9.0,
        radius_earth=1.31,
        equilibrium_temperature_k=750.0,
        orbital_period_days=5.715,
        greenhouse_offset_k=40.0,
        classification="Hot super-Earth",
        atmosphere_type="Very thick, hot",
    )

    add_planet(
        name="Kepler-62 c",
        semi_major_axis_au=0.0929,
        mass_earth=4.0,
        radius_earth=0.54,
        equilibrium_temperature_k=578.0,
        orbital_period_days=12.442,
        greenhouse_offset_k=25.0,
        classification="Rocky / sub-Neptune",
        atmosphere_type="Thick, hot",
    )

    add_planet(
        name="Kepler-62 d",
        semi_major_axis_au=0.120,
        mass_earth=14.0,
        radius_earth=1.95,
        equilibrium_temperature_k=510.0,
        orbital_period_days=18.164,
        greenhouse_offset_k=20.0,
        classification="Warm super-Earth",
        atmosphere_type="Thick, warm",
    )

    add_planet(
        name="Kepler-62 e",
        semi_major_axis_au=0.427,
        mass_earth=36.0,
        radius_earth=1.61,
        equilibrium_temperature_k=270.0,
        orbital_period_days=122.387,
        greenhouse_offset_k=33.0,
        classification="Habitable-zone super-Earth",
        atmosphere_type="Temperate, possible surface water",
    )

    add_planet(
        name="Kepler-62 f",
        semi_major_axis_au=0.718,
        mass_earth=35.0,
        radius_earth=1.41,
        equilibrium_temperature_k=208.0,
        orbital_period_days=267.291,
        greenhouse_offset_k=25.0,
        classification="Cold habitable-zone super-Earth",
        atmosphere_type="Thick, cold",
    )

    return bodies
