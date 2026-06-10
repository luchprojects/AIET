"""
AIET ML - Canonical Feature Builder
Single source of truth for feature extraction (training, inference, debug, Earth reference)
"""

import numpy as np
import json
import os
from typing import Tuple, Dict, Optional


def load_feature_schema(schema_path: str = None) -> dict:
    """Load the feature schema JSON."""
    if schema_path is None:
        from src.utils.paths import feature_schema_path
        schema_path = feature_schema_path()
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_features(
    planet_row: dict,
    star_row: Optional[dict] = None,
    return_meta: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Build feature vector from NASA planet + star data.
    
    This is the SINGLE canonical feature builder used for:
    - Training dataset generation
    - Runtime inference in AIET
    - Debug sanity checks
    - Earth reference vector
    
    Args:
        planet_row: dict with NASA planet columns (pl_rade, pl_masse, etc.)
        star_row: dict with NASA star columns (st_teff, st_mass, etc.)
                  If None, assumes planet_row contains merged star data
        return_meta: If True, return metadata dict with imputation flags
    
    Returns:
        x: np.ndarray of shape (12,) with features in schema order
        meta: dict with intermediate values + imputation flags
    """
    
    # Merge star data if provided separately
    if star_row is not None:
        data = {**planet_row, **star_row}
    else:
        data = planet_row
    
    # Initialize imputation tracking
    meta = {
        "imputed_fields": [],
        "intermediate_values": {},
        "warnings": []
    }
    
    # =============================================================================
    # PLANET FEATURES (indices 0-7)
    # =============================================================================
    
    # Feature 0: pl_rade (Planet radius in Earth radii)
    pl_rade = data.get("pl_rade")
    if pl_rade is None or np.isnan(pl_rade):
        pl_rade = 6.0  # Median fallback
        meta["imputed_fields"].append("pl_rade")
    pl_rade = float(np.clip(pl_rade, 0.1, 20.0))
    
    # Feature 1: pl_masse (Planet mass in Earth masses)
    pl_masse = data.get("pl_masse")
    if pl_masse is None or np.isnan(pl_masse):
        # Rocky planet M-R relation (Zeng et al. 2016)
        if pl_rade < 1.6:
            pl_masse = pl_rade ** 2.06
            meta["imputed_fields"].append("pl_masse (rocky M-R)")
        else:
            # Gas/ice giant - use median by radius bin
            if pl_rade < 4.0:
                pl_masse = 10.0  # Neptune-like
            elif pl_rade < 8.0:
                pl_masse = 50.0  # Saturn-like
            else:
                pl_masse = 200.0  # Jupiter-like
            meta["imputed_fields"].append("pl_masse (gas giant median)")
    pl_masse = float(np.clip(pl_masse, 0.001, 500.0))
    
    # Feature 2: pl_orbper (Orbital period in days)
    pl_orbper = data.get("pl_orbper")
    if pl_orbper is None or np.isnan(pl_orbper):
        pl_orbper = 10.0  # Median fallback
        meta["imputed_fields"].append("pl_orbper")
    pl_orbper = float(np.clip(pl_orbper, 0.1, 100000.0))
    
    # Feature 9: st_mass (needed for pl_orbsmax computation)
    st_mass = data.get("st_mass")
    if st_mass is None or np.isnan(st_mass):
        st_mass = 1.0  # Sun-like default
        meta["imputed_fields"].append("st_mass (for orbit calc)")
    st_mass = float(np.clip(st_mass, 0.08, 100.0))
    
    # Feature 3: pl_orbsmax (Semi-major axis in AU)
    pl_orbsmax = data.get("pl_orbsmax")
    if pl_orbsmax is None or np.isnan(pl_orbsmax):
        # Kepler's 3rd law: a^3 = P^2 * M_star (P in years, M in solar masses)
        P_yr = pl_orbper / 365.25
        pl_orbsmax = (P_yr ** 2 * st_mass) ** (1.0 / 3.0)
        meta["imputed_fields"].append("pl_orbsmax (Kepler 3rd law)")
    pl_orbsmax = float(np.clip(pl_orbsmax, 0.01, 1000.0))
    meta["intermediate_values"]["pl_orbsmax"] = pl_orbsmax
    
    # Feature 4: pl_orbeccen (Orbital eccentricity)
    pl_orbeccen = data.get("pl_orbeccen")
    if pl_orbeccen is None or np.isnan(pl_orbeccen):
        pl_orbeccen = 0.0  # Assume circular
        meta["imputed_fields"].append("pl_orbeccen")
    pl_orbeccen = float(np.clip(pl_orbeccen, 0.0, 1.0))
    
    # =============================================================================
    # STELLAR FEATURES (indices 8-11)
    # =============================================================================
    
    # Feature 8: st_teff (Stellar effective temperature in K)
    st_teff = data.get("st_teff")
    if st_teff is None or np.isnan(st_teff):
        st_teff = 5778.0  # Sun-like default
        meta["imputed_fields"].append("st_teff")
    st_teff = float(np.clip(st_teff, 2000.0, 50000.0))
    
    # Feature 10: st_rad (Stellar radius in Solar radii)
    st_rad = data.get("st_rad")
    if st_rad is None or np.isnan(st_rad):
        st_rad = 1.0  # Sun-like default
        meta["imputed_fields"].append("st_rad")
    st_rad = float(np.clip(st_rad, 0.1, 1000.0))
    
    # Feature 11: st_lum (Stellar luminosity in Solar luminosities)
    st_lum = data.get("st_lum")
    if st_lum is None or np.isnan(st_lum):
        # Stefan-Boltzmann law: L ∝ R^2 * T^4
        st_lum = (st_rad ** 2) * ((st_teff / 5778.0) ** 4)
        meta["imputed_fields"].append("st_lum (Stefan-Boltzmann)")
    st_lum = float(np.clip(st_lum, 0.0001, 1000000.0))
    meta["intermediate_values"]["st_lum"] = st_lum
    
    # =============================================================================
    # FLUX & TEMPERATURE (indices 5-6, need st_lum)
    # =============================================================================
    
    # Feature 5: pl_insol (Insolation flux in Earth flux units)
    pl_insol = data.get("pl_insol")
    if pl_insol is None or np.isnan(pl_insol):
        # Compute from stellar luminosity and distance
        pl_insol = st_lum / (pl_orbsmax ** 2)
        meta["imputed_fields"].append("pl_insol (L/a^2)")
    pl_insol = float(np.clip(pl_insol, 0.0001, 100.0))
    meta["intermediate_values"]["pl_insol"] = pl_insol
    
    # Feature 6: pl_eqt (Equilibrium temperature in K, no atmosphere)
    pl_eqt = data.get("pl_eqt")
    if pl_eqt is None or np.isnan(pl_eqt):
        # T_eq ∝ flux^0.25 (assuming Earth-like albedo ~0.3)
        # Earth: 1.0 flux → ~255K equilibrium (actual 288K with greenhouse)
        pl_eqt = 278.5 * (pl_insol ** 0.25)
        meta["imputed_fields"].append("pl_eqt (from flux)")
    pl_eqt = float(np.clip(pl_eqt, 50.0, 3000.0))
    meta["intermediate_values"]["pl_eqt"] = pl_eqt
    
    # Feature 7: pl_dens (Planet density in g/cm³)
    pl_dens = data.get("pl_dens")
    if pl_dens is None or np.isnan(pl_dens):
        # Compute from mass and radius
        # Density = Mass / Volume
        # Units: M⊕ = 5.972e24 kg, R⊕ = 6.371e6 m
        M_earth_kg = 5.972e24
        R_earth_m = 6.371e6
        
        mass_kg = pl_masse * M_earth_kg
        radius_m = pl_rade * R_earth_m
        volume_m3 = (4.0 / 3.0) * np.pi * (radius_m ** 3)
        
        if volume_m3 > 0:
            density_kg_m3 = mass_kg / volume_m3
            pl_dens = density_kg_m3 / 1000.0  # Convert kg/m³ to g/cm³
        else:
            pl_dens = 5.51  # Earth-like fallback
        meta["imputed_fields"].append("pl_dens (from M,R)")
    pl_dens = float(np.clip(pl_dens, 0.1, 30.0))
    meta["intermediate_values"]["pl_dens"] = pl_dens

    # =============================================================================
    # KOPPARAPU (2013) HZ — conservative Recent Venus / Early Mars (Table 3)
    # =============================================================================
    try:
        from src.physics.kopparapu_hz import kopparapu_hz_boundaries_au
    except ImportError:
        from physics.kopparapu_hz import kopparapu_hz_boundaries_au

    hz_inner_au, hz_outer_au, s_eff_in, s_eff_out = kopparapu_hz_boundaries_au(st_lum, st_teff)
    hz_width_au = max(hz_outer_au - hz_inner_au, 1e-9)
    # Linear position in HZ: 0 = Recent Venus edge, 1 = Early Mars edge (Earth ~0.24 at 1 AU)
    hz_lin_pos = (pl_orbsmax - hz_inner_au) / hz_width_au
    in_hz = 1.0 if hz_inner_au <= pl_orbsmax <= hz_outer_au else 0.0
    # Flux-based position in HZ (log insolation between inner/outer S_eff limits)
    li, lo, lp = np.log(max(s_eff_in, 1e-9)), np.log(max(s_eff_out, 1e-9)), np.log(max(pl_insol, 1e-9))
    denom = li - lo
    s_eff_log_pos = float((li - lp) / denom) if abs(denom) > 1e-12 else hz_lin_pos
    s_eff_log_pos = float(np.clip(s_eff_log_pos, 0.0, 1.0))

    hz_inner_au = float(np.clip(hz_inner_au, 0.001, 1000.0))
    hz_outer_au = float(np.clip(hz_outer_au, 0.001, 1000.0))
    hz_lin_pos = float(np.clip(hz_lin_pos, 0.0, 1.0))
    meta["intermediate_values"]["hz_inner_au"] = hz_inner_au
    meta["intermediate_values"]["hz_outer_au"] = hz_outer_au
    meta["intermediate_values"]["hz_s_eff_inner"] = float(s_eff_in)
    meta["intermediate_values"]["hz_s_eff_outer"] = float(s_eff_out)

    s_eff_inner = float(np.clip(s_eff_in, 0.01, 100.0))
    s_eff_outer = float(np.clip(s_eff_out, 0.01, 100.0))
    insol_vs_rv = float(pl_insol / s_eff_inner)

    # NASA archive + sim: stellar metallicity [Fe/H] and system star count
    st_met = data.get("st_met")
    if st_met is None or (isinstance(st_met, float) and np.isnan(st_met)):
        st_met = data.get("metallicity", 0.0)
    if st_met is None or (isinstance(st_met, float) and np.isnan(st_met)):
        st_met = 0.0
        meta["imputed_fields"].append("st_met")
    st_met = float(np.clip(st_met, -2.5, 1.0))

    sy_snum = data.get("sy_snum")
    if sy_snum is None or (isinstance(sy_snum, float) and np.isnan(sy_snum)):
        sy_snum = 1.0
        meta["imputed_fields"].append("sy_snum")
    sy_snum = float(np.clip(sy_snum, 1.0, 6.0))

    # Eccentric flux swing (periastron/mean insolation ratio, Keplerian)
    e = float(np.clip(pl_orbeccen, 0.0, 0.95))
    flux_ecc_ratio = float(np.sqrt((1.0 + e) / max(1.0 - e, 0.05)))

    # Tidal-locking proxy: short-period planets more likely locked (NASA-consistent heuristic)
    tidal_lock_proxy = float(np.clip(1.0 / (1.0 + (pl_orbper / 45.0) ** 2), 0.0, 1.0))
    
    # =============================================================================
    # ASSEMBLE FEATURE VECTOR (SCHEMA ORDER)
    # =============================================================================
    
    features = np.array([
        pl_rade,
        pl_masse,
        pl_orbper,
        pl_orbsmax,
        pl_orbeccen,
        pl_insol,
        pl_eqt,
        pl_dens,
        st_teff,
        st_mass,
        st_rad,
        st_lum,
        hz_inner_au,
        hz_outer_au,
        hz_lin_pos,
        s_eff_log_pos,
        in_hz,
        st_met,
        sy_snum,
        s_eff_inner,
        insol_vs_rv,
        flux_ecc_ratio,
        tidal_lock_proxy,
    ], dtype=np.float32)
    
    # Validate no NaNs
    if np.any(np.isnan(features)):
        nan_indices = np.where(np.isnan(features))[0]
        meta["warnings"].append(f"NaN values at indices: {nan_indices.tolist()}")
    
    if return_meta:
        return features, meta
    else:
        return features


def get_earth_reference_features() -> Tuple[np.ndarray, Dict]:
    """
    Get Earth's feature vector using exact Solar System values.
    Used for Earth normalization in UI display.
    """
    earth_data = {
        "pl_rade": 1.0,
        "pl_masse": 1.0,
        "pl_orbper": 365.25,
        "pl_orbsmax": 1.0,
        "pl_orbeccen": 0.0167,
        "pl_insol": 1.0,
        "pl_eqt": 255.0,
        "pl_dens": 5.51,
        "st_teff": 5778.0,
        "st_mass": 1.0,
        "st_rad": 1.0,
        "st_lum": 1.0,
        "st_met": 0.0,
        "metallicity": 0.0,
        "sy_snum": 1.0,
    }
    
    return build_features(earth_data, return_meta=True)


def validate_features(features: np.ndarray, feature_schema: dict) -> list:
    """
    Validate that features are within expected ranges.
    Returns list of warnings.
    """
    warnings = []
    
    for i, feat_def in enumerate(feature_schema["features"]):
        value = features[i]
        expected_range = feat_def["range"]
        
        if value < expected_range[0] or value > expected_range[1]:
            warnings.append(
                f"{feat_def['name']} = {value:.4f} outside range "
                f"{expected_range} ({feat_def['units']})"
            )
    
    return warnings


if __name__ == "__main__":
    # Test with Earth
    print("Testing feature builder with Earth reference:")
    print("=" * 70)
    
    earth_features, earth_meta = get_earth_reference_features()
    
    print("\nEarth Feature Vector:")
    schema = load_feature_schema()
    for i, feat in enumerate(schema["features"]):
        print(f"  {i:2d}. {feat['name']:15s} = {earth_features[i]:10.4f} {feat['units']}")
    
    print(f"\nImputed fields: {earth_meta['imputed_fields']}")
    print(f"Warnings: {earth_meta['warnings']}")
    
    # Validate
    validation_warnings = validate_features(earth_features, schema)
    if validation_warnings:
        print("\nValidation warnings:")
        for w in validation_warnings:
            print(f"  - {w}")
    else:
        print("\n[OK] All features within expected ranges")
