"""
AIET ML teacher — NASA + Kopparapu (2013) conservative HZ training labels.

Covers: stellar T_eff, L, [Fe/H], system multiplicity, insolation vs K13 S_eff,
orbital HZ placement, equilibrium temperature, bulk properties, eccentric flux, tidal proxy.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

from src.ml.ml_features_core import load_feature_schema

EARTH_HZ_LIN_POS = 0.33
EARTH_S_EFF_LOG_POS = 0.33
# Earth flux / K13 Recent-Venus S_eff for the Sun (~1 / 1.775)
EARTH_INSOL_VS_RV = 1.0 / 1.7753


def gaussian_penalty(value: float, optimal: float, sigma: float) -> float:
    return float(np.exp(-((value - optimal) / sigma) ** 2))


def compute_habitability_score(
    features: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """Teacher label in [0, 1] from planet/star features (schema order)."""
    if feature_names is None:
        schema = load_feature_schema()
        feature_names = [f["name"] for f in schema["features"]]

    fd = {name: float(features[i]) for i, name in enumerate(feature_names)}

    pl_rade = fd["pl_rade"]
    pl_orbeccen = fd["pl_orbeccen"]
    pl_insol = fd["pl_insol"]
    pl_eqt = fd["pl_eqt"]
    pl_dens = fd["pl_dens"]
    hz_lin_pos = fd.get("hz_lin_pos", 0.5)
    s_eff_log_pos = fd.get("s_eff_log_pos", hz_lin_pos)
    in_hz = fd.get("in_hz", 1.0)
    insol_vs_rv = fd.get("insol_vs_rv", pl_insol / 1.7753)
    st_met = fd.get("st_met", 0.0)
    sy_snum = fd.get("sy_snum", 1.0)
    flux_ecc_ratio = fd.get("flux_ecc_ratio", 1.0)
    tidal_lock_proxy = fd.get("tidal_lock_proxy", 0.0)

    components = {}

    f_hz_orbit = gaussian_penalty(hz_lin_pos, EARTH_HZ_LIN_POS, 0.50)
    f_hz_flux = gaussian_penalty(s_eff_log_pos, EARTH_S_EFF_LOG_POS, 0.50)
    components["f_hz"] = 0.5 * (f_hz_orbit + f_hz_flux)
    components["f_insol_rv"] = gaussian_penalty(insol_vs_rv, EARTH_INSOL_VS_RV, 0.18)
    components["f_flux"] = gaussian_penalty(pl_insol, 1.0, 0.90)
    components["f_temp"] = gaussian_penalty(pl_eqt, 255.0, 65.0)
    components["f_radius"] = gaussian_penalty(pl_rade, 1.0, 0.65)
    components["f_density"] = gaussian_penalty(pl_dens, 5.51, 2.2)
    components["f_eccentricity"] = gaussian_penalty(pl_orbeccen, 0.02, 0.12)
    components["f_met"] = gaussian_penalty(st_met, 0.0, 0.30)
    components["f_ecc_flux"] = gaussian_penalty(flux_ecc_ratio, 1.0, 0.18)
    components["f_tidal"] = gaussian_penalty(tidal_lock_proxy, 0.15, 0.35)

    if sy_snum <= 1.5:
        components["f_multi_star"] = 1.0
    elif sy_snum <= 2.5:
        components["f_multi_star"] = 0.75
    else:
        components["f_multi_star"] = 0.55

    weights = {
        "f_hz": 0.18,
        "f_insol_rv": 0.14,
        "f_flux": 0.16,
        "f_temp": 0.16,
        "f_radius": 0.12,
        "f_density": 0.08,
        "f_met": 0.04,
        "f_multi_star": 0.04,
        "f_ecc_flux": 0.04,
        "f_tidal": 0.02,
        "f_eccentricity": 0.02,
    }
    assert abs(sum(weights.values()) - 1.0) < 1e-6

    score = sum(components[k] * weights[k] for k in weights)

    regime_multiplier = 1.0
    regime_applied: List[str] = []

    if in_hz < 0.5:
        regime_multiplier *= 0.38
        regime_applied.append("outside_k13_hz")

    if insol_vs_rv > 1.05:
        regime_multiplier *= 0.65
        regime_applied.append("hotter_than_rv_limit")

    if pl_insol >= 5.0:
        regime_multiplier *= 0.10
        regime_applied.append("extreme_hot_5x")
    elif pl_insol >= 3.0:
        regime_multiplier *= 0.30
        regime_applied.append("severe_hot_3x")
    elif pl_insol >= 1.9:
        regime_multiplier *= 0.72
        regime_applied.append("moderate_hot_1.9x")

    if pl_insol <= 0.05:
        regime_multiplier *= 0.20
        regime_applied.append("extreme_cold_0.05x")
    elif pl_insol <= 0.15:
        regime_multiplier *= 0.48
        regime_applied.append("moderate_cold_0.15x")

    score *= regime_multiplier

    earth_like = (
        in_hz >= 0.5
        and 0.08 <= hz_lin_pos <= 0.62
        and 0.40 <= insol_vs_rv <= 0.75
        and 0.45 <= pl_insol <= 1.55
        and 210.0 <= pl_eqt <= 310.0
        and 0.45 <= pl_rade <= 1.65
        and 3.0 <= pl_dens <= 8.0
        and sy_snum <= 1.5
    )
    if earth_like:
        score = 1.0

    score = float(np.clip(score, 0.0, 1.0))

    return {
        "score": score,
        "components": components,
        "metadata": {
            "feature_dict": fd,
            "weights": weights,
            "regime_multiplier": regime_multiplier,
            "regime_applied": regime_applied,
            "earth_like": earth_like,
        },
    }
