"""
Solar System validation gates for the habitability model.
Training fails if these gates do not pass.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Callable, Dict

import numpy as np


def validate_solar_system_ranking(
    predict_fn: Callable,
    feature_builder_fn: Callable,
    export_path: str = None,
) -> Dict:
    solar_system_data = {
        "Mercury": {
            "pl_rade": 0.383, "pl_masse": 0.055, "pl_orbper": 88.0,
            "pl_orbsmax": 0.387, "pl_orbeccen": 0.2056, "pl_insol": 6.67,
            "pl_eqt": 440.0, "pl_dens": 5.43,
            "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0,
        },
        "Venus": {
            "pl_rade": 0.949, "pl_masse": 0.815, "pl_orbper": 225.0,
            "pl_orbsmax": 0.723, "pl_orbeccen": 0.0068, "pl_insol": 1.91,
            "pl_eqt": 327.0, "pl_dens": 5.24,
            "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0,
        },
        "Earth": {
            "pl_rade": 1.0, "pl_masse": 1.0, "pl_orbper": 365.25,
            "pl_orbsmax": 1.0, "pl_orbeccen": 0.0167, "pl_insol": 1.0,
            "pl_eqt": 255.0, "pl_dens": 5.51,
            "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0,
        },
        "Mars": {
            "pl_rade": 0.532, "pl_masse": 0.107, "pl_orbper": 687.0,
            "pl_orbsmax": 1.524, "pl_orbeccen": 0.0934, "pl_insol": 0.43,
            "pl_eqt": 210.0, "pl_dens": 3.93,
            "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0,
        },
        "Jupiter": {
            "pl_rade": 11.2, "pl_masse": 317.8, "pl_orbper": 4333.0,
            "pl_orbsmax": 5.203, "pl_orbeccen": 0.0484, "pl_insol": 0.037,
            "pl_eqt": 110.0, "pl_dens": 1.33,
            "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0,
        },
    }

    scores = {}
    print("\n" + "=" * 70)
    print("SOLAR SYSTEM VALIDATION")
    print("=" * 70)

    for planet_name, planet_data in solar_system_data.items():
        features, _meta = feature_builder_fn(planet_data, return_meta=True)
        score = float(predict_fn(features))
        scores[planet_name] = score
        print(f"  {planet_name:10s}: {score:.4f}")

    gates = {}
    rocky = ["Mercury", "Venus", "Earth", "Mars"]
    rocky_ranking = sorted(
        [(k, scores[k]) for k in rocky], key=lambda x: x[1], reverse=True
    )
    gates["earth_top_rocky"] = {
        "pass": rocky_ranking[0][0] == "Earth",
        "description": "Earth must be top-1 among rocky inner planets",
        "ranking": rocky_ranking,
        "top_planet": rocky_ranking[0][0],
    }
    gates["mars_gt_venus"] = {
        "pass": scores["Mars"] > scores["Venus"],
        "description": "Mars must score > Venus",
        "mars_score": scores["Mars"],
        "venus_score": scores["Venus"],
        "difference": scores["Mars"] - scores["Venus"],
    }
    gates["venus_penalty"] = {
        "pass": scores["Venus"] / scores["Earth"] < 0.55,
        "description": "Venus must score < 0.55 * Earth",
        "venus_score": scores["Venus"],
        "earth_score": scores["Earth"],
        "ratio": scores["Venus"] / scores["Earth"],
        "threshold": 0.55,
    }
    gates["mercury_penalty"] = {
        "pass": scores["Mercury"] / scores["Earth"] < 0.35,
        "description": "Mercury must score < 0.35 * Earth",
        "mercury_score": scores["Mercury"],
        "earth_score": scores["Earth"],
        "ratio": scores["Mercury"] / scores["Earth"],
        "threshold": 0.35,
    }
    gates["jupiter_penalty"] = {
        "pass": scores["Jupiter"] / scores["Earth"] < 0.05,
        "description": "Jupiter must score ~0 (gas giant)",
        "jupiter_score": scores["Jupiter"],
        "earth_score": scores["Earth"],
        "ratio": scores["Jupiter"] / scores["Earth"],
        "threshold": 0.05,
    }
    gates["earth_near_unity"] = {
        "pass": scores["Earth"] >= 0.90,
        "description": "Earth raw model score must be >= 0.90 (reference for index=100)",
        "threshold": 0.90,
        "earth_score": scores["Earth"],
    }

    all_pass = all(g["pass"] for g in gates.values())
    print("\n" + "=" * 70)
    print("[PASS] ALL GATES PASSED" if all_pass else "[FAIL] VALIDATION FAILED")
    print("=" * 70)

    report = {
        "timestamp": datetime.now().isoformat(),
        "scores": scores,
        "gates": gates,
        "all_pass": all_pass,
        "hz_model": "kopparapu_2013_conservative_rv_em",
    }
    if export_path:
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport: {export_path}")
    return report


def validate_model_predictions(
    model: Any,
    feature_builder_fn: Callable,
    export_dir: str = "exports",
) -> bool:
    def predict_fn(features):
        features_2d = features.reshape(1, -1)
        pred = model.predict(features_2d)
        return pred[0] if hasattr(pred, "__getitem__") else pred

    os.makedirs(export_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = os.path.join(export_dir, f"validation_{ts}.json")
    report = validate_solar_system_ranking(
        predict_fn=predict_fn,
        feature_builder_fn=feature_builder_fn,
        export_path=export_path,
    )
    return report["all_pass"]
