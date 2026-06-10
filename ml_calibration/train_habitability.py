"""
Train AIET habitability XGBoost model (NASA archive + Kopparapu HZ teacher).

Usage (from AIET repo root):
    python ml_calibration/train_habitability.py

Writes:
    ml_calibration/hab_xgb.json
    ml_calibration/training_summary.json
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.ml.ml_features_core import build_features, load_feature_schema
from src.ml.ml_teacher import compute_habitability_score
from src.ml.ml_validation import validate_model_predictions

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: pip install xgboost")
    sys.exit(1)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PLANETS_FILE = os.path.join(DATA_DIR, "exoplanets.csv")
STARS_FILE = os.path.join(DATA_DIR, "stellar_hosts.csv")
OUTPUT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(OUTPUT_DIR, "hab_xgb.json")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "training_summary.json")
EXPORTS_DIR = os.path.join(REPO_ROOT, "exports")

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}


def main() -> int:
    print("\n" + "=" * 80)
    print("AIET Habitability Model Training (K13 HZ + NASA features)")
    print("=" * 80)

    if not os.path.isfile(PLANETS_FILE) or not os.path.isfile(STARS_FILE):
        print(f"\nMissing NASA CSVs in {DATA_DIR}")
        print("Copy exoplanets.csv and stellar_hosts.csv from NASA Exoplanet Archive.")
        return 1

    print("\n[1/7] Loading NASA data...")
    planets_df = pd.read_csv(PLANETS_FILE, comment="#", low_memory=False)
    stars_df = pd.read_csv(STARS_FILE, comment="#", low_memory=False)
    print(f"  Planets: {len(planets_df):,}  Stars: {len(stars_df):,}")

    if "default_flag" in planets_df.columns:
        planets_df = planets_df[planets_df["default_flag"] == 1]
        print(f"  default_flag=1: {len(planets_df):,}")

    df = pd.merge(planets_df, stars_df, on="hostname", how="left", suffixes=("", "_star"))
    print(f"  Merged: {len(df):,}")

    print("\n[2/7] Teacher labels + features...")
    schema = load_feature_schema()
    feature_names = [f["name"] for f in schema["features"]]

    features_list = []
    labels_list = []
    weights_list = []
    imputation_stats = {"total": 0, "fields": {}}
    valid = invalid = 0

    for idx, row in df.iterrows():
        try:
            features, meta = build_features(row.to_dict(), return_meta=True)
            if np.any(np.isnan(features)):
                invalid += 1
                continue
            teacher = compute_habitability_score(features, feature_names)
            label = teacher["score"]
            fd = {feature_names[i]: float(features[i]) for i in range(len(feature_names))}
            w = 1.0
            if teacher["metadata"].get("earth_like"):
                w = 3.0
            elif fd.get("in_hz", 0) >= 0.5 and fd.get("hz_lin_pos", 0) >= 0.05:
                w = 1.5
            features_list.append(features)
            labels_list.append(label)
            weights_list.append(w)
            for field in meta["imputed_fields"]:
                imputation_stats["fields"][field] = imputation_stats["fields"].get(field, 0) + 1
            imputation_stats["total"] += len(meta["imputed_fields"])
            valid += 1
            if valid % 10000 == 0:
                print(f"  {valid:,}...")
        except Exception as e:
            invalid += 1
            if invalid <= 3:
                print(f"  skip {idx}: {e}")

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.float32)
    sw = np.array(weights_list, dtype=np.float32)
    print(f"  Valid: {valid:,}  Skipped: {invalid:,}  shape={X.shape}")

    print("\n[3/7] Train/test split...")
    X_train, X_test, y_train, y_test, sw_train, _sw_test = train_test_split(
        X, y, sw, test_size=0.2, random_state=42
    )

    print("\n[4/7] Training XGBoost...")
    model = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=20)
    model.fit(
        X_train, y_train,
        sample_weight=sw_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )
    print(f"  Best iteration: {model.best_iteration}")

    print("\n[5/7] Metrics...")
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    print(f"  Test MSE={test_mse:.6f}  MAE={test_mae:.6f}  R²={test_r2:.6f}")

    importances = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    print("  Top features:")
    for name, imp in importances[:6]:
        print(f"    {name}: {imp:.4f}")

    print("\n[6/7] Solar System gates...")
    if not validate_model_predictions(
        model=model,
        feature_builder_fn=build_features,
        export_dir=EXPORTS_DIR,
    ):
        print("\n[FAIL] Validation failed — model not saved.")
        return 1

    print("\n[7/7] Saving artifacts...")
    model.get_booster().save_model(MODEL_PATH)
    summary = {
        "model": "hab_xgb",
        "hz_model": "kopparapu_2013_conservative_rv_em",
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "training_date": datetime.now().isoformat(),
        "n_training_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "xgb_params": XGB_PARAMS,
        "best_iteration": int(model.best_iteration),
        "metrics": {
            "test": {
                "mse": float(test_mse),
                "mae": float(test_mae),
                "r2": float(test_r2),
            }
        },
        "feature_importances": {n: float(i) for n, i in importances},
        "imputation_stats": imputation_stats,
        "solar_system_validation": "PASSED",
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  Model: {MODEL_PATH}")
    print(f"  Summary: {SUMMARY_PATH}")
    print("\n[SUCCESS] Training complete.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
