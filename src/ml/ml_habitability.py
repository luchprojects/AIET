"""
AIET ML - Runtime Habitability Calculator
Uses XGBoost model with NASA-locked features
"""

import os
import sys
import json
import numpy as np
from typing import Dict, Tuple

try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print("Warning: XGBoost not installed. ML calculator will not be available.")

from src.ml.ml_features import build_features, get_earth_reference_features, load_feature_schema

try:
    from src.ml.ml_uncertainty import run_monte_carlo
except ImportError:
    run_monte_carlo = None


class MLHabitabilityCalculator:
    """
    ML Habitability Calculator
    
    Key improvements:
    - Uses NASA-locked features (no synthetic T_surf)
    - XGBoost model (more stable than PyTorch MLP)
    - Consistent feature builder for training/inference
    - Clear separation between raw score and Earth-normalized display
    """
    
    def __init__(self, model_path: str = None, schema_path: str = None):
        """
        Initialize ML calculator.
        
        Args:
            model_path: Path to XGBoost model file (hab_xgb.json)
            schema_path: Path to feature schema JSON (features.json)
        """
        
        if xgb is None:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        from src.utils.paths import feature_schema_path, model_path as get_model_path

        if schema_path is None:
            schema_path = feature_schema_path()
        self.feature_schema = load_feature_schema(schema_path)
        self.feature_names = [f["name"] for f in self.feature_schema["features"]]

        print(f"[ML] Loaded feature schema: {len(self.feature_names)} features")

        if model_path is None:
            model_path = get_model_path("hab_xgb.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"XGBoost model not found at {model_path}")
        
        # Load XGBoost model (compatible with XGBoost 3.x)
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        print(f"[ML] Loaded XGBoost model from: {model_path}")
        
        # Earth reference: canonical Solar System vector defines index = 100
        self.earth_features, earth_meta = get_earth_reference_features()
        self.earth_reference_raw = self._predict_raw(self.earth_features)
        self.earth_raw_score = self.earth_reference_raw  # legacy alias

        print(f"[ML] Earth reference raw: {self.earth_reference_raw:.4f}")
        print(f"[ML] Display scale: earth_reference_index (Earth = 100)")
        print(f"[ML] Initialization complete")

    def raw_to_display_index(self, raw_score: float) -> float:
        """
        Comparative habitability index: Earth reference prediction = 100.
        Other planets are percentages relative to that baseline (not per-planet forcing).
        """
        ref = self.earth_reference_raw
        if ref > 1e-6:
            return float(np.clip(100.0 * raw_score / ref, 0.0, 100.0))
        return float(np.clip(raw_score * 100.0, 0.0, 100.0))
    
    def _predict_raw(self, features: np.ndarray) -> float:
        """
        Get raw model prediction (0-1 scale).
        
        Args:
            features: Feature vector of shape (12,)
        
        Returns:
            Raw score in [0, 1]
        """
        # Convert to DMatrix for XGBoost Booster
        import xgboost as xgb
        features_2d = features.reshape(1, -1)
        dmatrix = xgb.DMatrix(features_2d)
        raw_score = self.model.predict(dmatrix)[0]
        return float(np.clip(raw_score, 0.0, 1.0))
    
    def predict(
        self,
        features: dict,
        return_raw: bool = False,
        return_meta: bool = False
    ) -> float | Tuple[float, Dict]:
        """
        Predict habitability score from planet/star features.
        
        Args:
            features: Dict with NASA column names (pl_rade, st_teff, etc.)
            return_raw: If True, return raw 0-1 score; if False, return Earth-normalized 0-100
            return_meta: If True, return (score, metadata dict) tuple
        
        Returns:
            Habitability score (0-100 by default, 0-1 if return_raw=True)
            If return_meta=True, returns (score, meta) tuple
        """
        
        try:
            # Build features using canonical builder
            feature_vector, meta = build_features(features, return_meta=True)
            
            # Get raw prediction
            raw_score = self._predict_raw(feature_vector)
            
            if return_raw:
                final_score = raw_score
            else:
                final_score = self.raw_to_display_index(raw_score)

            meta["raw_score"] = raw_score
            meta["earth_reference_raw"] = self.earth_reference_raw
            meta["display_score"] = final_score if not return_raw else self.raw_to_display_index(raw_score)
            meta["normalization_mode"] = "earth_reference_index"
            
            if return_meta:
                return final_score, meta
            else:
                return final_score
        
        except Exception as e:
            print(f"[ML] Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0 if not return_meta else (0.0, {"error": str(e)})
    
    def predict_batch(
        self,
        planet_rows: list,
        return_raw: bool = False
    ) -> np.ndarray:
        """
        Predict scores for multiple planets efficiently.
        
        Args:
            planet_rows: List of dicts with NASA column names
            return_raw: If True, return raw 0-1 scores; if False, Earth-normalized 0-100
        
        Returns:
            Array of scores
        """
        
        # Build feature matrix
        features_list = []
        for row in planet_rows:
            try:
                features = build_features(row, return_meta=False)
                features_list.append(features)
            except Exception as e:
                print(f"[ML] Failed to build features for planet: {e}")
                n_feat = len(self.feature_names)
                features_list.append(np.zeros(n_feat, dtype=np.float32))
        
        X = np.array(features_list, dtype=np.float32)
        
        # Predict using DMatrix
        import xgboost as xgb
        dmatrix = xgb.DMatrix(X)
        raw_scores = self.model.predict(dmatrix)
        raw_scores = np.clip(raw_scores, 0.0, 1.0)
        
        # Normalize if requested
        if return_raw:
            return raw_scores
        else:
            ref = self.earth_reference_raw
            if ref > 1e-6:
                return np.clip(raw_scores / ref * 100.0, 0.0, 100.0)
            return np.clip(raw_scores * 100.0, 0.0, 100.0)
    
    def predict_with_uncertainty(
        self,
        planet_data: dict,
        star_data: dict | None = None,
        N: int = 1000,
        seed: int | None = None,
        fallback_config: dict | None = None,
        tolerance: float | None = None,
        checkpoint_interval: int = 50,
    ) -> Dict:
        """
        Monte Carlo uncertainty propagation for the Habitability Index.

        Does not modify the existing deterministic predict() path. Samples input
        uncertainties (NASA err1/err2 or documented fallback), runs the full
        feature builder and XGBoost per sample, then Earth-normalizes and
        returns mean, std, and 95% CI of the display index.

        Convergence Diagnostics:
            This method tracks running mean as samples accumulate. Convergence
            diagnostics measure the STABILITY of Monte Carlo sampling, NOT model
            correctness. The standard_error field estimates sampling uncertainty
            of the mean (std_dev / sqrt(N)).

        Scientific note: The confidence interval reflects propagated INPUT
        uncertainty only. It does NOT capture model epistemic uncertainty.
        The result is NOT a probability of life.

        Args:
            planet_data: NASA-style planet dict (pl_rade, pl_masse, ...).
                May include pl_radeerr1, pl_radeerr2, etc.
            star_data: Optional NASA-style star dict. If None, planet_data
                is treated as merged planet+star.
            N: Number of Monte Carlo samples (default 1000).
            seed: Random seed for reproducibility.
            fallback_config: Optional override for fallback uncertainties.
            tolerance: Optional convergence tolerance for early stopping. If
                abs(running_mean[-1] - running_mean[-window]) < tolerance,
                sampling stops early. Default: None (no early stopping).
            checkpoint_interval: Interval for storing running mean checkpoints
                (default 50). Trade-off between tracking granularity and performance.

        Returns:
            Dict with:
              mean_index, std_dev, ci_95, ci_lower, ci_upper: Core statistics.
              sample_count / samples: Actual samples used (may be < N if early stopping).
              standard_error: std_dev / sqrt(N), sampling uncertainty of the mean.
              convergence_delta: Absolute difference between last two checkpoint means.
              running_means: List of (sample_count, running_mean) at each checkpoint.
              converged_early: True if tolerance triggered early stopping.
        """
        if run_monte_carlo is None:
            raise ImportError(
                "Monte Carlo uncertainty requires ml_uncertainty. "
                "Ensure ml_uncertainty.py is available."
            )
        return run_monte_carlo(
            self,
            planet_data,
            star_data,
            N=N,
            seed=seed,
            fallback_config=fallback_config,
            tolerance=tolerance,
            checkpoint_interval=checkpoint_interval,
        )

    def get_earth_score(self, raw: bool = False) -> float:
        """
        Get Earth's reference score.
        
        Args:
            raw: If True, return raw 0-1 score; if False, return 100.0
        
        Returns:
            Earth score
        """
        return self.earth_raw_score if raw else 100.0
    
    def explain_prediction(self, features: dict) -> Dict:
        """
        Get detailed explanation of prediction.
        
        Args:
            features: Dict with NASA column names
        
        Returns:
            Dict with feature values, score, and metadata
        """
        
        # Build features
        feature_vector, meta = build_features(features, return_meta=True)
        
        # Predict
        raw_score = self._predict_raw(feature_vector)
        normalized_score = self.raw_to_display_index(raw_score)
        
        # Get feature importance from booster
        # For XGBoost Booster, use get_score()
        importance_dict = self.model.get_score(importance_type='weight')
        # Map feature indices (f0, f1, ...) to names
        importances = np.zeros(len(self.feature_names))
        for feat_idx, score in importance_dict.items():
            idx = int(feat_idx.replace('f', ''))
            if idx < len(importances):
                importances[idx] = score
        
        # Normalize importances to sum to 1
        if importances.sum() > 0:
            importances = importances / importances.sum()
        
        explanation = {
            "raw_score": float(raw_score),
            "earth_normalized_score": float(normalized_score),
            "feature_values": {
                name: float(feature_vector[i])
                for i, name in enumerate(self.feature_names)
            },
            "feature_importances": {
                name: float(importances[i])
                for i, name in enumerate(self.feature_names)
            },
            "imputed_fields": meta["imputed_fields"],
            "warnings": meta.get("warnings", [])
        }
        
        return explanation


# =============================================================================
# BACKWARDS COMPATIBILITY WRAPPER
# =============================================================================

def get_ml_calculator(version: str = "model_1") -> 'MLHabitabilityCalculator':
    """
    Factory function to get ML calculator.
    
    Args:
        version: "model_1" (default). "ml" is accepted as a compatibility alias.
    
    Returns:
        ML calculator instance
    """
    if version in {"model_1", "ml"}:
        return MLHabitabilityCalculator()
    raise ValueError(f"Unsupported model selector '{version}'. Use 'model_1'.")


if __name__ == "__main__":
    # Test with Earth
    print("\n" + "="*70)
    print("Testing ML Calculator with Earth")
    print("="*70)
    
    try:
        calc = MLHabitabilityCalculator()
        
        earth_features = {
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
            "st_lum": 1.0
        }
        
        # Test prediction
        score, meta = calc.predict(earth_features, return_raw=False, return_meta=True)
        
        print(f"\nEarth Prediction:")
        print(f"  Earth-normalized score: {score:.2f}%")
        print(f"  Raw score: {meta['raw_score']:.4f}")
        print(f"  Imputed fields: {meta['imputed_fields']}")
        
        # Test explanation
        explanation = calc.explain_prediction(earth_features)
        
        print(f"\nTop 5 Feature Importances:")
        sorted_importances = sorted(
            explanation["feature_importances"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for name, importance in sorted_importances[:5]:
            value = explanation["feature_values"][name]
            print(f"  {name:15s}: {importance:.4f} (value: {value:.4f})")
        
        print("\n[OK] ML calculator test passed")
    
    except Exception as e:
        print(f"\n[ERROR] ML calculator test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
