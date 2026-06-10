"""
AIET Sensitivity & Feature Influence Analysis Module

Quantifies feature influence on habitability predictions, detects dominant
variables, and strengthens model interpretability. This module provides
multiple complementary sensitivity analysis methods.

Methods Implemented:
    1. LOCAL ONE-AT-A-TIME (OAT): Perturb each feature individually and
       measure score change. Fast, interpretable, but ignores interactions.
    
    2. MONTE CARLO CORRELATION: Correlate sampled input values with output
       scores during MC uncertainty propagation. Captures natural variation.
    
    3. SHAP VALUES (optional): Tree-based SHAP explanations for XGBoost model.
       Captures feature interactions and provides theoretically grounded
       importance measures.

Interpretation Guidelines:
    - High |sensitivity|: Small changes in feature cause large score changes
    - Positive sensitivity: Increasing feature increases score
    - Negative sensitivity: Increasing feature decreases score
    - For habitability, expect:
        - pl_insol, pl_eqt, pl_orbsmax: High influence (habitable zone)
        - pl_rade, pl_masse: Moderate influence (planet type)
        - st_teff, st_lum: Moderate influence (star type)

Limitations:
    - OAT ignores feature interactions
    - MC correlations depend on input uncertainty distribution
    - SHAP requires shap library installation
    - Sensitivity ≠ causal importance (correlation, not causation)

Usage:
    from sensitivity_analysis import (
        compute_local_sensitivity,
        compute_full_sensitivity_report,
        export_sensitivity_report_json,
        export_sensitivity_bar_chart,
    )
    
    report = compute_full_sensitivity_report(calculator, planet_data)
    export_sensitivity_bar_chart(report, "exports/sensitivity.png")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np


# =============================================================================
# CANONICAL FEATURE ORDER (matches ml_features schema)
# =============================================================================

FEATURE_NAMES: List[str] = [
    "pl_rade",      # 0: Planet radius (Earth radii)
    "pl_masse",     # 1: Planet mass (Earth masses)
    "pl_orbper",    # 2: Orbital period (days)
    "pl_orbsmax",   # 3: Semi-major axis (AU)
    "pl_orbeccen",  # 4: Orbital eccentricity
    "pl_insol",     # 5: Insolation flux (Earth flux)
    "pl_eqt",       # 6: Equilibrium temperature (K)
    "pl_dens",      # 7: Density (g/cm³)
    "st_teff",      # 8: Stellar effective temperature (K)
    "st_mass",      # 9: Stellar mass (solar masses)
    "st_rad",       # 10: Stellar radius (solar radii)
    "st_lum",       # 11: Stellar luminosity (solar luminosities)
]

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "pl_rade": "Planet Radius",
    "pl_masse": "Planet Mass",
    "pl_orbper": "Orbital Period",
    "pl_orbsmax": "Semi-Major Axis",
    "pl_orbeccen": "Eccentricity",
    "pl_insol": "Insolation Flux",
    "pl_eqt": "Equilibrium Temperature",
    "pl_dens": "Planet Density",
    "st_teff": "Stellar Temperature",
    "st_mass": "Stellar Mass",
    "st_rad": "Stellar Radius",
    "st_lum": "Stellar Luminosity",
}


# =============================================================================
# SENSITIVITY REPORT DATACLASS
# =============================================================================

@dataclass
class SensitivityReport:
    """
    Structured sensitivity analysis results.
    
    Attributes:
        local_oat_raw: Raw OAT sensitivities (units: score change per unit feature change)
        local_oat_normalized: Normalized OAT sensitivities (sum of |S| = 1)
        local_oat_ranked: OAT sensitivities sorted by absolute value (descending)
        mc_correlations: Pearson correlations between features and scores (if computed)
        mc_correlations_ranked: MC correlations sorted by absolute value
        shap_importance: Mean absolute SHAP values per feature (if computed)
        shap_importance_ranked: SHAP importance sorted (descending)
        baseline_score: Deterministic score before perturbation
        perturbation_fraction: Perturbation size used for OAT
        planet_name: Name of planet analyzed
        timestamp: When analysis was performed
    """
    
    local_oat_raw: Dict[str, float] = field(default_factory=dict)
    local_oat_normalized: Dict[str, float] = field(default_factory=dict)
    local_oat_ranked: Dict[str, float] = field(default_factory=dict)
    
    mc_correlations: Dict[str, float] = field(default_factory=dict)
    mc_correlations_ranked: Dict[str, float] = field(default_factory=dict)
    
    shap_importance: Optional[Dict[str, float]] = None
    shap_importance_ranked: Optional[Dict[str, float]] = None
    
    baseline_score: float = 0.0
    perturbation_fraction: float = 0.05
    planet_name: str = ""
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def get_top_features(self, method: str = "oat", n: int = 5) -> List[Tuple[str, float]]:
        """Get top N most influential features by specified method."""
        if method == "oat":
            source = self.local_oat_ranked
        elif method == "mc":
            source = self.mc_correlations_ranked
        elif method == "shap" and self.shap_importance_ranked:
            source = self.shap_importance_ranked
        else:
            source = self.local_oat_ranked
        
        return list(source.items())[:n]
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Sensitivity Analysis Report - {self.planet_name or 'Unknown Planet'}",
            "=" * 60,
            f"Baseline Score: {self.baseline_score:.2f}",
            f"Perturbation: ±{self.perturbation_fraction * 100:.1f}%",
            "",
            "Top 5 Features (Local OAT):",
        ]
        
        for feat, sens in self.get_top_features("oat", 5):
            direction = "+" if sens > 0 else "-"
            lines.append(f"  {FEATURE_DESCRIPTIONS.get(feat, feat):25s}: {sens:+.4f} ({direction})")
        
        if self.mc_correlations_ranked:
            lines.append("")
            lines.append("Top 5 Features (MC Correlation):")
            for feat, corr in self.get_top_features("mc", 5):
                lines.append(f"  {FEATURE_DESCRIPTIONS.get(feat, feat):25s}: {corr:+.4f}")
        
        if self.shap_importance_ranked:
            lines.append("")
            lines.append("Top 5 Features (SHAP):")
            for feat, imp in self.get_top_features("shap", 5):
                lines.append(f"  {FEATURE_DESCRIPTIONS.get(feat, feat):25s}: {imp:.4f}")
        
        lines.append("")
        lines.append(f"Generated: {self.timestamp}")
        
        return "\n".join(lines)


# =============================================================================
# LOCAL ONE-AT-A-TIME SENSITIVITY
# =============================================================================

def compute_local_sensitivity(
    calculator: Any,
    planet_data: Dict[str, float],
    star_data: Optional[Dict[str, float]] = None,
    perturbation: float = 0.05,
    absolute_perturbations: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute local one-at-a-time (OAT) sensitivity for each feature.
    
    For each feature:
        1. Compute baseline deterministic score
        2. Perturb feature by ±perturbation (relative or absolute)
        3. Compute perturbed scores
        4. Calculate symmetric sensitivity: S_i = (score(+δ) - score(-δ)) / (2δ)
    
    Args:
        calculator: MLHabitabilityCalculator instance.
        planet_data: NASA-style planet dict.
        star_data: Optional star dict (merged with planet_data if None).
        perturbation: Relative perturbation fraction (default 5%).
        absolute_perturbations: Optional dict of absolute perturbation sizes
            for specific features (e.g., {"pl_orbeccen": 0.05}).
    
    Returns:
        Dict with:
            baseline_score: Original deterministic score
            raw_sensitivities: {feature: sensitivity} raw values
            normalized_sensitivities: {feature: normalized} values summing to 1
            ranked_sensitivities: {feature: sensitivity} sorted by |sensitivity|
            perturbation_used: Perturbation fraction used
    """
    if absolute_perturbations is None:
        absolute_perturbations = {
            "pl_orbeccen": 0.05,
        }
    
    merged = {**planet_data, **(star_data or {})}
    
    try:
        baseline_score = calculator.predict(merged, return_raw=False)
    except Exception as e:
        raise ValueError(f"Baseline prediction failed: {e}")
    
    raw_sensitivities = {}
    
    for feature in FEATURE_NAMES:
        if feature not in merged:
            raw_sensitivities[feature] = 0.0
            continue
        
        base_value = merged[feature]
        if base_value == 0 or not np.isfinite(base_value):
            raw_sensitivities[feature] = 0.0
            continue
        
        if feature in absolute_perturbations:
            delta = absolute_perturbations[feature]
        else:
            delta = abs(base_value) * perturbation
        
        if delta == 0:
            raw_sensitivities[feature] = 0.0
            continue
        
        perturbed_plus = merged.copy()
        perturbed_plus[feature] = base_value + delta
        
        perturbed_minus = merged.copy()
        perturbed_minus[feature] = base_value - delta
        
        try:
            score_plus = calculator.predict(perturbed_plus, return_raw=False)
            score_minus = calculator.predict(perturbed_minus, return_raw=False)
            
            sensitivity = (score_plus - score_minus) / (2 * delta)
            raw_sensitivities[feature] = float(sensitivity)
            
        except Exception:
            raw_sensitivities[feature] = 0.0
    
    total_abs = sum(abs(s) for s in raw_sensitivities.values())
    if total_abs > 0:
        normalized = {k: abs(v) / total_abs for k, v in raw_sensitivities.items()}
    else:
        normalized = {k: 0.0 for k in raw_sensitivities}
    
    ranked = dict(sorted(raw_sensitivities.items(), key=lambda x: abs(x[1]), reverse=True))
    
    return {
        "baseline_score": baseline_score,
        "raw_sensitivities": raw_sensitivities,
        "normalized_sensitivities": normalized,
        "ranked_sensitivities": ranked,
        "perturbation_used": perturbation,
    }


# =============================================================================
# MONTE CARLO CORRELATION SENSITIVITY
# =============================================================================

def compute_mc_input_score_correlations(
    calculator: Any,
    planet_data: Dict[str, float],
    star_data: Optional[Dict[str, float]] = None,
    N: int = 500,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute Pearson correlation between each input feature and output score
    during Monte Carlo sampling.
    
    This captures how natural variation in each input (given its uncertainty)
    correlates with output variation.
    
    Args:
        calculator: MLHabitabilityCalculator instance.
        planet_data: NASA-style planet dict (may include *err1, *err2).
        star_data: Optional star dict.
        N: Number of MC samples.
        seed: Random seed for reproducibility.
    
    Returns:
        Dict with:
            correlations: {feature: pearson_r}
            ranked_correlations: {feature: pearson_r} sorted by |r|
            sample_count: Number of samples used
    """
    try:
        from src.ml.ml_uncertainty import sample_inputs, DEFAULT_FALLBACK_UNCERTAINTY, SCHEMA_BOUNDS
        from src.ml.ml_features import build_features
    except ImportError as e:
        raise ImportError(f"MC correlation requires ml_uncertainty: {e}")
    
    merged = {**planet_data, **(star_data or {})}
    rng = np.random.default_rng(seed)
    
    sampled_list = sample_inputs(merged, dict(DEFAULT_FALLBACK_UNCERTAINTY), N, rng)
    
    feature_values = {feat: [] for feat in FEATURE_NAMES}
    scores = []
    
    earth_raw = calculator.earth_raw_score
    
    for sample in sampled_list:
        for feat in FEATURE_NAMES:
            if feat in sample:
                feature_values[feat].append(sample[feat])
            else:
                feature_values[feat].append(np.nan)
        
        try:
            feat_vec = build_features(sample, return_meta=False)
            raw_score = calculator._predict_raw(feat_vec)
            raw_score = np.clip(raw_score, 0.0, 1.0)
            if hasattr(calculator, "raw_to_display_index"):
                display_score = calculator.raw_to_display_index(raw_score)
            elif earth_raw > 1e-6:
                display_score = float(np.clip(100.0 * raw_score / earth_raw, 0.0, 100.0))
            else:
                display_score = float(np.clip(raw_score * 100.0, 0.0, 100.0))
            scores.append(display_score)
        except Exception:
            scores.append(np.nan)
    
    scores = np.array(scores)
    valid_scores = ~np.isnan(scores)
    
    correlations = {}
    
    for feat in FEATURE_NAMES:
        feat_arr = np.array(feature_values[feat])
        valid = valid_scores & ~np.isnan(feat_arr)
        
        if np.sum(valid) < 10:
            correlations[feat] = 0.0
            continue
        
        feat_valid = feat_arr[valid]
        scores_valid = scores[valid]
        
        if np.std(feat_valid) < 1e-10 or np.std(scores_valid) < 1e-10:
            correlations[feat] = 0.0
            continue
        
        corr_matrix = np.corrcoef(feat_valid, scores_valid)
        correlations[feat] = float(corr_matrix[0, 1])
    
    ranked = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
    
    return {
        "correlations": correlations,
        "ranked_correlations": ranked,
        "sample_count": int(np.sum(valid_scores)),
    }


# =============================================================================
# SHAP FEATURE IMPORTANCE (Optional)
# =============================================================================

def compute_shap_importance(
    calculator: Any,
    planet_data: Dict[str, float],
    star_data: Optional[Dict[str, float]] = None,
    n_background: int = 100,
    seed: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Compute SHAP feature importance using TreeExplainer for XGBoost model.
    
    Args:
        calculator: MLHabitabilityCalculator instance with XGBoost model.
        planet_data: NASA-style planet dict.
        star_data: Optional star dict.
        n_background: Number of background samples for SHAP.
        seed: Random seed.
    
    Returns:
        Dict with shap_values, mean_abs_shap, ranked_importance.
        Returns None if SHAP not available.
    """
    try:
        import shap
    except ImportError:
        return None
    
    try:
        from src.ml.ml_features import build_features
        from src.ml.ml_uncertainty import sample_inputs, DEFAULT_FALLBACK_UNCERTAINTY
    except ImportError:
        return None
    
    merged = {**planet_data, **(star_data or {})}
    
    try:
        target_features = build_features(merged, return_meta=False)
        target_features = target_features.reshape(1, -1)
    except Exception:
        return None
    
    rng = np.random.default_rng(seed)
    sampled_list = sample_inputs(merged, dict(DEFAULT_FALLBACK_UNCERTAINTY), n_background, rng)
    
    background_features = []
    for sample in sampled_list:
        try:
            feat = build_features(sample, return_meta=False)
            background_features.append(feat)
        except Exception:
            pass
    
    if len(background_features) < 10:
        return None
    
    background = np.array(background_features)
    
    try:
        explainer = shap.TreeExplainer(calculator.model, background)
        shap_values = explainer.shap_values(target_features)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        shap_values = shap_values.flatten()
        
        mean_abs_shap = {
            FEATURE_NAMES[i]: float(abs(shap_values[i]))
            for i in range(len(FEATURE_NAMES))
        }
        
        ranked = dict(sorted(mean_abs_shap.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "shap_values": {FEATURE_NAMES[i]: float(shap_values[i]) for i in range(len(FEATURE_NAMES))},
            "mean_abs_shap": mean_abs_shap,
            "ranked_importance": ranked,
        }
        
    except Exception as e:
        print(f"[Sensitivity] SHAP computation failed: {e}")
        return None


# =============================================================================
# FULL SENSITIVITY REPORT GENERATOR
# =============================================================================

def compute_full_sensitivity_report(
    calculator: Any,
    planet_data: Dict[str, float],
    star_data: Optional[Dict[str, float]] = None,
    planet_name: str = "",
    perturbation: float = 0.05,
    run_mc_correlation: bool = True,
    mc_samples: int = 500,
    run_shap: bool = True,
    seed: Optional[int] = None,
) -> SensitivityReport:
    """
    Compute full sensitivity analysis report using all available methods.
    
    Args:
        calculator: MLHabitabilityCalculator instance.
        planet_data: NASA-style planet dict.
        star_data: Optional star dict.
        planet_name: Name of planet for report.
        perturbation: Perturbation fraction for OAT (default 5%).
        run_mc_correlation: If True, compute MC correlations.
        mc_samples: Number of MC samples for correlation analysis.
        run_shap: If True, attempt SHAP analysis.
        seed: Random seed for reproducibility.
    
    Returns:
        SensitivityReport with all computed analyses.
    """
    oat_result = compute_local_sensitivity(
        calculator, planet_data, star_data, perturbation
    )
    
    mc_result = None
    if run_mc_correlation:
        try:
            mc_result = compute_mc_input_score_correlations(
                calculator, planet_data, star_data, N=mc_samples, seed=seed
            )
        except Exception as e:
            print(f"[Sensitivity] MC correlation failed: {e}")
    
    shap_result = None
    if run_shap:
        try:
            shap_result = compute_shap_importance(
                calculator, planet_data, star_data, seed=seed
            )
        except Exception as e:
            print(f"[Sensitivity] SHAP failed: {e}")
    
    report = SensitivityReport(
        local_oat_raw=oat_result["raw_sensitivities"],
        local_oat_normalized=oat_result["normalized_sensitivities"],
        local_oat_ranked=oat_result["ranked_sensitivities"],
        mc_correlations=mc_result["correlations"] if mc_result else {},
        mc_correlations_ranked=mc_result["ranked_correlations"] if mc_result else {},
        shap_importance=shap_result["mean_abs_shap"] if shap_result else None,
        shap_importance_ranked=shap_result["ranked_importance"] if shap_result else None,
        baseline_score=oat_result["baseline_score"],
        perturbation_fraction=perturbation,
        planet_name=planet_name,
    )
    
    return report


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_sensitivity_report_json(
    report: SensitivityReport,
    output_path: str,
    indent: int = 2,
) -> str:
    """
    Export SensitivityReport to JSON file.
    
    Args:
        report: SensitivityReport instance.
        output_path: Path to save JSON file.
        indent: JSON indentation.
    
    Returns:
        Path to saved file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    report_dict = asdict(report)
    
    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=indent, default=str)
    
    print(f"[Sensitivity] Exported report to: {output_path}")
    return output_path


def export_sensitivity_bar_chart(
    report: SensitivityReport,
    output_path: str,
    method: str = "oat",
    top_n: int = 12,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
) -> str:
    """
    Export publication-quality sensitivity bar chart.
    
    Args:
        report: SensitivityReport instance.
        output_path: Path to save PNG file.
        method: Which sensitivity method to plot ("oat", "mc", "shap").
        top_n: Number of features to show.
        figsize: Figure dimensions.
        title: Plot title (auto-generated if None).
    
    Returns:
        Path to saved figure.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for export_sensitivity_bar_chart()")
    
    if method == "oat":
        data = report.local_oat_ranked
        y_label = "Sensitivity (score change per unit)"
        default_title = "Local One-at-a-Time Sensitivity"
    elif method == "mc":
        data = report.mc_correlations_ranked
        y_label = "Pearson Correlation"
        default_title = "Monte Carlo Input-Score Correlation"
    elif method == "shap" and report.shap_importance_ranked:
        data = report.shap_importance_ranked
        y_label = "Mean |SHAP Value|"
        default_title = "SHAP Feature Importance"
    else:
        data = report.local_oat_ranked
        y_label = "Sensitivity"
        default_title = "Feature Sensitivity"
    
    if title is None:
        title = f"{default_title}: {report.planet_name}" if report.planet_name else default_title
    
    features = list(data.keys())[:top_n]
    values = [data[f] for f in features]
    
    labels = [FEATURE_DESCRIPTIONS.get(f, f) for f in features]
    
    colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in values]
    
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    
    ax.set_xlabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8)
    ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    
    info_text = f"Baseline: {report.baseline_score:.1f}"
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
            fontsize=8, ha='right', va='bottom', color='gray')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor='white', edgecolor='none',
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    print(f"[Sensitivity] Exported bar chart to: {output_path}")
    return output_path


def export_sensitivity_comparison_chart(
    report: SensitivityReport,
    output_path: str,
    figsize: Tuple[float, float] = (12, 8),
) -> str:
    """
    Export comparison chart showing OAT vs MC correlation rankings.
    
    Args:
        report: SensitivityReport instance.
        output_path: Path to save PNG file.
        figsize: Figure dimensions.
    
    Returns:
        Path to saved figure.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for export_sensitivity_comparison_chart()")
    
    if not report.mc_correlations_ranked:
        return export_sensitivity_bar_chart(report, output_path, method="oat")
    
    features = FEATURE_NAMES
    labels = [FEATURE_DESCRIPTIONS.get(f, f) for f in features]
    
    oat_values = [abs(report.local_oat_normalized.get(f, 0)) for f in features]
    
    mc_abs = {k: abs(v) for k, v in report.mc_correlations.items()}
    mc_total = sum(mc_abs.values()) or 1.0
    mc_values = [mc_abs.get(f, 0) / mc_total for f in features]
    
    x = np.arange(len(features))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    
    bars1 = ax.bar(x - width/2, oat_values, width, label='OAT Sensitivity', color='#3498db', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, mc_values, width, label='MC Correlation', color='#e74c3c', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Feature', fontsize=11)
    ax.set_ylabel('Normalized Importance', fontsize=11)
    ax.set_title(f'Sensitivity Method Comparison: {report.planet_name}', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor='white', edgecolor='none',
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    print(f"[Sensitivity] Exported comparison chart to: {output_path}")
    return output_path


# =============================================================================
# VALIDATION (run with: python -m sensitivity_analysis)
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("AIET Sensitivity & Feature Influence Analysis")
    print("=" * 70)
    
    try:
        from src.ml.ml_habitability import MLHabitabilityCalculator
        calc = MLHabitabilityCalculator()
    except ImportError as e:
        print(f"Could not import ML calculator: {e}")
        sys.exit(1)
    
    earth_data = {
        "pl_rade": 1.0, "pl_masse": 1.0, "pl_orbper": 365.25,
        "pl_orbsmax": 1.0, "pl_orbeccen": 0.0167, "pl_insol": 1.0,
        "pl_eqt": 255.0, "pl_dens": 5.51,
        "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0,
    }
    
    print("\n1. Testing Earth-like planet sensitivity:")
    print("-" * 50)
    
    earth_report = compute_full_sensitivity_report(
        calc, earth_data, planet_name="Earth", seed=42
    )
    
    print(earth_report.summary())
    
    top_oat = list(earth_report.local_oat_ranked.keys())[:3]
    expected_influential = {"pl_insol", "pl_orbsmax", "pl_eqt", "pl_rade", "pl_masse"}
    found_expected = any(f in expected_influential for f in top_oat)
    
    if found_expected:
        print(f"\n[OK] Top OAT features include expected influential variables")
    else:
        print(f"\n[WARN] Top features {top_oat} don't include expected {expected_influential}")
    
    print("\n2. Testing gas giant sensitivity:")
    print("-" * 50)
    
    jupiter_data = {
        "pl_rade": 11.21, "pl_masse": 317.8, "pl_orbper": 4333.0,
        "pl_orbsmax": 5.203, "pl_orbeccen": 0.049, "pl_insol": 0.037,
        "pl_eqt": 110.0, "pl_dens": 1.33,
        "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0,
    }
    
    jupiter_report = compute_full_sensitivity_report(
        calc, jupiter_data, planet_name="Jupiter", run_shap=False, seed=42
    )
    
    print(f"Baseline score: {jupiter_report.baseline_score:.2f}")
    print("Top 3 OAT features:")
    for feat, sens in jupiter_report.get_top_features("oat", 3):
        print(f"  {FEATURE_DESCRIPTIONS.get(feat, feat)}: {sens:+.4f}")
    
    print("\n3. Checking for NaN values:")
    print("-" * 50)
    
    has_nan = False
    for feat, val in earth_report.local_oat_raw.items():
        if np.isnan(val):
            print(f"[FAIL] NaN in OAT for {feat}")
            has_nan = True
    
    for feat, val in earth_report.mc_correlations.items():
        if np.isnan(val):
            print(f"[FAIL] NaN in MC correlation for {feat}")
            has_nan = True
    
    if not has_nan:
        print("[OK] No NaN values in sensitivity results")
    
    print("\n4. Testing exports:")
    print("-" * 50)
    
    os.makedirs("exports", exist_ok=True)
    
    json_path = export_sensitivity_report_json(earth_report, "exports/sensitivity_earth.json")
    if os.path.exists(json_path):
        print(f"[OK] JSON exported: {json_path}")
    
    try:
        chart_path = export_sensitivity_bar_chart(earth_report, "exports/sensitivity_earth.png")
        if os.path.exists(chart_path):
            print(f"[OK] Bar chart exported: {chart_path}")
        
        if earth_report.mc_correlations_ranked:
            comp_path = export_sensitivity_comparison_chart(earth_report, "exports/sensitivity_comparison.png")
            if os.path.exists(comp_path):
                print(f"[OK] Comparison chart exported: {comp_path}")
    except ImportError as e:
        print(f"[SKIP] matplotlib not available: {e}")
    
    print("\n" + "=" * 70)
    print("Interpretation Guidelines:")
    print("-" * 70)
    print("• High |sensitivity|: Feature strongly influences score")
    print("• Positive: Increasing feature increases habitability")
    print("• Negative: Increasing feature decreases habitability")
    print("• OAT: Fast, ignores interactions")
    print("• MC Correlation: Uses uncertainty distributions")
    print("• SHAP: Captures interactions (requires shap library)")
    print("=" * 70)
    
    sys.exit(0)
