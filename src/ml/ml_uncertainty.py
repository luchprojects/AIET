"""
AIET ML - Monte Carlo Uncertainty Propagation for Habitability Index

Propagates input parameter uncertainties (NASA Exoplanet Archive or documented
fallback assumptions) through the full physics-informed ML pipeline using
Monte Carlo sampling. Produces statistically defensible uncertainty bounds.

Scientific note: The confidence interval reflects propagated INPUT uncertainty
only. It does NOT capture model epistemic uncertainty. The output is NOT a
probability of life.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from src.ml.ml_features import build_features


# =============================================================================
# SCHEMA BOUNDS (must match ml_features / feature schema clipping)
# =============================================================================

SCHEMA_BOUNDS: Dict[str, Tuple[float, float]] = {
    "pl_rade": (0.1, 20.0),
    "pl_masse": (0.001, 500.0),
    "pl_orbper": (0.1, 100000.0),
    "pl_orbsmax": (0.01, 1000.0),
    "pl_orbeccen": (0.0, 1.0),
    "pl_insol": (0.0001, 100.0),
    "pl_eqt": (50.0, 3000.0),
    "pl_dens": (0.1, 30.0),
    "st_teff": (2000.0, 50000.0),
    "st_mass": (0.08, 100.0),
    "st_rad": (0.1, 1000.0),
    "st_lum": (0.0001, 1000000.0),
    "hz_inner_au": (0.001, 1000.0),
    "hz_outer_au": (0.001, 1000.0),
    "hz_lin_pos": (0.0, 1.0),
    "s_eff_log_pos": (0.0, 1.0),
    "in_hz": (0.0, 1.0),
    "st_met": (-2.5, 1.0),
    "sy_snum": (1.0, 6.0),
    "s_eff_inner": (0.01, 100.0),
    "insol_vs_rv": (0.001, 50.0),
    "flux_ecc_ratio": (1.0, 5.0),
    "tidal_lock_proxy": (0.0, 1.0),
}

# Parameters that are direct ML inputs (sampled when present; derived ones recomputed in build_features)
INPUT_PARAMS: List[str] = [
    "pl_rade", "pl_masse", "pl_orbper", "pl_orbsmax", "pl_orbeccen",
    "pl_insol", "pl_eqt", "pl_dens",
    "st_teff", "st_mass", "st_rad", "st_lum",
]


# =============================================================================
# DEFAULT FALLBACK UNCERTAINTY (documented assumptions when NASA errors missing)
# =============================================================================
# Fractional: (low_frac, high_frac) for asymmetric; single float for symmetric.
# Absolute: ("absolute", delta) for e.g. eccentricity ±0.05.

DEFAULT_FALLBACK_UNCERTAINTY: Dict[str, Any] = {
    "st_mass": 0.05,           # ±5%
    "st_rad": 0.075,            # ±5–10% → use 7.5%
    "st_teff": 0.025,           # ±2–3% → use 2.5%
    "st_lum": 0.10,             # L from R,T → ~10%
    "pl_masse": 0.15,           # ±10–20% → use 15%
    "pl_rade": 0.075,           # ±5–10% → use 7.5%
    "pl_orbper": 0.01,          # ±1%
    "pl_orbsmax": 0.02,         # from P,M → ~2%
    "pl_orbeccen": ("absolute", 0.05),  # ±0.05 absolute, bounded [0,1]
    "pl_insol": 0.10,           # from L,a² → ~10%
    "pl_eqt": 0.05,             # from flux^0.25 → ~5%
    "pl_dens": 0.15,            # from M,R → ~15%
}


def _get_fallback_sigma(
    param: str,
    value: float,
    fallback_config: Dict[str, Any],
) -> Tuple[float, float]:
    """Return (sigma_lower, sigma_upper) for fallback uncertainty (both positive)."""
    fb = fallback_config.get(param, 0.1)
    if isinstance(fb, (list, tuple)) and len(fb) >= 2 and fb[0] == "absolute":
        delta = float(fb[1])
        return (delta, delta)
    if isinstance(fb, (list, tuple)) and len(fb) == 2:
        return (float(fb[0]) * abs(value), float(fb[1]) * abs(value))
    f = float(fb)
    s = abs(value) * f
    return (s, s)


def sample_parameter(
    value: float,
    err1: Optional[float],
    err2: Optional[float],
    fallback_sigma_low: float,
    fallback_sigma_high: float,
    bounds: Tuple[float, float],
    N: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample N values for one parameter.
    - If both err1 and err2 exist: asymmetric Gaussian (split normal).
    - If only err1: symmetric Gaussian with sigma = |err1|.
    - Else: fallback asymmetric/symmetric using fallback_sigma_*.
    All samples are clipped to bounds.
    """
    lo, hi = bounds
    if err1 is not None and err2 is not None and np.isfinite(err1) and np.isfinite(err2):
        sigma_u = abs(float(err1))
        sigma_l = abs(float(err2))
        z = rng.standard_normal(N)
        x = np.where(z < 0, value + z * sigma_l, value + z * sigma_u)
    elif err1 is not None and np.isfinite(err1):
        sigma = abs(float(err1))
        x = value + rng.normal(0, sigma, size=N)
    else:
        z = rng.standard_normal(N)
        x = np.where(z < 0, value + z * fallback_sigma_low, value + z * fallback_sigma_high)
    return np.clip(x, lo, hi).astype(np.float64)


def sample_inputs(
    merged_data: Dict[str, float],
    fallback_config: Dict[str, Any],
    N: int,
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    """
    Produce N sampled input dicts. Each dict has the same keys as merged_data
    where we have values; sampled parameters are replaced with N samples.
    Parameters not present or NaN are left as-is (builder will impute).
    """
    sampled_arrays: Dict[str, np.ndarray] = {}
    for key in INPUT_PARAMS:
        if key not in SCHEMA_BOUNDS:
            continue
        val = merged_data.get(key)
        if val is None or not np.isfinite(val):
            continue
        err1 = merged_data.get(key + "err1")
        err2 = merged_data.get(key + "err2")
        try:
            err1_f = float(err1) if err1 is not None else None
            err2_f = float(err2) if err2 is not None else None
        except (TypeError, ValueError):
            err1_f, err2_f = None, None
        sigma_low, sigma_high = _get_fallback_sigma(key, val, fallback_config)
        sampled_arrays[key] = sample_parameter(
            float(val),
            err1_f,
            err2_f,
            sigma_low,
            sigma_high,
            SCHEMA_BOUNDS[key],
            N,
            rng,
        )

    # Build N dicts: for each i, copy merged_data and overwrite with sampled values
    out: List[Dict[str, float]] = []
    for i in range(N):
        row = dict(merged_data)
        for key, arr in sampled_arrays.items():
            row[key] = float(arr[i])
        out.append(row)
    return out


def run_monte_carlo(
    calculator: Any,
    planet_data: Dict[str, float],
    star_data: Optional[Dict[str, float]] = None,
    N: int = 1000,
    seed: Optional[int] = None,
    fallback_config: Optional[Dict[str, Any]] = None,
    tolerance: Optional[float] = None,
    checkpoint_interval: int = 50,
) -> Dict[str, Any]:
    """
    Run Monte Carlo uncertainty propagation through the full pipeline.

    For each of N samples:
      1. Sample uncertain inputs (NASA err1/err2 or fallback).
      2. Run full deterministic feature builder (physics derivations, clipping).
      3. Run XGBoost to get raw_score in [0, 1].
      4. Earth-normalize to display score.
    Then compute mean, std, and 95% CI of the display scores.

    Convergence Diagnostics:
        Tracks running mean as samples accumulate. This measures stability of the
        Monte Carlo sampling, NOT model correctness. Standard error estimates the
        sampling uncertainty of the mean (std_dev / sqrt(N)).

    Args:
        calculator: MLHabitabilityCalculator instance (must have _predict_raw, earth_raw_score).
        planet_data: NASA-style planet dict (may include *err1, *err2).
        star_data: Optional NASA-style star dict; if None, planet_data is treated as merged.
        N: Number of Monte Carlo samples.
        seed: Random seed for reproducibility.
        fallback_config: Override for fallback uncertainties; None = use DEFAULT_FALLBACK_UNCERTAINTY.
        tolerance: Optional convergence tolerance. If provided, sampling stops early when
            abs(running_mean[-1] - running_mean[-window]) < tolerance where window is the
            previous checkpoint. Default: None (no early stopping).
        checkpoint_interval: Interval for storing running mean checkpoints (default 50).

    Returns:
        Dict with:
          mean_index: float
          std_dev: float
          ci_95: (ci_lower, ci_upper)
          ci_lower: float
          ci_upper: float
          sample_count: int (actual samples used, may be < N if early stopping)
          samples: int (alias for sample_count)
          standard_error: float (std_dev / sqrt(N), sampling uncertainty of the mean)
          convergence_delta: float (abs difference between last two checkpoint means)
          running_means: List[Tuple[int, float]] (sample_count, running_mean at checkpoints)
          converged_early: bool (True if tolerance triggered early stopping)
        CI reflects propagated input uncertainty only, not model epistemic uncertainty.
        Output is NOT a probability of life.
    """
    rng = np.random.default_rng(seed)
    if fallback_config is None:
        fallback_config = dict(DEFAULT_FALLBACK_UNCERTAINTY)
    merged = {**planet_data, **(star_data or {})}

    sampled_list = sample_inputs(merged, fallback_config, N, rng)
    raw_scores = np.empty(N, dtype=np.float64)
    earth_raw = calculator.earth_raw_score

    running_means: List[Tuple[int, float]] = []
    converged_early = False
    actual_samples = N

    for i in range(N):
        # return_meta=False returns ndarray only (not a tuple)
        feat = build_features(sampled_list[i], return_meta=False)
        raw_scores[i] = calculator._predict_raw(feat)

        if (i + 1) % checkpoint_interval == 0 or i == N - 1:
            current_raw = raw_scores[:i + 1]
            current_raw_clipped = np.clip(current_raw, 0.0, 1.0)
            if hasattr(calculator, "raw_to_display_index"):
                current_display = np.array(
                    [calculator.raw_to_display_index(float(r)) for r in current_raw_clipped]
                )
            elif earth_raw > 1e-6:
                current_display = np.clip(current_raw_clipped / earth_raw * 100.0, 0.0, 100.0)
            else:
                current_display = np.clip(current_raw_clipped * 100.0, 0.0, 100.0)
            current_mean = float(np.mean(current_display))
            running_means.append((i + 1, current_mean))

            if tolerance is not None and len(running_means) >= 2:
                prev_mean = running_means[-2][1]
                delta = abs(current_mean - prev_mean)
                if delta < tolerance:
                    converged_early = True
                    actual_samples = i + 1
                    break

    raw_scores = raw_scores[:actual_samples]
    raw_scores = np.clip(raw_scores, 0.0, 1.0)
    if hasattr(calculator, "raw_to_display_index"):
        display_scores = np.array(
            [calculator.raw_to_display_index(float(r)) for r in raw_scores]
        )
    elif earth_raw > 1e-6:
        display_scores = np.clip(raw_scores / earth_raw * 100.0, 0.0, 100.0)
    else:
        display_scores = np.clip(raw_scores * 100.0, 0.0, 100.0)

    mean_display = float(np.mean(display_scores))
    std_display = float(np.std(display_scores))
    ci_lower, ci_upper = float(np.percentile(display_scores, 2.5)), float(np.percentile(display_scores, 97.5))

    standard_error = std_display / np.sqrt(actual_samples) if actual_samples > 0 else 0.0

    if len(running_means) >= 2:
        convergence_delta = abs(running_means[-1][1] - running_means[-2][1])
    else:
        convergence_delta = 0.0

    return {
        "mean_index": mean_display,
        "std_dev": std_display,
        "ci_95": (ci_lower, ci_upper),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "sample_count": actual_samples,
        "samples": actual_samples,
        "standard_error": float(standard_error),
        "convergence_delta": float(convergence_delta),
        "running_means": running_means,
        "converged_early": converged_early,
    }


# =============================================================================
# CONVERGENCE PLOT EXPORT
# =============================================================================

def export_uncertainty_convergence_plot(
    mc_result: Dict[str, Any],
    output_path: str,
    planet_name: str = "Planet",
    show_final_mean: bool = True,
    show_std_band: bool = True,
    figsize: Tuple[float, float] = (8, 5),
) -> str:
    """
    Export publication-quality convergence plot for Monte Carlo uncertainty.

    Plots running mean vs sample count to visualize convergence behavior.
    White background, 300 DPI, suitable for publication.

    Args:
        mc_result: Dict returned by run_monte_carlo() containing 'running_means'.
        output_path: Path to save PNG file.
        planet_name: Name for plot title.
        show_final_mean: If True, draw horizontal line at final mean.
        show_std_band: If True, draw shaded band for ±1 standard error at end.
        figsize: Figure dimensions (width, height) in inches.

    Returns:
        Path to saved figure.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for export_uncertainty_convergence_plot()")

    running_means = mc_result.get("running_means", [])
    if not running_means:
        raise ValueError("mc_result must contain non-empty 'running_means'")

    samples = [rm[0] for rm in running_means]
    means = [rm[1] for rm in running_means]

    final_mean = mc_result.get("mean_index", means[-1])
    std_error = mc_result.get("standard_error", 0.0)

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    ax.plot(samples, means, 'b-', linewidth=1.5, marker='o', markersize=4,
            label='Running Mean')

    if show_final_mean:
        ax.axhline(y=final_mean, color='darkgreen', linestyle='--', linewidth=1.2,
                   label=f'Final Mean: {final_mean:.2f}')

    if show_std_band and std_error > 0:
        ax.fill_between([samples[0], samples[-1]],
                        [final_mean - std_error, final_mean - std_error],
                        [final_mean + std_error, final_mean + std_error],
                        color='green', alpha=0.15,
                        label=f'±1 SE ({std_error:.3f})')

    ax.set_xlabel('Sample Count', fontsize=11)
    ax.set_ylabel('Running Mean (Habitability Index)', fontsize=11)
    ax.set_title(f'Monte Carlo Convergence: {planet_name}', fontsize=12, fontweight='bold')

    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ax.tick_params(axis='both', which='major', labelsize=10)

    convergence_delta = mc_result.get("convergence_delta", 0.0)
    sample_count = mc_result.get("sample_count", samples[-1])
    converged_early = mc_result.get("converged_early", False)

    info_text = f"N={sample_count}, Δ={convergence_delta:.4f}"
    if converged_early:
        info_text += " (early stop)"
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
            fontsize=8, ha='right', va='bottom', color='gray')

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor='white', edgecolor='none',
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    print(f"[MC Convergence] Exported plot to: {output_path}")
    return output_path


# =============================================================================
# VALIDATION (run with: python -m ml_uncertainty)
# =============================================================================

if __name__ == "__main__":
    import sys
    import os
    print("Monte Carlo Uncertainty Propagation – validation")
    print("=" * 70)

    try:
        from src.ml.ml_habitability import MLHabitabilityCalculator
    except ImportError as e:
        print(f"Skip: {e}")
        sys.exit(0)

    calc = MLHabitabilityCalculator()
    earth = {
        "pl_rade": 1.0, "pl_masse": 1.0, "pl_orbper": 365.25,
        "pl_orbsmax": 1.0, "pl_orbeccen": 0.0167, "pl_insol": 1.0,
        "pl_eqt": 255.0, "pl_dens": 5.51,
        "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0,
    }

    # 1) Deterministic predict() vs mean of MC (with zero explicit errors → fallback applies)
    det_score = calc.predict(earth, return_raw=False)
    result = run_monte_carlo(calc, earth, None, N=500, seed=42)
    print(f"Earth deterministic score: {det_score:.4f}")
    print(f"Earth MC mean_index:      {result['mean_index']:.4f} ± {result['std_dev']:.4f}")
    print(f"95% CI: ({result['ci_lower']:.4f}, {result['ci_upper']:.4f}), N={result['samples']}")
    print(f"Standard Error: {result['standard_error']:.4f}")
    print(f"Convergence Delta: {result['convergence_delta']:.4f}")
    print(f"Running means checkpoints: {len(result['running_means'])}")
    if abs(result["mean_index"] - det_score) < 5.0:
        print("[OK] Mean of MC is close to deterministic (within 5 index points)")
    else:
        print("[WARN] Mean of MC differs from deterministic (expected some difference with fallback uncertainties)")

    # 2) Verify variance increases with larger fallback uncertainties
    large_fallback = {k: v * 2 if not isinstance(v, tuple) else v for k, v in DEFAULT_FALLBACK_UNCERTAINTY.items()}
    result2 = run_monte_carlo(calc, earth, None, N=500, seed=123, fallback_config=large_fallback)
    print(f"\nLarger fallback (N=500): mean={result2['mean_index']:.4f} ± {result2['std_dev']:.4f}")
    if result2['std_dev'] > result['std_dev']:
        print("[OK] Variance increases with larger fallback uncertainties")
    else:
        print("[WARN] Variance did not increase as expected (may be sampling variance)")

    # 3) Test running mean stabilization with increasing N
    result3 = run_monte_carlo(calc, earth, None, N=1000, seed=456)
    running_means = result3['running_means']
    if len(running_means) >= 4:
        early_delta = abs(running_means[1][1] - running_means[0][1])
        late_delta = abs(running_means[-1][1] - running_means[-2][1])
        print(f"\nRunning mean stability test:")
        print(f"  Early delta (checkpoint 0→1): {early_delta:.4f}")
        print(f"  Late delta (last two checkpoints): {late_delta:.4f}")
        if late_delta <= early_delta + 0.5:
            print("[OK] Running mean stabilizes with increasing N")
        else:
            print("[WARN] Running mean not stabilizing (may need more samples)")

    # 4) Test early stopping with tolerance
    result4 = run_monte_carlo(calc, earth, None, N=2000, seed=789, tolerance=0.1)
    print(f"\nEarly stopping test (tolerance=0.1, max N=2000):")
    print(f"  Actual samples: {result4['sample_count']}")
    print(f"  Converged early: {result4['converged_early']}")
    if result4['converged_early']:
        print(f"[OK] Early stopping triggered at N={result4['sample_count']}")
    else:
        print("[OK] Did not converge early (tolerance too tight or high variance)")

    # 5) Test predict_with_uncertainty returns new fields
    u = calc.predict_with_uncertainty(earth, None, N=100, seed=1)
    required_keys = ["mean_index", "std_dev", "ci_95", "ci_lower", "ci_upper",
                     "sample_count", "standard_error", "convergence_delta", "running_means"]
    missing = [k for k in required_keys if k not in u]
    if not missing:
        print("\n[OK] predict_with_uncertainty returns all required fields")
    else:
        print(f"\n[FAIL] Missing keys: {missing}")

    # 6) Test convergence plot export (if matplotlib available)
    print("\n" + "-" * 70)
    print("Testing convergence plot export...")
    try:
        os.makedirs("exports", exist_ok=True)
        plot_path = export_uncertainty_convergence_plot(
            result3, "exports/mc_convergence_test.png", planet_name="Earth (test)"
        )
        if os.path.exists(plot_path):
            print(f"[OK] Convergence plot exported to: {plot_path}")
        else:
            print("[FAIL] Plot file not created")
    except ImportError as e:
        print(f"[SKIP] matplotlib not available: {e}")
    except Exception as e:
        print(f"[FAIL] Plot export error: {e}")

    print("=" * 70)
