"""
Kopparapu et al. (2013) habitable-zone validation for AIET.

Validates `src.physics.kopparapu_hz` against:
  - K13 Table 1 solar Recent Venus / Early Mars distances (0.75–1.77 AU)
  - K13 Table 3 S_eff at T_eff = 5780 K (coefficients in kopparapu_hz.py)
  - Internal flux–radius consistency (r = sqrt(L / S_eff))
  - Solar-system placement sanity (Earth/Mars in HZ, Venus outside)
  - Multi-star spot checks at F/G/K/M effective temperatures

Run from the project root:
    python run_validate_kopparapu.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

from src.physics.kopparapu_hz import (
    T_EFF_MAX_K,
    T_EFF_MIN_K,
    kopparapu_hz_boundaries_au,
    planet_in_kopparapu_hz,
)


# ---------------------------------------------------------------------------
# Reference anchors (Kopparapu et al. 2013, ApJ 765, 131)
# ---------------------------------------------------------------------------

# Table 1 — Sun, conservative empirical pair used by AIET (Recent Venus / Early Mars).
K13_TABLE1_SUN_RV_EM = {
    "inner_au": 0.75,
    "outer_au": 1.77,
    "luminosity_solar": 1.0,
    "t_eff_k": 5780.0,
    "source": "Kopparapu et al. 2013, Table 1 (Recent Venus / Early Mars)",
}

# Table 3 — S_eff at T_eff = 5780 K (T* = 0); matches coefficients in kopparapu_hz.py.
K13_TABLE3_SOLAR_S_EFF = {
    "recent_venus": 1.7753,
    "early_mars": 0.3179,
}

# Spot checks at representative F/G/K/M temperatures (precomputed from this module).
MULTI_STAR_CASES = [
    {
        "name": "F0 dwarf",
        "luminosity_solar": 2.5,
        "t_eff_k": 7200.0,
        "inner_au": 1.1299,
        "outer_au": 2.5369,
    },
    {
        "name": "K0 dwarf",
        "luminosity_solar": 0.4,
        "t_eff_k": 4800.0,
        "inner_au": 0.4933,
        "outer_au": 1.2214,
    },
    {
        "name": "M0 dwarf",
        "luminosity_solar": 0.05,
        "t_eff_k": 3800.0,
        "inner_au": 0.1799,
        "outer_au": 0.4662,
    },
    {
        "name": "TRAPPIST-1",
        "luminosity_solar": 5.53e-4,
        "t_eff_k": 2559.0,
        "inner_au": 0.01936,
        "outer_au": 0.05268,
    },
]

SOLAR_SYSTEM_PLACEMENT = [
    ("Earth", 1.0, True),
    ("Mars", 1.524, True),
    ("Venus", 0.723, False),
    ("Mercury", 0.387, False),
]

DEFAULT_AU_TOLERANCE_FRAC = 0.01  # 1% — paper Table 1 values are rounded to 2 decimals
DEFAULT_S_EFF_TOLERANCE = 1e-4


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class ValidationReport:
    timestamp: str
    hz_model: str = "kopparapu_2013_conservative_rv_em"
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(CheckResult(name=name, passed=passed, detail=detail))


def _rel_error(observed: float, expected: float) -> float:
    if expected == 0.0:
        return abs(observed)
    return abs(observed - expected) / abs(expected)


def _check_near(
    observed: float,
    expected: float,
    tolerance: float,
    label: str,
) -> Tuple[bool, str]:
    err = abs(observed - expected)
    rel = _rel_error(observed, expected)
    passed = err <= tolerance or rel <= DEFAULT_AU_TOLERANCE_FRAC
    return passed, (
        f"{label}: got {observed:.6g}, expected {expected:.6g} "
        f"(abs err {err:.3g}, rel err {rel:.3g})"
    )


def validate_table3_solar_s_eff(report: ValidationReport) -> None:
    """S_eff at T_eff = 5780 K must match Table 3 constants."""
    _, _, s_inner, s_outer = kopparapu_hz_boundaries_au(
        K13_TABLE1_SUN_RV_EM["luminosity_solar"],
        K13_TABLE1_SUN_RV_EM["t_eff_k"],
    )

    ok_in, detail_in = _check_near(
        s_inner,
        K13_TABLE3_SOLAR_S_EFF["recent_venus"],
        DEFAULT_S_EFF_TOLERANCE,
        "S_eff inner (Recent Venus)",
    )
    ok_out, detail_out = _check_near(
        s_outer,
        K13_TABLE3_SOLAR_S_EFF["early_mars"],
        DEFAULT_S_EFF_TOLERANCE,
        "S_eff outer (Early Mars)",
    )
    passed = ok_in and ok_out
    report.add(
        "Table 3 solar S_eff",
        passed,
        f"{detail_in}; {detail_out}",
    )


def validate_table1_sun_distances(report: ValidationReport) -> None:
    """Sun HZ radii must match K13 Table 1 within tolerance."""
    l_star = K13_TABLE1_SUN_RV_EM["luminosity_solar"]
    t_eff = K13_TABLE1_SUN_RV_EM["t_eff_k"]
    inner_au, outer_au, _, _ = kopparapu_hz_boundaries_au(l_star, t_eff)

    ok_in, detail_in = _check_near(
        inner_au,
        K13_TABLE1_SUN_RV_EM["inner_au"],
        K13_TABLE1_SUN_RV_EM["inner_au"] * DEFAULT_AU_TOLERANCE_FRAC,
        "inner AU",
    )
    ok_out, detail_out = _check_near(
        outer_au,
        K13_TABLE1_SUN_RV_EM["outer_au"],
        K13_TABLE1_SUN_RV_EM["outer_au"] * DEFAULT_AU_TOLERANCE_FRAC,
        "outer AU",
    )
    passed = ok_in and ok_out
    report.add(
        "Table 1 Sun RV/EM distances",
        passed,
        f"{detail_in}; {detail_out}",
    )


def validate_flux_radius_consistency(report: ValidationReport) -> None:
    """r_AU must equal sqrt(L / S_eff) for every boundary."""
    cases = [
        ("Sun", 1.0, 5780.0),
        ("M dwarf", 0.01, 2600.0),
        ("F dwarf", 2.5, 7200.0),
    ]
    details: List[str] = []
    passed = True
    for name, l_star, t_eff in cases:
        inner_au, outer_au, s_in, s_out = kopparapu_hz_boundaries_au(l_star, t_eff)
        expected_inner = math.sqrt(l_star / s_in)
        expected_outer = math.sqrt(l_star / s_out)
        ok_in = math.isclose(inner_au, expected_inner, rel_tol=1e-9, abs_tol=1e-12)
        ok_out = math.isclose(outer_au, expected_outer, rel_tol=1e-9, abs_tol=1e-12)
        if not (ok_in and ok_out):
            passed = False
        details.append(
            f"{name}: inner {inner_au:.6g} vs sqrt(L/S_in) {expected_inner:.6g}, "
            f"outer {outer_au:.6g} vs sqrt(L/S_out) {expected_outer:.6g}"
        )
    report.add("Flux-radius consistency (r = sqrt(L/S_eff))", passed, "; ".join(details))


def validate_inner_outer_ordering(report: ValidationReport) -> None:
    """Outer HZ edge must be farther than inner edge for all test stars."""
    passed = True
    details: List[str] = []
    for case in MULTI_STAR_CASES + [{"name": "Sun", **K13_TABLE1_SUN_RV_EM}]:
        inner_au, outer_au, _, _ = kopparapu_hz_boundaries_au(
            case["luminosity_solar"], case["t_eff_k"]
        )
        ok = outer_au > inner_au
        passed = passed and ok
        details.append(f"{case['name']}: {inner_au:.5f} < {outer_au:.5f} -> {ok}")
    report.add("Inner < outer ordering", passed, "; ".join(details))


def validate_teff_clamping(report: ValidationReport) -> None:
    """Teff outside 2600–7200 K should be clamped, not rejected."""
    # Below minimum: should match T_eff = 2600 K result
    inner_lo, outer_lo, _, _ = kopparapu_hz_boundaries_au(1.0, 1000.0)
    inner_ref, outer_ref, _, _ = kopparapu_hz_boundaries_au(1.0, T_EFF_MIN_K)
    ok_lo = math.isclose(inner_lo, inner_ref, rel_tol=1e-9) and math.isclose(
        outer_lo, outer_ref, rel_tol=1e-9
    )

    # Above maximum: should match T_eff = 7200 K result
    inner_hi, outer_hi, _, _ = kopparapu_hz_boundaries_au(1.0, 10000.0)
    inner_ref_hi, outer_ref_hi, _, _ = kopparapu_hz_boundaries_au(1.0, T_EFF_MAX_K)
    ok_hi = math.isclose(inner_hi, inner_ref_hi, rel_tol=1e-9) and math.isclose(
        outer_hi, outer_ref_hi, rel_tol=1e-9
    )

    passed = ok_lo and ok_hi
    report.add(
        "Teff clamping (2600-7200 K)",
        passed,
        f"below min: {ok_lo}; above max: {ok_hi}",
    )


def validate_multi_star_cases(report: ValidationReport) -> None:
    """Spot-check F/G/K/M and TRAPPIST-1 against precomputed reference radii."""
    details: List[str] = []
    passed = True
    for case in MULTI_STAR_CASES:
        inner_au, outer_au, _, _ = kopparapu_hz_boundaries_au(
            case["luminosity_solar"], case["t_eff_k"]
        )
        ok_in, detail_in = _check_near(
            inner_au,
            case["inner_au"],
            case["inner_au"] * DEFAULT_AU_TOLERANCE_FRAC,
            f"{case['name']} inner",
        )
        ok_out, detail_out = _check_near(
            outer_au,
            case["outer_au"],
            case["outer_au"] * DEFAULT_AU_TOLERANCE_FRAC,
            f"{case['name']} outer",
        )
        case_ok = ok_in and ok_out
        passed = passed and case_ok
        details.append(f"{case['name']}: {'OK' if case_ok else 'FAIL'} ({detail_in}; {detail_out})")
    report.add("Multi-star spot checks", passed, " | ".join(details))


def validate_solar_system_placement(report: ValidationReport) -> None:
    """Earth and Mars in HZ; Venus and Mercury outside (Sun-like star)."""
    details: List[str] = []
    passed = True
    for planet, au, expected_in_hz in SOLAR_SYSTEM_PLACEMENT:
        in_hz = planet_in_kopparapu_hz(au, 1.0, 5780.0)
        ok = in_hz == expected_in_hz
        passed = passed and ok
        details.append(f"{planet} @ {au} AU: in_hz={in_hz}, expected={expected_in_hz}")
    report.add("Solar system HZ placement", passed, "; ".join(details))


def run_validation() -> ValidationReport:
    report = ValidationReport(timestamp=datetime.now().isoformat())

    validate_table3_solar_s_eff(report)
    validate_table1_sun_distances(report)
    validate_flux_radius_consistency(report)
    validate_inner_outer_ordering(report)
    validate_teff_clamping(report)
    validate_multi_star_cases(report)
    validate_solar_system_placement(report)

    return report


def format_report_text(report: ValidationReport) -> str:
    lines: List[str] = [
        "Kopparapu (2013) Habitable Zone Validation",
        f"Model: {report.hz_model}",
        f"Timestamp: {report.timestamp}",
        "",
    ]
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"[{status}] {check.name}")
        lines.append(f"  {check.detail}")
        lines.append("")
    lines.append(
        f"Overall: {'PASS' if report.all_passed else 'FAIL'} "
        f"({sum(c.passed for c in report.checks)}/{len(report.checks)} checks)"
    )
    return "\n".join(lines)


def export_report(
    report: ValidationReport,
    log_dir: str = "logs",
    export_json: bool = True,
) -> Tuple[str, Optional[str]]:
    os.makedirs(log_dir, exist_ok=True)
    text_path = os.path.join(log_dir, "kopparapu_validation_report.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(format_report_text(report))

    json_path: Optional[str] = None
    if export_json:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(log_dir, f"kopparapu_validation_{ts}.json")
        payload = {
            "timestamp": report.timestamp,
            "hz_model": report.hz_model,
            "all_passed": report.all_passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in report.checks
            ],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return text_path, json_path


def main() -> int:
    report = run_validation()
    text = format_report_text(report)
    print(text)

    text_path, json_path = export_report(report)
    print(f"\nReport saved: {text_path}")
    if json_path:
        print(f"JSON saved: {json_path}")

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
