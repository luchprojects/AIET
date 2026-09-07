"""
Run AIET scientific validation suite (Kopparapu HZ + Earth reference index).

Run from the project root:
    python run_validate_kopparapu.py
"""

from __future__ import annotations

import sys

from src.ml.ml_validation import run_earth_reference_validation
from src.science.validate_kopparapu_hz import run_validation as run_kopparapu_validation
from src.science.validate_kopparapu_hz import format_report_text, export_report


def main() -> int:
    print("=" * 72)
    print("AIET VALIDATION SUITE")
    print("=" * 72)
    print()

    kopparapu_report = run_kopparapu_validation()
    kopparapu_text = format_report_text(kopparapu_report)
    print(kopparapu_text)
    text_path, json_path = export_report(kopparapu_report)
    print(f"\nReport saved: {text_path}")
    if json_path:
        print(f"JSON saved: {json_path}")

    print()
    print("=" * 72)
    print()

    _, earth_ok = run_earth_reference_validation()

    all_ok = kopparapu_report.all_passed and earth_ok
    print()
    print("=" * 72)
    print(f"SUITE OVERALL: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
