"""
Launcher for the AIET UI in `src/ui/main_window.py`.

Run from the project root:
    python .\run_main_window.py
"""

from __future__ import annotations


def main() -> int:
    # Import from the package-style tree (`src/...`).
    # This file must be run from the repository root so `src` is importable.
    from src.simulation_engine import SimulationEngine
    from src.ui.main_window import SolarSystemVisualizer

    engine = SimulationEngine()
    ui = SolarSystemVisualizer()

    running = True
    while running:
        running = ui.handle_events()
        ui.render(engine)

    try:
        import pygame

        pygame.quit()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

