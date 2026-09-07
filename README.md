# AIET v2.0 — Artificial Intelligence for Extraterrestrial

AIET is a physics-informed machine learning and visualization platform for the exploratory analysis of exoplanet habitability using data from the NASA Exoplanet Archive.

> **Development:** `main` is the working tree (v2 in progress). Official downloads are GitHub Releases (`v1.0` now).

## Project Purpose

AIET is engineered for exploration, multi-variable comparison, and educational modeling rather than definitive life detection. The system provides an interactive framework to analyze and visualize how complex stellar, orbital, and planetary parameters dynamically influence relative habitability outcomes across thousands of known alien worlds.

## Key Features

- **Physics-Informed Feature Engineering:** Programmatic evaluation of mass-radius bulk densities, stellar insolation fluxes (S_eff), escape velocities (v_esc), and tidal locking synchronization regimes.
- **Machine Learning Architecture:** Multi-variable relative habitability scoring utilizing an optimized XGBoost classification engine.
- **Headless Math Verification:** Command-line validation framework to audit orbital integrator energy conservation bounds (ΔE / E_initial ≤ 10⁻⁶) and dimensional consistency across deep data arrays.
- **Interactive Visualization:** High-performance rendering engine built on Pygame for comparative planetary analysis and parameter space exploration.

## Installation & Deployment

AIET requires **Python 3.11+** (64-bit). Clone the repository and initialize the workspace tracking your target dependencies:

```bash
# Clone the repository
git clone https://github.com/luchprojects/AIET.git
cd AIET

# Initialize virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install core runtime dependencies
pip install -r requirements.txt