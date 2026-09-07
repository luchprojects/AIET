import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
try:
    from src.ml.ml_habitability import MLHabitabilityCalculator
except ImportError:
    # xgboost or the model file may be unavailable; engine runs without ML scoring.
    MLHabitabilityCalculator = None

# Mass conversion: 1 M_sun = 333,000 M_earth (matches main_window.py convention).
_M_EARTH_PER_M_SUN = 333000.0

@dataclass
class CelestialBody:
    name: str
    mass: float  # in Earth masses
    radius: float  # in Earth radii
    position: np.ndarray  # in AU
    velocity: np.ndarray  # in AU/year
    temperature: float  # in Kelvin
    atmosphere: Dict[str, float]  # composition by mass fraction
    type: str  # 'star', 'planet', etc.
    orbper: float = 365.25  # orbital period in days
    orbeccen: float = 0.0  # orbital eccentricity
    
    def __post_init__(self):
        self.acceleration = np.zeros(3)
        self.habitability_score = 0.0
        # For stellar parameters if it's a star
        self.lum = 1.0  # in Solar luminosities
        if self.type == 'star':
            # Stefan-Boltzmann calculation for luminosity if not provided
            # st_lum = (st_rad^2) * (st_teff / 5778.0)^4
            # radius is in Earth radii, convert to solar radii: 1 solar radius = 109.2 Earth radii
            solar_rad = self.radius / 109.2
            self.lum = (solar_rad**2) * (self.temperature / 5778.0)**4

class SimulationEngine:
    def __init__(self):
        self.bodies: List[CelestialBody] = []
        self.G = 39.478  # Gravitational constant in AU^3/(M_sun * year^2)
        self.time_step = 0.01  # in years
        try:
            self.ml_calculator = MLHabitabilityCalculator() if MLHabitabilityCalculator else None
        except Exception as e:
            print(f"Warning: Could not initialize ML calculator: {e}")
            self.ml_calculator = None
        
    def add_body(self, body: CelestialBody):
        self.bodies.append(body)

    @staticmethod
    def physics_mode(placed_bodies) -> str:
        """
        Select the physics integration path based on the placed bodies.

        - 'nbody'     : 2+ active stars are present (e.g. Alpha Centauri).
                        Stars and planets are integrated together with KDK leapfrog
                        so stars actually orbit their common barycenter.
        - 'keplerian' : single-star (or zero-star) systems use analytic Kepler
                        orbits; stars are stationary anchors. This is the default
                        for Solar System / TRAPPIST-1 / sandbox builds.
        """
        if not placed_bodies:
            return "keplerian"
        star_count = 0
        for b in placed_bodies:
            if b is None:
                continue
            if b.get("type") == "star" and not b.get("is_destroyed", False):
                star_count += 1
                if star_count >= 2:
                    return "nbody"
        return "keplerian"

    @staticmethod
    def _mass_solar(body) -> float:
        """
        Return a body's mass in solar masses.

        Stars store mass in M_sun directly; planets and moons store mass in M_earth
        (Earth-mass), so they are converted via 1 M_sun = 333,000 M_earth.
        Destroyed bodies and zero/invalid masses return 0.0 so they contribute
        nothing to gravity / barycenter computations.
        """
        if body is None or body.get("is_destroyed", False):
            return 0.0
        try:
            m = float(body.get("mass", 0.0))
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(m) or m <= 0.0:
            return 0.0
        if body.get("type") == "star":
            return m
        return m / _M_EARTH_PER_M_SUN

    def _initialize_nbody_state(self, placed_bodies, scale_px) -> None:
        """
        Prepare placed_bodies for N-body integration.

        Steps:
        1. Ensure every body has 'position_au' (derived from pixel 'position'
           if missing) and 'velocity_au' (Keplerian-circular-around-parent for
           planets/moons missing one; zero for stars without a preset velocity).
        2. Subtract the system's barycenter position and velocity so the
           integrated state is centered and momentum-balanced (otherwise the
           whole system drifts off-screen during long runs).
        3. Update each body's pixel 'position' to match the post-COM
           position_au * scale_px. Callers (main_window.py) shift the camera
           by +com_shift_px to compensate so the user sees no visual jump.
        """
        if not placed_bodies:
            return

        scale_px_f = float(scale_px) if scale_px else 1.0

        # Pass 1: ensure position_au exists everywhere first (parents need it before
        # children can compute Keplerian velocities relative to them).
        for body in placed_bodies:
            if body is None:
                continue
            pos_au = body.get("position_au")
            if pos_au is None or len(np.asarray(pos_au).reshape(-1)) < 2:
                pos_px = body.get("position")
                if pos_px is not None and len(pos_px) >= 2 and scale_px_f > 0:
                    body["position_au"] = np.array(
                        [float(pos_px[0]), float(pos_px[1])], dtype=float
                    ) / scale_px_f
                else:
                    body["position_au"] = np.zeros(2, dtype=float)
            else:
                body["position_au"] = np.array(
                    [float(pos_au[0]), float(pos_au[1])], dtype=float
                )

        # Pass 2: ensure velocity_au exists. Stars without a preset velocity get
        # zero (will pick up motion from the COM-velocity subtraction below);
        # planets/moons get a circular orbit around their parent star + parent's velocity.
        for body in placed_bodies:
            if body is None:
                continue
            vel_au = body.get("velocity_au")
            if vel_au is not None:
                arr = np.asarray(vel_au).reshape(-1)
                if len(arr) >= 2:
                    body["velocity_au"] = np.array(
                        [float(arr[0]), float(arr[1])], dtype=float
                    )
                    continue

            btype = body.get("type")
            if btype == "star":
                body["velocity_au"] = np.zeros(2, dtype=float)
                continue

            parent = body.get("parent_obj")
            if parent is None or parent.get("position_au") is None:
                body["velocity_au"] = np.zeros(2, dtype=float)
                continue

            r_vec = body["position_au"] - np.array(
                [float(parent["position_au"][0]), float(parent["position_au"][1])],
                dtype=float,
            )
            r = float(np.linalg.norm(r_vec))
            M_p = self._mass_solar(parent)
            if r <= 1e-9 or M_p <= 0.0:
                body["velocity_au"] = np.zeros(2, dtype=float)
                continue

            v_circ = math.sqrt(self.G * M_p / r)
            tangent = np.array([-r_vec[1], r_vec[0]], dtype=float) / r
            v = v_circ * tangent
            parent_v = parent.get("velocity_au")
            if parent_v is not None and len(parent_v) >= 2:
                v = v + np.array(
                    [float(parent_v[0]), float(parent_v[1])], dtype=float
                )
            body["velocity_au"] = v

        # Pass 3: subtract barycenter position and velocity.
        total_mass = 0.0
        com_pos = np.zeros(2, dtype=float)
        com_vel = np.zeros(2, dtype=float)
        for body in placed_bodies:
            if body is None:
                continue
            m = self._mass_solar(body)
            if m <= 0.0:
                continue
            total_mass += m
            com_pos += m * body["position_au"]
            com_vel += m * body["velocity_au"]
        if total_mass > 1e-30:
            com_pos /= total_mass
            com_vel /= total_mass
            for body in placed_bodies:
                if body is None:
                    continue
                if body.get("position_au") is not None:
                    body["position_au"] = body["position_au"] - com_pos
                if body.get("velocity_au") is not None:
                    body["velocity_au"] = body["velocity_au"] - com_vel

        # Pass 4: sync pixel positions to the (now barycentric) AU positions.
        for body in placed_bodies:
            if body is None or body.get("position_au") is None:
                continue
            body["position"] = body["position_au"] * scale_px_f

    def step_nbody(
        self,
        dt_years: float,
        placed_bodies,
        n_sub: int = 20,
        scale_px_per_au: float = 400.0,
    ) -> None:
        """
        Advance the N-body system by dt_years using KDK leapfrog with n_sub sub-steps.

        - Symplectic (energy-bounded) integrator suitable for the Alpha Centauri
          binary plus its planets. Caller is responsible for choosing n_sub so the
          per-substep dt stays small (main_window enforces dt_sub <= 0.001 yr).
        - All bodies (stars, planets, moons) are integrated together; stars actually
          orbit their barycenter and planets feel the moving stars.
        - 2D integration: only the (x, y) components of position_au / velocity_au
          are evolved. The z-axis is unused in the current renderer.
        - Destroyed bodies are excluded from forces and not updated.
        """
        if dt_years <= 0 or n_sub <= 0 or not placed_bodies:
            return

        active = [
            b for b in placed_bodies
            if b is not None and not b.get("is_destroyed", False)
            and b.get("position_au") is not None and b.get("velocity_au") is not None
        ]
        if not active:
            return

        # Stack into numpy arrays for vectorized force evaluation. Mass is in M_sun
        # so G * M has the right units for AU / yr dynamics (G = 39.478).
        pos = np.array(
            [[float(b["position_au"][0]), float(b["position_au"][1])] for b in active],
            dtype=float,
        )  # (n, 2)
        vel = np.array(
            [[float(b["velocity_au"][0]), float(b["velocity_au"][1])] for b in active],
            dtype=float,
        )  # (n, 2)
        mass = np.array([self._mass_solar(b) for b in active], dtype=float)  # (n,)

        n = len(active)
        if n < 2:
            # Single body: just drift in straight line (no self-gravity).
            pos = pos + vel * float(dt_years)
        else:
            # Softening^2 in AU^2: ~1e-3 AU is well below any visible scale yet
            # large enough to keep accelerations finite during near-misses.
            EPS2 = 1e-6
            G = float(self.G)
            dt_sub = float(dt_years) / int(n_sub)
            half = 0.5 * dt_sub

            def compute_accel(p):
                # All-pairs gravity: a_i = sum_{j != i} G * m_j * (x_j - x_i) / |x_j - x_i|^3
                dx = p[np.newaxis, :, :] - p[:, np.newaxis, :]      # (n, n, 2): row i, col j -> x_j - x_i
                r2 = (dx * dx).sum(axis=2) + EPS2                   # (n, n)
                # Diagonal would self-attract; pushing it to inf zeroes the contribution.
                np.fill_diagonal(r2, np.inf)
                inv_r3 = r2 ** -1.5                                 # (n, n)
                weight = mass[np.newaxis, :] * inv_r3               # (n, n)
                a = G * (dx * weight[:, :, np.newaxis]).sum(axis=1) # (n, 2)
                return a

            a = compute_accel(pos)
            for _ in range(int(n_sub)):
                vel = vel + a * half        # half-kick
                pos = pos + vel * dt_sub    # drift
                a = compute_accel(pos)      # recompute accels at new positions
                vel = vel + a * half        # half-kick

        # Write back to body dicts (including the pixel position the renderer reads).
        scale_px = float(scale_px_per_au)
        for i, body in enumerate(active):
            new_pos_au = pos[i].copy()
            new_vel_au = vel[i].copy()
            body["position_au"] = new_pos_au
            body["velocity_au"] = new_vel_au
            body["position"] = new_pos_au * scale_px

    @staticmethod
    def compute_nbody_flux_metrics(placed_bodies, body, period_yr, n_samples=32):
        """
        Return (s_avg, s_max, thermal_inst) for `body` in a multi-star system.

        Approximation: evaluates flux from every active star at the body's CURRENT
        N-body position and sums them. n_samples / period_yr are accepted for API
        compatibility with the UI but not used here — the integrator already drives
        geometry forward, and main_window calls this once per ~60 frames so a
        snapshot is sufficient for the displayed habitability metrics.

        - s_avg / s_max are in S_earth (1 L_sun at 1 AU = 1 S_earth).
        - thermal_inst is the std/mean of per-star contributions; high in
          binaries where a planet is bathed by very different stellar fluxes.
        """
        if body is None or body.get("position_au") is None:
            return (0.0, 0.0, 0.0)

        pos_au = np.asarray(body["position_au"], dtype=float).reshape(-1)
        if len(pos_au) < 2:
            return (0.0, 0.0, 0.0)
        body_xy = pos_au[:2]

        per_star_flux = []
        for s in placed_bodies or []:
            if s is None or s is body:
                continue
            if s.get("type") != "star" or s.get("is_destroyed", False):
                continue
            s_pos = s.get("position_au")
            if s_pos is None:
                continue
            s_arr = np.asarray(s_pos, dtype=float).reshape(-1)
            if len(s_arr) < 2:
                continue
            r2 = float(np.sum((body_xy - s_arr[:2]) ** 2))
            if r2 < 1e-9:
                continue
            try:
                L = float(s.get("luminosity", 1.0))
            except (TypeError, ValueError):
                L = 1.0
            if not math.isfinite(L) or L <= 0.0:
                continue
            per_star_flux.append(L / r2)

        if not per_star_flux:
            return (0.0, 0.0, 0.0)

        arr = np.asarray(per_star_flux, dtype=float)
        s_total = float(arr.sum())
        s_max = float(arr.max())
        mean = float(arr.mean())
        thermal_inst = float(arr.std() / mean) if mean > 1e-12 else 0.0
        return (s_total, s_max, thermal_inst)
        
    def find_host_star(self, body: CelestialBody) -> Optional[CelestialBody]:
        """Find the most massive star that this body might be orbiting"""
        stars = [b for b in self.bodies if b.type == 'star']
        if not stars:
            return None
        # Return the most massive star for now
        return max(stars, key=lambda s: s.mass)

    def calculate_acceleration(self, body: CelestialBody) -> np.ndarray:
        """Calculate gravitational acceleration on a body due to all other bodies"""
        acceleration = np.zeros(3)
        for other in self.bodies:
            if other != body:
                # Convert mass from Earth masses to Solar masses for the calculation
                # if the other body is a star, mass might already be in solar masses?
                # Stars are in Solar masses, planets in Earth masses.
                other_mass_solar = other.mass if other.type == 'star' else other.mass / _M_EARTH_PER_M_SUN
                
                r = other.position - body.position
                r_mag = np.linalg.norm(r)
                if r_mag > 1e-5:  # Avoid division by zero and near-collisions
                    acceleration += self.G * other_mass_solar * r / (r_mag ** 3)
        return acceleration
    
    def update_positions(self):
        """Update positions and velocities using Verlet integration"""
        for body in self.bodies:
            body.acceleration = self.calculate_acceleration(body)
            body.velocity += body.acceleration * self.time_step
            body.position += body.velocity * self.time_step
            
    def calculate_habitability(self, body: CelestialBody) -> float:
        """Calculate habitability score using ML model if available, otherwise use basic physics"""
        if body.type != 'planet':
            return 0.0

        star = self.find_host_star(body)
        if not star:
            return 0.0

        # Calculate distance to star for insolation
        distance = np.linalg.norm(body.position - star.position)
        if distance < 1e-5:
            distance = 1.0 # default to 1 AU
            
        # pl_insol = st_lum / (distance^2)
        pl_insol = star.lum / (distance**2)

        if self.ml_calculator:
            features = {
                "pl_rade": body.radius,
                "pl_masse": body.mass,
                "pl_orbper": body.orbper,
                "pl_orbeccen": body.orbeccen,
                "pl_insol": pl_insol,
                "st_teff": star.temperature,
                "st_mass": star.mass,
                "st_rad": star.radius / 109.2, # Convert Earth radii back to Solar radii for ML
                "st_lum": star.lum
            }
            # The ML model returns percentage (0-100)
            try:
                return self.ml_calculator.predict(features)
            except Exception as e:
                print(f"[ML] Habitability prediction failed for {body.name}: {e}; using basic fallback")
        
        # Fallback to basic calculation if ML is not available
        score = 0.0
        # ... (rest of old logic)
        temp_factor = 1.0 - abs(body.temperature - 288) / 100
        score += max(0, temp_factor) * 0.3
        mass_factor = 1.0 - abs(body.mass - 1.0) / 2.0
        score += max(0, mass_factor) * 0.3
        if 'O2' in body.atmosphere and body.atmosphere['O2'] > 0.1:
            score += 0.2
        if 'H2O' in body.atmosphere and body.atmosphere['H2O'] > 0:
            score += 0.2
        return min(1.0, score) * 100.0 # Convert to percentage to match ML output
    
    def step(self, dt: float = 0.01):
        """Advance the simulation by one time step"""
        self.update_positions()
        for body in self.bodies:
            if body.type == 'planet':
                body.habitability_score = self.calculate_habitability(body) 