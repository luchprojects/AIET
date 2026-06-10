"""
AIET Scientific Diagnostics UI Panel

Provides a professional-grade, collapsible side panel exposing uncertainty,
convergence, validation, and sensitivity outputs in a structured interface.

This module contains all rendering and event handling logic for the
Scientific Diagnostics panel. It integrates with visualization.py via
simple method calls.

Design Principles:
    - No animation, no decorative colors
    - Professional spacing, clean typography
    - Units displayed clearly
    - No emojis
    - Async computation with loading states

Integration:
    In visualization.py __init__:
        from scientific_diagnostics_panel import ScientificDiagnosticsPanel
        self.diagnostics_panel = ScientificDiagnosticsPanel(self)
    
    In render loop:
        self.diagnostics_panel.render()
    
    In event handler:
        if self.diagnostics_panel.handle_event(event):
            continue
"""

from __future__ import annotations

import math
import os
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pygame


# =============================================================================
# PANEL STATE AND CONFIGURATION
# =============================================================================

@dataclass
class DiagnosticsState:
    """State container for all diagnostics data."""
    
    # Uncertainty state
    uncertainty_computed: bool = False
    uncertainty_computing: bool = False
    uncertainty_data: Optional[Dict[str, Any]] = None
    
    # Integrator state
    integrator_computed: bool = False
    integrator_computing: bool = False
    integrator_data: Optional[Any] = None
    integrator_energy_drift_series: Optional[List[float]] = None  # ΔE/E0 time series for sparkline
    
    # Sensitivity state
    sensitivity_computed: bool = False
    sensitivity_computing: bool = False
    sensitivity_data: Optional[Any] = None
    
    # Error messages
    last_error: Optional[str] = None


@dataclass
class IntegratorValidationResult:
    """Minimal integrator diagnostics payload used by the panel UI."""
    integrator_type: str
    max_energy_drift: float
    final_energy_drift: float
    mean_energy_drift: float
    max_angular_momentum_drift: float
    energy_history: List[float] = field(default_factory=list)


# Panel configuration
PANEL_CONFIG = {
    "width": 380,
    "min_height": 400,
    "padding": 18,
    "line_spacing": 22,
    "section_spacing": 28,
    "button_height": 32,
    "button_spacing": 10,
    "border_radius": 8,
    
    # Colors (professional, muted)
    "bg_color": (35, 38, 48),
    "border_color": (100, 105, 120),
    "header_color": (140, 180, 220),
    "label_color": (180, 180, 185),
    "value_color": (240, 240, 245),
    "section_bg": (42, 45, 55),
    
    # Status colors (muted, professional)
    "status_green": (80, 160, 100),
    "status_yellow": (180, 160, 80),
    "status_red": (180, 90, 90),
    "status_gray": (120, 120, 130),
    
    # Button colors
    "button_normal": (60, 65, 80),
    "button_hover": (75, 80, 95),
    "button_text": (220, 220, 225),
    "button_border": (90, 95, 110),
}

# Thresholds for status indicators
THRESHOLDS = {
    "se_good": 0.5,
    "se_marginal": 1.0,
    "convergence_good": 0.3,
    "convergence_marginal": 0.5,
    "energy_good": 1e-6,
    "energy_marginal": 1e-4,
}
# N-Body Numerical Integrity Meter (reads _last_energy_drift from SimulationEngine)
NBODY_DRIFT_GOOD = 1e-5   # Research Grade (green)
NBODY_DRIFT_WARNING = 1e-3  # Numerical noise (yellow); suggest lowering time_scale
# > NBODY_DRIFT_WARNING = Unstable / integrator failure (red)


class ScientificDiagnosticsPanel:
    """
    Scientific Diagnostics UI Panel.
    
    Renders a collapsible side panel with:
    - Uncertainty section (MC results, convergence)
    - Physical Integrity section (energy/momentum conservation)
    - Feature Influence section (sensitivity analysis)
    - Export buttons
    
    Args:
        visualizer: Reference to main AIETVisualizer instance.
    """
    
    def __init__(self, visualizer: Any):
        self.viz = visualizer
        self.screen = visualizer.screen
        self.width = visualizer.width
        self.height = visualizer.height
        
        # Panel state
        self.visible = False
        self.state = DiagnosticsState()
        self.scroll_y = 0
        self.max_scroll = 0
        
        # Button rects (populated during render)
        self.toggle_button_rect = None
        self.panel_rect = None
        self.close_button_rect = None
        self.compute_uncertainty_btn = None
        self.run_integrator_btn = None
        self.compute_sensitivity_btn = None
        self.export_integrity_btn = None
        self.export_sensitivity_btn = None
        self.export_convergence_btn = None
        
        # Thread handles
        self._uncertainty_thread = None
        self._integrator_thread = None
        self._sensitivity_thread = None
        # Worker posts here; main thread applies in render() to avoid layout races (GIL release during MC/XGBoost).
        self._diagnostics_uncertainty_pending: Optional[Dict[str, Any]] = None
        
        # Fonts (will be set from visualizer)
        self._fonts_initialized = False
    
    def _init_fonts(self):
        """Initialize font references from visualizer."""
        if self._fonts_initialized:
            return
        self.title_font = self.viz.subtitle_font
        self.label_font = self.viz.tiny_font
        self.value_font = self.viz.tiny_font
        self.button_font = self.viz.tiny_font
        self._fonts_initialized = True
    
    def toggle(self):
        """Toggle panel visibility."""
        self.visible = not self.visible
        if self.visible:
            # Diagnostics is mutually exclusive with other UI panels/menus
            self.scroll_y = 0
            try:
                self.viz.show_customization_panel = False
                self.viz.show_rename_edit = False
            except Exception:
                pass
            try:
                # Close System (burger) menu and submenus
                self.viz.system_menu_visible = False
                self.viz.system_menu_load_submenu_visible = False
                self.viz.system_menu_export_submenu_visible = False
                self.viz.load_system_list_visible = False
            except Exception:
                pass
            try:
                # Close Add Body and its submenus
                self.viz.add_body_dropdown_visible = False
                self.viz.planets_family_dropdown_visible = False
                self.viz.planets_list_dropdown_visible = False
                self.viz.planets_family_selected = None
                self.viz.stars_family_dropdown_visible = False
                self.viz.stars_list_dropdown_visible = False
                self.viz.stars_family_selected = None
            except Exception:
                pass
    
    def get_toggle_button_rect(self) -> 'pygame.Rect':
        """Get rect for toggle button in main UI — adjacent to Reset View (see main_window diagnostics_toggle_rect)."""
        import pygame
        if getattr(self.viz, "diagnostics_toggle_rect", None) is not None:
            r = self.viz.diagnostics_toggle_rect
            return pygame.Rect(int(r.x), int(r.y), int(r.width), int(r.height))
        btn_width = 80
        btn_height = 28
        if hasattr(self.viz, "reset_dropdown_rect"):
            btn_x = self.viz.reset_dropdown_rect.left - 6 - btn_width
        elif hasattr(self.viz, "export_button"):
            btn_x = self.viz.export_button.left - 6 - btn_width
        else:
            btn_x = max(10, self.viz.width - 400)
        btn_y = 8
        return pygame.Rect(btn_x, btn_y, btn_width, btn_height)
    
    def render_toggle_button(self):
        """Render the toggle button in main UI toolbar with flat styling."""
        import pygame
        
        self._init_fonts()
        
        self.toggle_button_rect = self.get_toggle_button_rect()
        mouse_pos = pygame.mouse.get_pos()
        is_hover = self.toggle_button_rect.collidepoint(mouse_pos)
        
        # Flat button colors (no transparency)
        about_open = bool(getattr(self.viz, "show_about_panel", False))
        if self.visible:
            base_bg = (70, 75, 90)
        elif is_hover:
            base_bg = (65, 70, 85)
        else:
            base_bg = (50, 55, 70)

        def _under_black_overlay(rgb, alpha=180):
            # Emulate being drawn under the About overlay (black with alpha=180).
            f = max(0.0, min(1.0, 1.0 - (alpha / 255.0)))
            return (int(rgb[0] * f), int(rgb[1] * f), int(rgb[2] * f))

        bg_color = _under_black_overlay(base_bg, 180) if about_open else base_bg
        
        # Draw flat button background
        pygame.draw.rect(self.screen, bg_color, self.toggle_button_rect, border_radius=4)
        
        # Subtle border
        border = _under_black_overlay((80, 85, 100), 180) if about_open else (80, 85, 100)
        pygame.draw.rect(self.screen, border, self.toggle_button_rect, 1, border_radius=4)
        
        label = "Diagnostics" if not self.visible else "Hide Diag."
        text_color = _under_black_overlay((220, 220, 225), 180) if about_open else (220, 220, 225)
        text_surf = self.button_font.render(label, True, text_color)
        text_rect = text_surf.get_rect(center=self.toggle_button_rect.center)
        self.screen.blit(text_surf, text_rect)
    
    def render(self):
        """Render the diagnostics panel if visible."""
        import pygame
        
        self._flush_uncertainty_worker_result()
        
        self.render_toggle_button()
        
        if not self.visible:
            return
        
        self._init_fonts()
        
        cfg = PANEL_CONFIG
        panel_width = cfg["width"]
        panel_x = self.viz.width - panel_width - 20
        panel_y = 60
        
        self.compute_uncertainty_btn = None
        self.run_integrator_btn = None
        self.compute_sensitivity_btn = None
        
        content_height = self._measure_total_content_height()
        panel_height = min(self.viz.height - 100, max(cfg["min_height"], content_height + 80))
        
        self.panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        
        shadow = self.panel_rect.copy()
        shadow.move_ip(4, 4)
        pygame.draw.rect(self.screen, (15, 15, 20, 150), shadow, border_radius=cfg["border_radius"])
        
        pygame.draw.rect(self.screen, cfg["bg_color"], self.panel_rect, 
                         border_radius=cfg["border_radius"])
        pygame.draw.rect(self.screen, cfg["border_color"], self.panel_rect, 
                         2, border_radius=cfg["border_radius"])
        
        title = self.title_font.render("Scientific Diagnostics", True, cfg["value_color"])
        title_rect = title.get_rect(midtop=(self.panel_rect.centerx, self.panel_rect.top + 12))
        self.screen.blit(title, title_rect)
        
        close_size = 18
        self.close_button_rect = pygame.Rect(
            self.panel_rect.right - close_size - 10,
            self.panel_rect.top + 10,
            close_size, close_size
        )
        pygame.draw.line(self.screen, cfg["label_color"],
                         self.close_button_rect.topleft, self.close_button_rect.bottomright, 2)
        pygame.draw.line(self.screen, cfg["label_color"],
                         (self.close_button_rect.left, self.close_button_rect.bottom),
                         (self.close_button_rect.right, self.close_button_rect.top), 2)
        
        content_rect = pygame.Rect(
            self.panel_rect.left + 5,
            self.panel_rect.top + 45,
            panel_width - 10,
            panel_height - 55
        )
        
        content_surface = pygame.Surface((panel_width - 10, content_height), pygame.SRCALPHA)
        content_surface.fill((0, 0, 0, 0))
        
        y = 0
        y = self._render_uncertainty_section(content_surface, y)
        y = self._render_integrity_section(content_surface, y)
        y = self._render_sensitivity_section(content_surface, y)
        # Export section intentionally removed.
        # Exports are now centralized in the main Export panel (to avoid duplication).
        
        self.max_scroll = max(0, content_height - content_rect.height)
        self.scroll_y = max(0, min(self.scroll_y, self.max_scroll))
        
        visible_rect = pygame.Rect(0, int(self.scroll_y), content_rect.width, content_rect.height)
        self.screen.blit(content_surface, content_rect, visible_rect)
        
        if self.max_scroll > 0:
            self._render_scrollbar(content_rect)
    
    def _measure_total_content_height(self) -> int:
        """Measure scrollable content height by running section layout (avoids clipping buttons)."""
        import pygame
        
        cfg = PANEL_CONFIG
        w = max(1, cfg["width"] - 10)
        # Height is irrelevant for layout math; draws may clip but returned y is correct.
        dummy = pygame.Surface((w, 8), pygame.SRCALPHA)
        y = 0
        y = self._render_uncertainty_section(dummy, y)
        y = self._render_integrity_section(dummy, y)
        y = self._render_sensitivity_section(dummy, y)
        return max(cfg["min_height"], y + 40)
    
    def _render_section_header(self, surface: 'pygame.Surface', y: int, text: str) -> int:
        """Render a section header."""
        import pygame
        cfg = PANEL_CONFIG
        
        header_rect = pygame.Rect(cfg["padding"] - 5, y, 
                                   surface.get_width() - 2 * cfg["padding"] + 10, 24)
        pygame.draw.rect(surface, cfg["section_bg"], header_rect, border_radius=4)
        
        header = self.title_font.render(text, True, cfg["header_color"])
        surface.blit(header, (cfg["padding"], y + 3))
        
        return y + 28
    
    def _render_stat_row(self, surface: 'pygame.Surface', y: int, 
                         label: str, value: str, unit: str = "",
                         status_color: Optional[Tuple[int, int, int]] = None) -> int:
        """Render a label: value row."""
        cfg = PANEL_CONFIG
        
        label_surf = self.label_font.render(label, True, cfg["label_color"])
        surface.blit(label_surf, (cfg["padding"] + 8, y))
        
        if unit:
            value_text = f"{value} {unit}"
        else:
            value_text = str(value)
        
        value_surf = self.value_font.render(value_text, True, cfg["value_color"])
        surface.blit(value_surf, (surface.get_width() - cfg["padding"] - value_surf.get_width(), y))
        
        if status_color:
            import pygame
            indicator_rect = pygame.Rect(cfg["padding"], y + 4, 4, 10)
            pygame.draw.rect(surface, status_color, indicator_rect)
        
        return y + cfg["line_spacing"]
    
    @staticmethod
    def _safe_mc_float(value: Any, default: float = 0.0) -> float:
        try:
            x = float(value)
            return x if math.isfinite(x) else default
        except (TypeError, ValueError):
            return default
    
    def _flush_uncertainty_worker_result(self) -> None:
        """Apply Monte Carlo result on the main thread (avoids mid-frame layout/surface size races)."""
        pending = self._diagnostics_uncertainty_pending
        if pending is None:
            return
        self._diagnostics_uncertainty_pending = None
        self.state.uncertainty_computing = False
        err = pending.get("error")
        res = pending.get("result")
        if err:
            self.state.last_error = str(err)
        elif res is not None:
            self.state.uncertainty_data = res
            self.state.uncertainty_computed = True
    
    def _render_button(self, surface: 'pygame.Surface', y: int, text: str,
                       computing: bool = False) -> Tuple[int, 'pygame.Rect']:
        """Render a button and return its rect (relative to surface)."""
        import pygame
        cfg = PANEL_CONFIG
        
        btn_width = surface.get_width() - 2 * cfg["padding"]
        btn_rect = pygame.Rect(cfg["padding"], y, btn_width, cfg["button_height"])
        
        if computing:
            color = cfg["status_gray"]
            label = "Computing..."
        else:
            color = cfg["button_normal"]
            label = text
        
        pygame.draw.rect(surface, color, btn_rect, border_radius=4)
        pygame.draw.rect(surface, cfg["button_border"], btn_rect, 1, border_radius=4)
        
        text_surf = self.button_font.render(label, True, cfg["button_text"])
        text_rect = text_surf.get_rect(center=btn_rect.center)
        surface.blit(text_surf, text_rect)
        
        return y + cfg["button_height"] + cfg["button_spacing"], btn_rect
    
    def _render_uncertainty_section(self, surface: 'pygame.Surface', y: int) -> int:
        """Render the Habitability Uncertainty section."""
        cfg = PANEL_CONFIG
        y = self._render_section_header(surface, y, "Habitability Uncertainty")
        y += 5
        
        if self.state.uncertainty_computing:
            y = self._render_stat_row(surface, y, "Status:", "Computing...")
            y += cfg["section_spacing"]
            return y
        
        if not self.state.uncertainty_computed or self.state.uncertainty_data is None:
            y = self._render_stat_row(surface, y, "Status:", self._pending_status_text("Not computed"))
            y += 5
            y, self.compute_uncertainty_btn = self._render_button(
                surface, y, "Compute Uncertainty", self.state.uncertainty_computing
            )
            y += cfg["section_spacing"]
            return y
        
        data = self.state.uncertainty_data
        
        mean = self._safe_mc_float(data.get("mean_index", 0))
        std = self._safe_mc_float(data.get("std_dev", 0))
        y = self._render_stat_row(surface, y, "Mean Index:", f"{mean:.2f} +/- {std:.2f}")
        
        ci_lower = self._safe_mc_float(data.get("ci_lower", 0))
        ci_upper = self._safe_mc_float(data.get("ci_upper", 0))
        y = self._render_stat_row(surface, y, "95% CI:", f"[{ci_lower:.2f}, {ci_upper:.2f}]")
        
        se = self._safe_mc_float(data.get("standard_error", 0))
        se_status = self._get_status_color(se, THRESHOLDS["se_good"], THRESHOLDS["se_marginal"])
        y = self._render_stat_row(surface, y, "Standard Error:", f"{se:.4f}", status_color=se_status)
        
        delta = self._safe_mc_float(data.get("convergence_delta", 0))
        delta_status = self._get_status_color(delta, THRESHOLDS["convergence_good"], 
                                               THRESHOLDS["convergence_marginal"])
        y = self._render_stat_row(surface, y, "Convergence Delta:", f"{delta:.4f}", 
                                   status_color=delta_status)
        
        samples = data.get("sample_count", data.get("samples", 0))
        early = " (early)" if data.get("converged_early", False) else ""
        y = self._render_stat_row(surface, y, "Samples:", f"{samples}{early}")
        
        y += cfg["section_spacing"]
        return y
    
    def _render_integrity_section(self, surface: 'pygame.Surface', y: int) -> int:
        """Render the Physical Integrity section."""
        cfg = PANEL_CONFIG
        y = self._render_section_header(surface, y, "Physical Integrity")
        y += 5
        
        # N-Body mode: Numerical Integrity Meter (reads _last_energy_drift from SimulationEngine)
        physics_mode = getattr(self.viz, "physics_mode", "keplerian")
        if physics_mode == "nbody":
            engine = getattr(self.viz, "_simulation_engine", None)
            drift = getattr(engine, "_last_energy_drift", None) if engine else None
            if drift is not None:
                if drift < NBODY_DRIFT_GOOD:
                    stability_text = "Good (Research Grade)"
                    ns_status = cfg["status_green"]
                elif drift < NBODY_DRIFT_WARNING:
                    stability_text = "Warning (noise; lower time scale)"
                    ns_status = cfg["status_yellow"]
                else:
                    stability_text = "Unstable (non-physical)"
                    ns_status = cfg["status_red"]
                y = self._render_stat_row(surface, y, "Numerical Integrity:", stability_text, status_color=ns_status)
                y = self._render_stat_row(surface, y, "  |dE/E0|:", f"{drift:.2e}")
            else:
                y = self._render_stat_row(surface, y, "Numerical Integrity:", "N-Body (Leapfrog)")
            # N-Body: Time-averaged flux and peak flux (replace static Stellar Flux when in N-body)
            sel = getattr(self.viz, "selected_body", None)
            if sel and sel.get("type") == "planet":
                s_avg = sel.get("stellarFlux")
                s_max = sel.get("_nbody_s_max_flux")
                if s_avg is not None:
                    y = self._render_stat_row(surface, y, "S_avg (EFU):", f"{s_avg:.4f}")
                if s_max is not None:
                    y = self._render_stat_row(surface, y, "S_max (EFU):", f"{s_max:.4f}")
            else:
                # Show first planet's flux if no planet selected
                placed = getattr(self.viz, "placed_bodies", [])
                for b in placed:
                    if b.get("type") == "planet" and not b.get("is_destroyed", False):
                        s_avg = b.get("stellarFlux")
                        s_max = b.get("_nbody_s_max_flux")
                        if s_avg is not None:
                            y = self._render_stat_row(surface, y, "S_avg (EFU):", f"{s_avg:.4f}")
                        if s_max is not None:
                            y = self._render_stat_row(surface, y, "S_max (EFU):", f"{s_max:.4f}")
                        break
            y += 3
        
        if self.state.integrator_computing:
            y = self._render_stat_row(surface, y, "Status:", "Running validation...")
            y += cfg["section_spacing"]
            return y
        
        if not self.state.integrator_computed or self.state.integrator_data is None:
            y = self._render_stat_row(surface, y, "Status:", self._pending_status_text("Not validated"))
            y += 5
            y, self.run_integrator_btn = self._render_button(
                surface, y, "Run Integrator Validation", self.state.integrator_computing
            )
            y += cfg["section_spacing"]
            return y
        
        data = self.state.integrator_data
        
        int_type = getattr(data, "integrator_type", "verlet") if hasattr(data, "integrator_type") else "verlet"
        y = self._render_stat_row(surface, y, "Integrator:", int_type.upper())
        
        max_e = getattr(data, "max_energy_drift", 0)
        e_status = self._get_status_color(max_e, THRESHOLDS["energy_good"], 
                                           THRESHOLDS["energy_marginal"])
        y = self._render_stat_row(surface, y, "Max |dE/E0|:", f"{max_e:.2e}", status_color=e_status)
        
        max_l = getattr(data, "max_angular_momentum_drift", 0)
        l_status = self._get_status_color(max_l, THRESHOLDS["energy_good"], 
                                           THRESHOLDS["energy_marginal"])
        y = self._render_stat_row(surface, y, "Max |dL/L0|:", f"{max_l:.2e}", status_color=l_status)
        
        conserved = max_e < THRESHOLDS["energy_marginal"] and max_l < THRESHOLDS["energy_marginal"]
        status_text = "Conserved" if conserved else "Drift detected"
        status_color = cfg["status_green"] if conserved else cfg["status_red"]
        y = self._render_stat_row(surface, y, "Status:", status_text, status_color=status_color)

        # Sparkline: energy drift ΔE/E0 (flat line => stable)
        if getattr(self.state, "integrator_energy_drift_series", None):
            y += 6
            y = self._render_energy_drift_sparkline(surface, y, self.state.integrator_energy_drift_series)
        
        y += cfg["section_spacing"]
        return y
    
    def _render_sensitivity_section(self, surface: 'pygame.Surface', y: int) -> int:
        """Render the Feature Influence section."""
        cfg = PANEL_CONFIG
        y = self._render_section_header(surface, y, "Feature Influence")
        y += 5
        
        if self.state.sensitivity_computing:
            y = self._render_stat_row(surface, y, "Status:", "Analyzing...")
            y += cfg["section_spacing"]
            return y
        
        if not self.state.sensitivity_computed or self.state.sensitivity_data is None:
            y = self._render_stat_row(surface, y, "Status:", self._pending_status_text("Not computed"))
            y += 5
            y, self.compute_sensitivity_btn = self._render_button(
                surface, y, "Compute Sensitivity", self.state.sensitivity_computing
            )
            y += cfg["section_spacing"]
            return y
        
        data = self.state.sensitivity_data
        
        ranked = getattr(data, "local_oat_ranked", {}) if hasattr(data, "local_oat_ranked") else {}
        if not ranked:
            ranked = data.get("local_oat_ranked", {}) if isinstance(data, dict) else {}
        
        y = self._render_stat_row(surface, y, "Top Influential Features:", "")
        
        try:
            from src.science.sensitivity_analysis import FEATURE_DESCRIPTIONS
        except ImportError:
            FEATURE_DESCRIPTIONS = {}
        
        for i, (feat, sens) in enumerate(list(ranked.items())[:3]):
            feat_label = FEATURE_DESCRIPTIONS.get(feat, feat)
            sign = "+" if sens >= 0 else ""
            y = self._render_stat_row(surface, y, f"  {i+1}. {feat_label}:", f"{sign}{sens:.3f}")
        
        y += cfg["section_spacing"]
        return y
    
    def _render_export_section(self, surface: 'pygame.Surface', y: int) -> int:
        """Render the export buttons section."""
        cfg = PANEL_CONFIG
        y = self._render_section_header(surface, y, "Export")
        y += 5
        
        y, self.export_integrity_btn = self._render_button(surface, y, "Export Integrity Report")
        y, self.export_sensitivity_btn = self._render_button(surface, y, "Export Sensitivity Report")
        y, self.export_convergence_btn = self._render_button(surface, y, "Export Convergence Plot")
        
        return y
    
    def _render_scrollbar(self, content_rect: 'pygame.Rect'):
        """Render scrollbar if needed."""
        import pygame
        
        scrollbar_width = 6
        scrollbar_x = content_rect.right - scrollbar_width - 2
        track_height = content_rect.height
        
        track_rect = pygame.Rect(scrollbar_x, content_rect.top, scrollbar_width, track_height)
        pygame.draw.rect(self.screen, (50, 50, 60), track_rect, border_radius=3)
        
        thumb_height = max(20, int(track_height * track_height / (self.max_scroll + track_height)))
        thumb_y = content_rect.top + int((track_height - thumb_height) * self.scroll_y / self.max_scroll)
        thumb_rect = pygame.Rect(scrollbar_x, thumb_y, scrollbar_width, thumb_height)
        pygame.draw.rect(self.screen, (100, 105, 115), thumb_rect, border_radius=3)
    
    def _get_status_color(self, value: float, good_threshold: float, 
                          marginal_threshold: float) -> Tuple[int, int, int]:
        """Get status indicator color based on thresholds."""
        cfg = PANEL_CONFIG
        if value < good_threshold:
            return cfg["status_green"]
        elif value < marginal_threshold:
            return cfg["status_yellow"]
        else:
            return cfg["status_red"]
    
    def handle_event(self, event: 'pygame.event.Event') -> bool:
        """
        Handle pygame events for the diagnostics panel.
        
        Returns True if event was consumed.
        """
        import pygame
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # When About overlay is open, Diagnostics toggle should be inert (can't be clicked).
            about_open = bool(getattr(self.viz, "show_about_panel", False))
            if self.toggle_button_rect and self.toggle_button_rect.collidepoint(event.pos):
                if about_open:
                    return False
                self.toggle()
                return True
            
            if self.visible:
                if self.close_button_rect and self.close_button_rect.collidepoint(event.pos):
                    self.visible = False
                    return True
                
                if self.panel_rect and not self.panel_rect.collidepoint(event.pos):
                    # Click outside closes the diagnostics panel.
                    # If the click is on a top-bar control, allow the event to propagate
                    # so the user can "switch" directly to that control in one click.
                    self.visible = False
                    try:
                        p = event.pos
                        top_bar_rects = [
                            getattr(self.viz, "add_body_dropdown_rect", None),
                            getattr(self.viz, "system_menu_rect", None),
                            getattr(self.viz, "about_button", None),
                            getattr(self.viz, "reset_dropdown_rect", None),
                            getattr(self.viz, "export_button", None),
                            getattr(self.viz, "placement_cancel_button_rect", None) if getattr(self.viz, "placement_mode_active", False) else None,
                        ]
                        if any(r is not None and r.collidepoint(p) for r in top_bar_rects):
                            return False
                    except Exception:
                        pass
                    return True
                
                content_rect = pygame.Rect(
                    self.panel_rect.left + 5,
                    self.panel_rect.top + 45,
                    self.panel_rect.width - 10,
                    self.panel_rect.height - 55,
                )
                if not content_rect.collidepoint(event.pos):
                    return True
                rel_x = event.pos[0] - content_rect.left
                rel_y = event.pos[1] - content_rect.top + self.scroll_y
                
                if self._check_button_click(self.compute_uncertainty_btn, rel_x, rel_y):
                    self._start_uncertainty_computation()
                    return True
                
                if self._check_button_click(self.run_integrator_btn, rel_x, rel_y):
                    self._start_integrator_validation()
                    return True
                
                if self._check_button_click(self.compute_sensitivity_btn, rel_x, rel_y):
                    self._start_sensitivity_computation()
                    return True
        
        if event.type == pygame.MOUSEWHEEL and self.visible:
            if self.panel_rect and self.panel_rect.collidepoint(pygame.mouse.get_pos()):
                self.scroll_y -= event.y * 30
                self.scroll_y = max(0, min(self.scroll_y, self.max_scroll))
                return True
        
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and self.visible:
            self.visible = False
            return True
        
        return False
    
    def _check_button_click(
        self, btn_rect: Optional['pygame.Rect'], rel_x: int, rel_y: int
    ) -> bool:
        """Check if a button was clicked in content coordinates (respects scroll and X)."""
        if btn_rect is None:
            return False
        return btn_rect.collidepoint(rel_x, rel_y)
    
    def _get_selected_planet_data(self) -> Optional[Dict[str, Any]]:
        """Get NASA-style data for selected planet."""
        body = self.viz.selected_body
        if not body or body.get("type") not in ("planet", "moon"):
            return None
        
        data = {
            "pl_rade": body.get("radius", 1.0),
            "pl_masse": body.get("mass", 1.0),
            "pl_orbper": body.get("orbitalPeriod", 365.25),
            "pl_orbsmax": body.get("semiMajorAxis", 1.0),
            "pl_orbeccen": body.get("eccentricity", 0.0),
            "pl_insol": body.get("insolation", 1.0),
            "pl_eqt": body.get("equilibrium_temperature", 255.0),
            "pl_dens": body.get("density", 5.51),
        }
        
        parent = body.get("parent_obj")
        if parent and parent.get("type") == "star":
            data["st_teff"] = parent.get("temperature", 5778.0)
            data["st_mass"] = parent.get("mass", 1.0)
            data["st_rad"] = parent.get("radius", 1.0) / 109.2
            data["st_lum"] = parent.get("luminosity", 1.0)
        else:
            data["st_teff"] = 5778.0
            data["st_mass"] = 1.0
            data["st_rad"] = 1.0
            data["st_lum"] = 1.0
        
        return data
    
    def _get_simulation_bodies(self) -> List[Dict[str, Any]]:
        """Get list of bodies for integrator validation."""
        bodies = []
        for body in self.viz.placed_bodies:
            if "position" in body and "velocity" in body:
                bodies.append({
                    "name": body.get("name", "unknown"),
                    "type": body.get("type", "planet"),
                    "mass": body.get("mass", 1.0),
                    "position": np.array(body["position"], dtype=np.float64),
                    "velocity": np.array(body["velocity"], dtype=np.float64),
                })
        return bodies
    
    def _start_uncertainty_computation(self):
        """Start async uncertainty computation."""
        if self.state.uncertainty_computing:
            return
        
        planet_data = self._get_selected_planet_data()
        if not planet_data:
            self.state.last_error = "No planet selected"
            toast = getattr(self.viz, "_show_export_toast", None)
            if callable(toast):
                toast("Diagnostics: select a planet or moon first")
            return
        
        self.state.uncertainty_computing = True
        
        def compute():
            err: Optional[str] = None
            result: Any = None
            try:
                calculator = getattr(self.viz, "ml_calculator", None)
                if calculator is None:
                    from src.ml.ml_habitability import MLHabitabilityCalculator
                    calculator = MLHabitabilityCalculator()
                    self.viz.ml_calculator = calculator

                result = calculator.predict_with_uncertainty(
                    planet_data=planet_data,
                    N=1000,
                    seed=42,
                )
            except Exception as e:
                err = str(e)
                print(f"[Diagnostics] Uncertainty computation failed: {e}")
            self._diagnostics_uncertainty_pending = {"result": result, "error": err}
        
        self._uncertainty_thread = threading.Thread(target=compute, daemon=True)
        self._uncertainty_thread.start()
    
    def _start_integrator_validation(self):
        """Start async integrator validation."""
        if self.state.integrator_computing:
            return
        
        bodies = self._get_simulation_bodies()
        if len(bodies) < 2:
            self.state.last_error = "Need at least 2 bodies for validation"
            return
        
        self.state.integrator_computing = True
        
        def compute():
            try:
                result = None
                try:
                    from src.physics.integrator_diagnostics import run_integrator_validation
                    result = run_integrator_validation(
                        bodies=bodies,
                        duration=1.0,
                        dt=0.001,
                        integrator="verlet",
                        record_interval=50,
                    )
                except Exception:
                    # Fallback path for builds where integrator_diagnostics module is absent.
                    result = self._run_integrator_validation_fallback(
                        bodies=bodies,
                        duration=1.0,
                        dt=0.001,
                        integrator="verlet",
                        record_interval=50,
                    )
                
                self.state.integrator_data = result
                self.state.integrator_computed = True
                self.state.integrator_energy_drift_series = self._extract_energy_drift_series(result)
                
            except Exception as e:
                self.state.last_error = str(e)
                print(f"[Diagnostics] Integrator validation failed: {e}")
            finally:
                self.state.integrator_computing = False
        
        self._integrator_thread = threading.Thread(target=compute, daemon=True)
        self._integrator_thread.start()

    def _run_integrator_validation_fallback(
        self,
        bodies: List[Dict[str, Any]],
        duration: float = 1.0,
        dt: float = 0.001,
        integrator: str = "verlet",
        record_interval: int = 50,
    ) -> IntegratorValidationResult:
        """Internal lightweight n-body validator used when diagnostics module is unavailable."""
        G = 39.478  # AU^3 / (Msun * yr^2)

        def _to3(v: np.ndarray) -> np.ndarray:
            arr = np.asarray(v, dtype=np.float64).reshape(-1)
            if arr.size >= 3:
                return arr[:3].copy()
            out = np.zeros(3, dtype=np.float64)
            out[:arr.size] = arr
            return out

        masses = np.array([max(float(b.get("mass", 0.0)), 1e-15) for b in bodies], dtype=np.float64)
        pos = np.array([_to3(np.asarray(b["position"], dtype=np.float64)) for b in bodies], dtype=np.float64)
        vel = np.array([_to3(np.asarray(b["velocity"], dtype=np.float64)) for b in bodies], dtype=np.float64)

        def accelerations(p: np.ndarray) -> np.ndarray:
            n = len(p)
            acc = np.zeros_like(p)
            for i in range(n):
                for j in range(i + 1, n):
                    r = p[j] - p[i]
                    d2 = float(np.dot(r, r)) + 1e-16
                    inv_d3 = 1.0 / (d2 * np.sqrt(d2))
                    f = G * r * inv_d3
                    acc[i] += masses[j] * f
                    acc[j] -= masses[i] * f
            return acc

        def total_energy(p: np.ndarray, v: np.ndarray) -> float:
            kinetic = 0.5 * np.sum(masses[:, None] * (v * v))
            potential = 0.0
            n = len(p)
            for i in range(n):
                for j in range(i + 1, n):
                    r = float(np.linalg.norm(p[j] - p[i])) + 1e-16
                    potential -= G * masses[i] * masses[j] / r
            return float(kinetic + potential)

        def total_ang_momentum_mag(p: np.ndarray, v: np.ndarray) -> float:
            L = np.zeros(3, dtype=np.float64)
            for i in range(len(p)):
                L += masses[i] * np.cross(p[i], v[i])
            return float(np.linalg.norm(L))

        steps = max(1, int(duration / dt))
        interval = max(1, int(record_interval))

        a = accelerations(pos)
        v_half = vel + 0.5 * dt * a

        e_hist: List[float] = []
        l_hist: List[float] = []

        e0 = total_energy(pos, vel)
        l0 = total_ang_momentum_mag(pos, vel)
        e_hist.append(e0)
        l_hist.append(l0)

        for step in range(steps):
            pos = pos + dt * v_half
            a_new = accelerations(pos)
            v_half = v_half + dt * a_new
            vel_curr = v_half - 0.5 * dt * a_new

            if (step + 1) % interval == 0 or step == steps - 1:
                e_hist.append(total_energy(pos, vel_curr))
                l_hist.append(total_ang_momentum_mag(pos, vel_curr))

        e_ref = e_hist[0]
        if abs(e_ref) > 1e-15:
            e_drifts = [abs((e - e_ref) / e_ref) for e in e_hist]
        else:
            e_drifts = [abs(e - e_ref) for e in e_hist]

        l_ref = l_hist[0]
        if abs(l_ref) > 1e-15:
            l_drifts = [abs((l - l_ref) / l_ref) for l in l_hist]
        else:
            l_drifts = [abs(l - l_ref) for l in l_hist]

        return IntegratorValidationResult(
            integrator_type=integrator,
            max_energy_drift=float(max(e_drifts) if e_drifts else 0.0),
            final_energy_drift=float(e_drifts[-1] if e_drifts else 0.0),
            mean_energy_drift=float(sum(e_drifts) / len(e_drifts) if e_drifts else 0.0),
            max_angular_momentum_drift=float(max(l_drifts) if l_drifts else 0.0),
            energy_history=[float(v) for v in e_hist],
        )

    def _pending_status_text(self, fallback: str) -> str:
        """
        If the ML deferred scoring queue is processing, show a "Pending Data" pulse.
        Otherwise return the provided fallback string.
        """
        if self._ml_deferred_queue_busy():
            # Simple text pulse without heavy animation
            phase = int(time.time() * 2.0) % 4  # ~2 Hz, 4 states
            dots = "." * phase
            return f"Pending Data{dots}"
        return fallback

    def _ml_deferred_queue_busy(self) -> bool:
        """
        Returns True if ML scoring queue is processing for the selected planet (or globally busy).
        """
        try:
            q = getattr(self.viz, "ml_scoring_queue", None)
            if not q:
                return False
            # Prefer stable ID reference (selected_body_id) when available.
            sel_id = getattr(self.viz, "selected_body_id", None)
            if sel_id in q:
                return True
            sel = getattr(self.viz, "selected_body", None)
            if sel and isinstance(sel, dict) and sel.get("id") in q:
                return True

            # If ANY ML work is queued, treat diagnostics as pending (queue is short-lived).
            return len(q) > 0
        except Exception:
            return False

    def _extract_energy_drift_series(self, metrics: Any) -> Optional[List[float]]:
        """
        Build ΔE/E0 series from DriftMetrics if available.
        """
        try:
            energies = getattr(metrics, "energy_history", None)
            if not energies or len(energies) < 2:
                return None
            E0 = float(energies[0])
            if abs(E0) > 1e-15:
                drift = [(float(E) - E0) / E0 for E in energies]
            else:
                drift = [float(E) - E0 for E in energies]
            return drift
        except Exception:
            return None

    def _render_energy_drift_sparkline(self, surface: 'pygame.Surface', y: int, series: List[float]) -> int:
        """Render a compact sparkline of ΔE/E0."""
        import pygame
        cfg = PANEL_CONFIG

        w = surface.get_width() - 2 * cfg["padding"]
        h = 34
        rect = pygame.Rect(cfg["padding"], y, w, h)
        pygame.draw.rect(surface, (30, 33, 42), rect, border_radius=4)
        pygame.draw.rect(surface, (80, 90, 110), rect, 1, border_radius=4)

        if not series or len(series) < 2:
            return y + h + 6

        # Downsample to fit width
        n = len(series)
        max_pts = max(8, min(w, 120))
        if n > max_pts:
            idx = np.linspace(0, n - 1, max_pts).astype(int)
            ys = [float(series[i]) for i in idx]
        else:
            ys = [float(v) for v in series]

        max_abs = max(1e-18, max(abs(v) for v in ys))
        # Keep tiny drifts visible
        max_abs = max_abs if max_abs > 0 else 1.0

        x0 = rect.left + 6
        x1 = rect.right - 6
        y_mid = rect.centery
        y_amp = (h // 2) - 6

        pts = []
        for i, v in enumerate(ys):
            t = i / (len(ys) - 1)
            x = x0 + t * (x1 - x0)
            # invert y (positive drift up)
            yv = y_mid - (v / max_abs) * y_amp
            pts.append((int(x), int(yv)))

        # zero line
        pygame.draw.line(surface, (70, 75, 90), (x0, y_mid), (x1, y_mid), 1)
        pygame.draw.lines(surface, (150, 190, 255), False, pts, 2)

        label = self.label_font.render("ΔE/E0", True, (190, 195, 210))
        surface.blit(label, (rect.left + 8, rect.top + 6))

        return y + h + 6
    
    def _start_sensitivity_computation(self):
        """Start async sensitivity computation."""
        if self.state.sensitivity_computing:
            return
        
        planet_data = self._get_selected_planet_data()
        if not planet_data:
            self.state.last_error = "No planet selected"
            toast = getattr(self.viz, "_show_export_toast", None)
            if callable(toast):
                toast("Diagnostics: select a planet or moon first")
            return
        
        self.state.sensitivity_computing = True
        
        def compute():
            try:
                calculator = getattr(self.viz, "ml_calculator", None)
                if calculator is None:
                    from src.ml.ml_habitability import MLHabitabilityCalculator
                    calculator = MLHabitabilityCalculator()
                    self.viz.ml_calculator = calculator

                from src.science.sensitivity_analysis import compute_full_sensitivity_report
                
                planet_name = ""
                if self.viz.selected_body:
                    planet_name = self.viz.selected_body.get("name", "")
                
                result = compute_full_sensitivity_report(
                    calculator=calculator,
                    planet_data=planet_data,
                    planet_name=planet_name,
                    run_shap=False,
                    seed=42,
                )
                
                self.state.sensitivity_data = result
                self.state.sensitivity_computed = True
                
            except Exception as e:
                self.state.last_error = str(e)
                print(f"[Diagnostics] Sensitivity computation failed: {e}")
            finally:
                self.state.sensitivity_computing = False
        
        self._sensitivity_thread = threading.Thread(target=compute, daemon=True)
        self._sensitivity_thread.start()
    
    def _export_integrity_report(self, output_dir: str = "exports") -> Optional[str]:
        """Export integrity report to JSON. Returns output path on success."""
        try:
            planet_data = self._get_selected_planet_data()
            if not planet_data:
                print("[Diagnostics] No planet selected for export")
                return None

            planet_name = self.viz.selected_body.get("name", "planet") if self.viz.selected_body else "planet"
            safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in planet_name) or "planet"
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"integrity_{safe_name}_{timestamp}.json")

            report = None
            try:
                from src.science.model_integrity import generate_model_integrity_report, export_integrity_report_json
                calculator = getattr(self.viz, "ml_calculator", None)
                if calculator is None:
                    from src.ml.ml_habitability import MLHabitabilityCalculator
                    calculator = MLHabitabilityCalculator()
                bodies = self._get_simulation_bodies()
                report = generate_model_integrity_report(
                    calculator=calculator,
                    planet_data=planet_data,
                    bodies=bodies if len(bodies) >= 2 else None,
                    planet_name=planet_name,
                    mc_samples=500,
                    run_integrator=len(bodies) >= 2,
                )
                export_integrity_report_json(report, output_path)
            except Exception:
                # Fallback report format for builds without src.science.model_integrity.
                metrics = self.state.integrator_data
                report = {
                    "planet_name": planet_name,
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "planet_data": planet_data,
                    "integrator": {
                        "computed": bool(metrics is not None),
                        "integrator_type": getattr(metrics, "integrator_type", "verlet") if metrics is not None else None,
                        "max_energy_drift": float(getattr(metrics, "max_energy_drift", 0.0)) if metrics is not None else None,
                        "final_energy_drift": float(getattr(metrics, "final_energy_drift", 0.0)) if metrics is not None else None,
                        "mean_energy_drift": float(getattr(metrics, "mean_energy_drift", 0.0)) if metrics is not None else None,
                        "max_angular_momentum_drift": float(getattr(metrics, "max_angular_momentum_drift", 0.0)) if metrics is not None else None,
                    },
                    "uncertainty_available": bool(self.state.uncertainty_computed and self.state.uncertainty_data is not None),
                    "sensitivity_available": bool(self.state.sensitivity_computed and self.state.sensitivity_data is not None),
                }
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
            return output_path
        except Exception as e:
            print(f"[Diagnostics] Export failed: {e}")
            return None
    
    def _export_sensitivity_report(self, output_dir: str = "exports") -> Optional[str]:
        """Export sensitivity report to JSON. Returns output path on success."""
        if not self.state.sensitivity_computed or self.state.sensitivity_data is None:
            print("[Diagnostics] No sensitivity data to export. Compute first.")
            return None
        
        try:
            from src.science.sensitivity_analysis import export_sensitivity_report_json
            
            os.makedirs(output_dir, exist_ok=True)
            planet_name = getattr(self.state.sensitivity_data, "planet_name", "planet") or "planet"
            safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in planet_name) or "planet"
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"sensitivity_{safe_name}_{timestamp}.json")
            
            export_sensitivity_report_json(self.state.sensitivity_data, output_path)
            return output_path
        except Exception as e:
            print(f"[Diagnostics] Sensitivity export failed: {e}")
            return None
    
    def _export_convergence_plot(self, output_dir: str = "exports") -> Optional[str]:
        """Export convergence plot to PNG. Returns output path on success."""
        if not self.state.uncertainty_computed or self.state.uncertainty_data is None:
            print("[Diagnostics] No uncertainty data to plot. Compute first.")
            return None
        
        try:
            from src.ml.ml_uncertainty import export_uncertainty_convergence_plot
            
            os.makedirs(output_dir, exist_ok=True)
            planet_name = ""
            if self.viz.selected_body:
                planet_name = self.viz.selected_body.get("name", "planet")
            safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in planet_name) or "planet"
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"convergence_{safe_name}_{timestamp}.png")
            
            export_uncertainty_convergence_plot(
                self.state.uncertainty_data,
                output_path,
                planet_name=planet_name,
            )
            return output_path
        except Exception as e:
            print(f"[Diagnostics] Convergence plot export failed: {e}")
            return None
    
    def reset_state(self):
        """Reset all computed state (e.g., when planet selection changes)."""
        self.state = DiagnosticsState()
        self.scroll_y = 0
