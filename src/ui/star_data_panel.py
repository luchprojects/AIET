"""
Star Data panel: spectroscopy, H-R diagram, evolution and derived properties
for the currently selected star.

Presentation-only. Reads the selected star's parameter dict every frame so it
stays live while the user edits mass / temperature / radius / luminosity in the
customization panel. All numbers come from src/science/stellar_data.py.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pygame

from src.science import stellar_data as sd
from src.ui import theme as ui_theme
from src.ui.theme import THEME


TABS: Sequence[Tuple[str, str]] = (
    ("spectrum", "Spectrum"),
    ("hr", "H–R Diagram"),
    ("evolution", "Evolution"),
    ("hz", "HZ over time"),
    ("properties", "Properties"),
)

Color = Tuple[int, int, int]


def _mix(a: Color, b: Color, t: float) -> Color:
    t = max(0.0, min(1.0, t))
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t), int(a[2] + (b[2] - a[2]) * t))


def _wavelength_to_rgb(lam_nm: float) -> Color:
    """Approximate sRGB of a monochromatic wavelength (380–750 nm), for the spectrum strip."""
    w = lam_nm
    if w < 380 or w > 750:
        return (0, 0, 0)
    if w < 440:
        r, g, b = -(w - 440) / 60, 0.0, 1.0
    elif w < 490:
        r, g, b = 0.0, (w - 440) / 50, 1.0
    elif w < 510:
        r, g, b = 0.0, 1.0, -(w - 510) / 20
    elif w < 580:
        r, g, b = (w - 510) / 70, 1.0, 0.0
    elif w < 645:
        r, g, b = 1.0, -(w - 645) / 65, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    # Intensity falloff at the edges of vision
    if w < 420:
        f = 0.3 + 0.7 * (w - 380) / 40
    elif w > 700:
        f = 0.3 + 0.7 * (750 - w) / 50
    else:
        f = 1.0
    return (int(255 * r * f), int(255 * g * f), int(255 * b * f))


def _dashed_line(surface: pygame.Surface, color: Color, p0: Tuple[float, float], p1: Tuple[float, float],
                 dash: int = 5, gap: int = 4, width: int = 1) -> None:
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        pygame.draw.line(surface, color, (x0 + ux * pos, y0 + uy * pos), (x0 + ux * end, y0 + uy * end), width)
        pos = end + gap


def _dashed_polyline(surface: pygame.Surface, color: Color, pts: Sequence[Tuple[float, float]],
                     dash: int = 4, gap: int = 4, width: int = 1) -> None:
    if len(pts) < 2:
        return
    on = True
    carry = 0.0
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg < 1e-6:
            continue
        ux, uy = (x1 - x0) / seg, (y1 - y0) / seg
        pos = 0.0
        while pos < seg:
            step = (dash if on else gap) - carry
            end = min(pos + step, seg)
            if on:
                pygame.draw.line(surface, color, (x0 + ux * pos, y0 + uy * pos), (x0 + ux * end, y0 + uy * end), width)
            if end >= seg:
                carry += seg - pos
                break
            carry = 0.0
            on = not on
            pos = end


def _fmt(v: float, digits: int = 2) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e5 or a < 1e-3:
        return f"{v:.{digits}e}"
    if a >= 100:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.1f}"
    return f"{v:.{digits}f}"


class StarDataPanel:
    """Tabbed instrument panel for the selected star. Non-modal: it sits over the sandbox
    viewport and leaves the customization panel interactive so edits update the charts live."""

    def __init__(self, visualizer: Any):
        self.viz = visualizer
        self.visible = False
        self.tab = "spectrum"
        self.star_id: Optional[str] = None
        self.panel_rect: Optional[pygame.Rect] = None
        self.close_rect: Optional[pygame.Rect] = None
        self.save_rect: Optional[pygame.Rect] = None
        self.tab_rects: Dict[str, pygame.Rect] = {}
        self.hr_show_others = True
        self.hr_others_rect: Optional[pygame.Rect] = None
        self._opened_at = 0.0
        self._track_cache_key: Optional[Tuple[float, float]] = None
        self._track_cache: List[sd.StarState] = []
        self._hz_cache_key: Optional[float] = None
        self._hz_cache: List[sd.HZTimeSample] = []

    # ------------------------------------------------------------------ state
    @property
    def screen(self) -> pygame.Surface:
        return self.viz.screen

    @property
    def theme(self):
        return getattr(self.viz, "theme", THEME)

    def open(self, star: Optional[Dict[str, Any]], tab: Optional[str] = None) -> None:
        if not star or star.get("type") != "star":
            return
        if tab:
            self.tab = tab
        self.star_id = star.get("id")
        if not self.visible:
            self._opened_at = time.time()
        self.visible = True

    def close(self) -> None:
        self.visible = False

    def _current_star(self) -> Optional[Dict[str, Any]]:
        body = getattr(self.viz, "selected_body", None)
        if body and body.get("type") == "star":
            return body
        return None

    def _other_stars(self, star: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = []
        for b in getattr(self.viz, "placed_bodies", []):
            if b.get("type") == "star" and b is not star and not b.get("is_destroyed"):
                out.append(b)
        return out

    def _provenance(self, star: Dict[str, Any]) -> Tuple[str, str]:
        """('measured'|'modified'|'assumed', label) based on the star's catalog preset."""
        presets = getattr(self.viz, "star_presets", {}) or {}
        preset = presets.get(star.get("name"))
        if not preset:
            return "assumed", "CUSTOM STAR"
        # Preset loaders re-derive some values (e.g. L from R and T), so allow a small
        # relative tolerance; user edits through the dropdowns are far larger than this.
        for key in ("mass", "radius", "temperature", "luminosity"):
            try:
                ref = float(preset.get(key, 0.0))
                if abs(float(star.get(key, 0.0)) - ref) > 0.025 * max(1e-9, abs(ref)):
                    return "modified", "MODIFIED FROM CATALOG"
            except Exception:
                return "modified", "MODIFIED FROM CATALOG"
        return "measured", "CATALOG VALUES"

    # ----------------------------------------------------------------- events
    def _input_blocked(self) -> bool:
        v = self.viz
        try:
            if v._any_dropdown_active():
                return True
        except Exception:
            pass
        for flag in ("show_engulfment_modal", "show_planet_moon_engulfment_modal", "show_placement_engulfment_modal",
                     "show_custom_modal", "reset_system_confirm_modal", "system_menu_confirm_modal",
                     "save_system_modal_visible", "system_seed_export_modal_visible", "system_seed_import_modal_visible",
                     "save_prefab_modal_visible", "show_about_panel", "show_export_panel", "show_info_panel"):
            if getattr(v, flag, False):
                return True
        diag = getattr(v, "diagnostics_panel", None)
        if diag is not None and getattr(diag, "visible", False):
            return True
        return False

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Returns True if the event was consumed."""
        if self._input_blocked():
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            # Launcher buttons live in the star customization panel (rects set by main_window).
            launcher = getattr(self.viz, "star_data_tab_button_rects", None) or {}
            star = self._current_star()
            if star and getattr(self.viz, "show_customization_panel", False):
                for tab_id, rect in launcher.items():
                    if rect and rect.collidepoint(pos):
                        if self.visible and self.tab == tab_id:
                            self.close()
                        else:
                            self.open(star, tab_id)
                        return True
            if not self.visible or not self.panel_rect:
                return False
            if self.close_rect and self.close_rect.collidepoint(pos):
                self.close()
                return True
            if self.save_rect and self.save_rect.collidepoint(pos):
                self._save_png()
                return True
            if self.hr_others_rect and self.hr_others_rect.collidepoint(pos):
                self.hr_show_others = not self.hr_show_others
                return True
            for tab_id, rect in self.tab_rects.items():
                if rect.collidepoint(pos):
                    self.tab = tab_id
                    return True
            if self.panel_rect.collidepoint(pos):
                return True  # swallow clicks on the panel body so the sandbox doesn't react
            return False
        if self.visible and self.panel_rect and event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEWHEEL):
            pos = getattr(event, "pos", None) or self.viz._mouse_pos()
            if self.panel_rect.collidepoint(pos):
                return True
        if self.visible and event.type == pygame.KEYDOWN:
            hotkeys = (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7)[:len(TABS)]
            if event.key in hotkeys and not self._text_input_active():
                idx = hotkeys.index(event.key)
                self.tab = TABS[idx][0]
                return True
        return False

    def _text_input_active(self) -> bool:
        v = self.viz
        if any(getattr(v, f, False) for f in ("rename_edit_active", "show_rename_edit", "custom_modal_visible", "show_custom_modal")):
            return True
        # Any customization-panel numeric field currently taking keyboard input.
        return any(name.endswith("_input_active") and bool(val) for name, val in vars(v).items())

    def _save_png(self) -> None:
        if not self.panel_rect:
            return
        try:
            out_dir = self.viz.create_export_directory()
        except Exception:
            out_dir = "exports"
            os.makedirs(out_dir, exist_ok=True)
        star = self._current_star() or {}
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(star.get("name", "star")))
        path = os.path.join(out_dir, f"star_data_{safe}_{self.tab}_{time.strftime('%Y%m%d_%H%M%S')}.png")
        try:
            sub = self.screen.subsurface(self.panel_rect.clip(self.screen.get_rect()))
            pygame.image.save(sub.copy(), path)
            if hasattr(self.viz, "_show_export_toast"):
                self.viz._show_export_toast(path)
        except Exception as exc:  # pragma: no cover
            print(f"[StarData] PNG export failed: {exc}")

    # ----------------------------------------------------------------- render
    def _fonts(self):
        v = self.viz
        return getattr(v, "headline_font", v.subtitle_font), v.tab_font, v.tiny_font, v.micro_font, v.font

    def render(self) -> None:
        star = self._current_star()
        if not self.visible:
            return
        if star is None or not getattr(self.viz, "show_customization_panel", False):
            # Selection moved off a star (or the panel was closed) → hide.
            self.visible = False
            return
        self.star_id = star.get("id")
        theme = self.theme
        v = self.viz
        title_font, label_font, small_font, micro_font, _ = self._fonts()

        top = v.tab_height + 2 * v.tab_margin + 12
        left = 16
        right = v.width - v.customization_panel_width - 16
        bottom = v.height - 16
        rect = pygame.Rect(left, top, right - left, bottom - top)
        self.panel_rect = rect

        ui_theme.draw_modal_frame(self.screen, rect, tone="neutral", scrim=False, theme=theme)

        pad = theme.space_lg
        mouse = v._mouse_pos()

        # Header ----------------------------------------------------------------
        derived = sd.derive_star_quantities(star)
        color = derived.color_rgb
        swatch_c = (rect.left + pad + 9, rect.top + pad + 9)
        pygame.draw.circle(self.screen, color, swatch_c, 9)
        pygame.draw.circle(self.screen, theme.panel_border, swatch_c, 9, 1)
        title = title_font.render(f"Star Data · {star.get('name', 'Star')}", True, theme.text_primary)
        title_rect = title.get_rect(midleft=(swatch_c[0] + 18, swatch_c[1]))
        self.screen.blit(title, title_rect)
        cls_label = str(star.get("spectral_class") or f"{derived.class_from_temperature}-type")
        kind, prov_label = self._provenance(star)
        tag_rect = ui_theme.draw_tag(self.screen, micro_font, cls_label, (title_rect.right + 10, title_rect.centery - 8),
                                     kind="measured" if kind == "measured" else "modified", theme=theme)
        ui_theme.draw_tag(self.screen, micro_font, prov_label, (tag_rect.right + 6, title_rect.centery - 8),
                          kind=kind, theme=theme)

        close_size = 22
        self.close_rect = pygame.Rect(rect.right - close_size - 12, rect.top + 12, close_size, close_size)
        ui_theme.draw_close_x(self.screen, self.close_rect, hover=self.close_rect.collidepoint(mouse), theme=theme)
        self.save_rect = pygame.Rect(self.close_rect.left - 10 - 92, rect.top + 11, 92, 26)
        ui_theme.draw_button(self.screen, self.save_rect, "Save PNG", small_font, kind="secondary",
                             hover=self.save_rect.collidepoint(mouse), theme=theme)

        # Tabs ------------------------------------------------------------------
        tab_y = rect.top + pad + 30
        tab_h = 28
        x = rect.left + pad
        self.tab_rects = {}
        for i, (tab_id, label) in enumerate(TABS):
            w = label_font.size(label)[0] + 26
            r = pygame.Rect(x, tab_y, w, tab_h)
            self.tab_rects[tab_id] = r
            active = tab_id == self.tab
            ui_theme.draw_button(self.screen, r, label, small_font, kind="secondary",
                                 hover=r.collidepoint(mouse), active=active, theme=theme)
            if active:
                pygame.draw.rect(self.screen, theme.accent, pygame.Rect(r.left + 8, r.bottom + 4, r.width - 16, 2))
            hint = micro_font.render(str(i + 1), True, theme.text_tertiary)
            self.screen.blit(hint, hint.get_rect(midright=(r.right - 6, r.top + 7)))
            x = r.right + 6
        ui_theme.divider(self.screen, rect.left + pad, tab_y + tab_h + 10, rect.width - 2 * pad, theme)

        content = pygame.Rect(rect.left + pad, tab_y + tab_h + 20, rect.width - 2 * pad, rect.bottom - pad - (tab_y + tab_h + 20))
        prev_clip = self.screen.get_clip()
        self.screen.set_clip(content.inflate(4, 4))
        try:
            if self.tab == "spectrum":
                self._render_spectrum(content, star, derived)
            elif self.tab == "hr":
                self._render_hr(content, star, derived)
            elif self.tab == "evolution":
                self._render_evolution(content, star, derived)
            elif self.tab == "hz":
                self._render_hz_time(content, star, derived)
            else:
                self._render_properties(content, star, derived)
        finally:
            self.screen.set_clip(prev_clip)

    # --------------------------------------------------------------- helpers
    def _chart_frame(self, rect: pygame.Rect) -> None:
        theme = self.theme
        pygame.draw.rect(self.screen, theme.field_bg, rect, border_radius=theme.radius_sm)
        pygame.draw.rect(self.screen, theme.panel_border, rect, 1, border_radius=theme.radius_sm)

    def _readout(self, x: int, y: int, label: str, value: str, value_color: Optional[Color] = None, width: int = 170) -> pygame.Rect:
        theme = self.theme
        _, _, small_font, micro_font, _ = self._fonts()
        lab = micro_font.render(label.upper(), True, theme.text_tertiary)
        val = small_font.render(value, True, value_color or theme.text_primary)
        self.screen.blit(lab, (x, y))
        self.screen.blit(val, (x, y + 13))
        return pygame.Rect(x, y, width, 34)

    def _wrapped(self, text: str, font: pygame.font.Font, color: Color, x: int, y: int, max_w: int, line_h: Optional[int] = None) -> int:
        lh = line_h or (font.get_linesize() + 1)
        for line in ui_theme.wrap_text(font, text, max_w):
            self.screen.blit(font.render(line, True, color), (x, y))
            y += lh
        return y

    # -------------------------------------------------------------- spectrum
    def _render_spectrum(self, area: pygame.Rect, star: Dict[str, Any], d: sd.DerivedStarQuantities) -> None:
        theme = self.theme
        title_font, label_font, small_font, micro_font, _ = self._fonts()
        t_eff = d.t_eff_k
        cls = sd.spectral_class_from_temperature(t_eff)
        peak = d.peak_wavelength_nm
        band_name = next((name for lo, hi, name in sd.BANDS if lo <= peak < hi), "Infrared" if peak >= 1300 else "Far-UV")
        vis_frac = sd.band_fraction(t_eff, 380, 750)
        uv_frac = sd.band_fraction(t_eff, 10, 380)

        # Readouts
        y = area.top
        x = area.left
        self._readout(x, y, "Effective temperature", f"{t_eff:,.0f} K", d.color_rgb, width=150); x += 150
        self._readout(x, y, "Wien peak λmax", f"{peak:,.0f} nm · {band_name}", width=170); x += 170
        self._readout(x, y, "Flux in visible / UV", f"{vis_frac * 100:.0f}% / {uv_frac * 100:.0f}%", width=140); x += 140
        sig = cls.signature
        while micro_font.size(sig)[0] > area.right - x - 4 and len(sig) > 8:
            sig = sig[:-4].rstrip() + "…"
        self._readout(x, y, f"Class {cls.letter} from T_eff", sig, width=area.right - x)

        # Chart geometry
        chart = pygame.Rect(area.left, area.top + 52, area.width, area.height - 52 - 92)
        self._chart_frame(chart)
        plot = pygame.Rect(chart.left + 46, chart.top + 34, chart.width - 62, chart.height - 78)
        strip_h = 14
        strip = pygame.Rect(plot.left, plot.bottom + 6, plot.width, strip_h)

        lam_min, lam_max = 100.0, 1300.0
        segments = ((100.0, 380.0, 0.0, 0.10), (380.0, 750.0, 0.10, 0.54), (750.0, 1300.0, 0.54, 1.0))

        def xpos(lam: float) -> float:
            for lo, hi, f0, f1 in segments:
                if lo <= lam <= hi:
                    return plot.left + (f0 + (f1 - f0) * (lam - lo) / (hi - lo)) * plot.width
            return plot.left if lam < lam_min else plot.right

        def ypos(v: float) -> float:
            return plot.bottom - v * plot.height * 0.92

        # Band shading + labels
        for lo, hi, name in sd.BANDS:
            bx0, bx1 = xpos(lo), xpos(hi)
            shade = pygame.Surface((max(1, int(bx1 - bx0)), plot.height), pygame.SRCALPHA)
            shade.fill((255, 255, 255, 6 if name == "Visible" else 0))
            self.screen.blit(shade, (bx0, plot.top))
            lab = micro_font.render(name.upper(), True, theme.text_tertiary)
            self.screen.blit(lab, lab.get_rect(midtop=((bx0 + bx1) / 2, chart.top + 8)))
            if lo > lam_min:
                _dashed_line(self.screen, theme.chart_grid, (bx0, plot.top), (bx0, plot.bottom), 3, 4)

        # Grid + ticks
        for lam in (200, 380, 450, 500, 550, 600, 650, 700, 750, 900, 1100, 1300):
            gx = xpos(lam)
            pygame.draw.line(self.screen, theme.chart_grid, (gx, plot.top), (gx, plot.bottom), 1)
            t = micro_font.render(str(lam), True, theme.chart_axis_text)
            self.screen.blit(t, t.get_rect(midtop=(gx, strip.bottom + 4)))
        for frac in (0.25, 0.5, 0.75, 1.0):
            gy = ypos(frac)
            pygame.draw.line(self.screen, theme.chart_grid, (plot.left, gy), (plot.right, gy), 1)
            t = micro_font.render(f"{frac:.2f}", True, theme.chart_axis_text)
            self.screen.blit(t, t.get_rect(midright=(plot.left - 6, gy)))
        pygame.draw.line(self.screen, theme.chart_axis, (plot.left, plot.top), (plot.left, plot.bottom), 1)
        pygame.draw.line(self.screen, theme.chart_axis, (plot.left, plot.bottom), (plot.right, plot.bottom), 1)
        ax = micro_font.render("Wavelength λ (nm)", True, theme.text_tertiary)
        self.screen.blit(ax, ax.get_rect(midtop=(plot.centerx, strip.bottom + 16)))
        ay = pygame.transform.rotate(micro_font.render("Relative B_λ (peak = 1)", True, theme.text_tertiary), 90)
        self.screen.blit(ay, ay.get_rect(center=(chart.left + 14, plot.centery)))

        # Spectrum strip (visible band rendered as a rainbow, with class-appropriate absorption lines)
        for px in range(int(xpos(380)), int(xpos(750)) + 1):
            lam = 380 + (750 - 380) * (px - xpos(380)) / max(1.0, xpos(750) - xpos(380))
            c = _wavelength_to_rgb(lam)
            pygame.draw.line(self.screen, c, (px, strip.top), (px, strip.bottom))
        pygame.draw.rect(self.screen, theme.panel_border, strip, 1)

        # Curves
        pts_star = [(xpos(l), ypos(val)) for l, val in sd.spectrum_samples(t_eff, lam_min, lam_max, 260)]
        fill_pts = [(plot.left, plot.bottom)] + pts_star + [(plot.right, plot.bottom)]
        fill = pygame.Surface((plot.width + 1, plot.height + 1), pygame.SRCALPHA)
        pygame.draw.polygon(fill, (*d.color_rgb, 38), [(px - plot.left, py - plot.top) for px, py in fill_pts])
        self.screen.blit(fill, plot.topleft)
        if abs(t_eff - sd.T_SUN_K) > 30:
            pts_sun = [(xpos(l), ypos(val)) for l, val in sd.spectrum_samples(sd.T_SUN_K, lam_min, lam_max, 200)]
            _dashed_polyline(self.screen, theme.chart_reference, pts_sun, 4, 4, 1)
        pygame.draw.lines(self.screen, d.color_rgb, False, pts_star, 2)

        # Wien marker
        if lam_min <= peak <= lam_max:
            wx = xpos(peak)
            _dashed_line(self.screen, theme.accent_soft, (wx, plot.top), (wx, plot.bottom), 4, 3)
            wl = micro_font.render(f"λmax {peak:.0f} nm", True, theme.accent_soft)
            self.screen.blit(wl, wl.get_rect(midbottom=(min(max(wx, plot.left + 40), plot.right - 40), plot.top - 2)))
        else:
            wl = micro_font.render(f"λmax = {peak:,.0f} nm (outside plotted range)", True, theme.accent_soft)
            self.screen.blit(wl, wl.get_rect(topright=(plot.right, plot.top + 4)))

        # Absorption features appropriate to the class (positions only)
        letter = cls.letter
        def show_line(species: str, label: str) -> bool:
            if label == "Lyα":
                return True
            if species == "H I":
                return letter in "OBAFGK"
            if species == "Ca II":
                return letter in "AFGKM"
            if species == "Mg I":
                return letter in "FGKM"
            if species == "Na I":
                return letter in "GKM"
            return True
        placed: List[Tuple[float, int]] = []
        for lam, label, species in sd.SPECTRAL_LINES:
            if not show_line(species, label):
                continue
            lx = xpos(lam)
            strong = (species == "H I" and letter in "BAF") or (species == "Ca II" and letter in "GKM")
            col = _mix(theme.field_bg, theme.text_primary, 0.15)
            pygame.draw.line(self.screen, col, (lx, strip.top + 1), (lx, strip.bottom - 1), 2 if strong else 1)
            pygame.draw.line(self.screen, theme.chart_line_marker, (lx, plot.bottom - 10), (lx, plot.bottom), 1)
            # stagger labels onto up to three rows so neighbours don't overlap
            row = 0
            while row < 2 and any(r == row and abs(lx - px) < 34 for px, r in placed):
                row += 1
            placed.append((lx, row))
            tl = micro_font.render(label, True, theme.text_secondary if strong else theme.text_tertiary)
            self.screen.blit(tl, tl.get_rect(midbottom=(lx, plot.bottom - 12 - row * 11)))

        # Legend + note
        ly = chart.bottom + 10
        pygame.draw.line(self.screen, d.color_rgb, (area.left, ly + 7), (area.left + 18, ly + 7), 2)
        first = small_font.render(f"{star.get('name', 'Star')} · idealized blackbody at {t_eff:,.0f} K", True, theme.text_secondary)
        self.screen.blit(first, (area.left + 24, ly))
        if abs(t_eff - sd.T_SUN_K) > 30:
            lx0 = area.left + 24 + first.get_width() + 28
            _dashed_line(self.screen, theme.chart_reference, (lx0, ly + 7), (lx0 + 18, ly + 7), 4, 3)
            self.screen.blit(small_font.render("Sun, 5,778 K (shape reference)", True, theme.text_secondary), (lx0 + 24, ly))
        note = ("Each curve is normalized to its own peak. Continuum is a Planck blackbody for the star's T_eff. Line markers "
                "are laboratory rest wavelengths of features typical for this spectral class (Fraunhofer / NIST); line depths "
                "and the real, non-blackbody photospheric spectrum are not modeled.")
        self._wrapped(note, micro_font, theme.text_tertiary, area.left, ly + 22, area.width)

    # ------------------------------------------------------------------ H-R
    def _evolution_track(self, mass: float) -> Tuple[sd.EvolutionAnchor, List[sd.StarState]]:
        key = (round(mass, 4), 0.0)
        anchor = sd.interpolate_evolution(mass)
        if self._track_cache_key != key:
            self._track_cache_key = key
            self._track_cache = sd.sample_evolution_track(anchor, 80)
        return anchor, self._track_cache

    def _render_hr(self, area: pygame.Rect, star: Dict[str, Any], d: sd.DerivedStarQuantities) -> None:
        theme = self.theme
        title_font, label_font, small_font, micro_font, _ = self._fonts()
        side_w = 214
        chart = pygame.Rect(area.left, area.top, area.width - side_w - 12, area.height)
        self._chart_frame(chart)
        plot = pygame.Rect(chart.left + 48, chart.top + 30, chart.width - 64, chart.height - 66)

        t_min, t_max = 2000.0, 45000.0
        l_min, l_max = -4.5, 6.5
        lt_min, lt_max = math.log10(t_min), math.log10(t_max)

        def xpos(t: float) -> float:
            t = max(t_min, min(t_max, t))
            return plot.left + (1 - (math.log10(t) - lt_min) / (lt_max - lt_min)) * plot.width

        def ypos(log_l: float) -> float:
            log_l = max(l_min, min(l_max, log_l))
            return plot.bottom - (log_l - l_min) / (l_max - l_min) * plot.height

        # Spectral-class bands along the top
        bounds = [(45000, "O"), (30000, "B"), (10000, "A"), (7500, "F"), (6000, "G"), (5200, "K"), (3700, "M"), (2000, None)]
        for i in range(len(bounds) - 1):
            hi_t, letter = bounds[i]
            lo_t = bounds[i + 1][0]
            x0, x1 = xpos(hi_t), xpos(lo_t)
            if lo_t not in (2000,):
                _dashed_line(self.screen, theme.chart_grid, (x1, plot.top), (x1, plot.bottom), 3, 5)
            band_col = sd.temperature_to_rgb((hi_t + lo_t) / 2)
            band = pygame.Surface((max(1, int(x1 - x0)), 4), pygame.SRCALPHA)
            band.fill((*band_col, 110))
            self.screen.blit(band, (x0, plot.top - 8))
            lab = micro_font.render(letter, True, theme.text_secondary)
            self.screen.blit(lab, lab.get_rect(midbottom=((x0 + x1) / 2, plot.top - 10)))

        # Grid
        for t in (30000, 10000, 6000, 4000, 3000):
            gx = xpos(t)
            pygame.draw.line(self.screen, theme.chart_grid, (gx, plot.top), (gx, plot.bottom), 1)
            tl = micro_font.render(f"{t:,}", True, theme.chart_axis_text)
            self.screen.blit(tl, tl.get_rect(midtop=(gx, plot.bottom + 4)))
        for ll in range(-4, 7, 2):
            gy = ypos(ll)
            pygame.draw.line(self.screen, theme.chart_grid, (plot.left, gy), (plot.right, gy), 1)
            tl = micro_font.render(f"{ll:+d}" if ll else "0", True, theme.chart_axis_text)
            self.screen.blit(tl, tl.get_rect(midright=(plot.left - 6, gy)))
        pygame.draw.line(self.screen, theme.chart_axis, (plot.left, plot.top), (plot.left, plot.bottom), 1)
        pygame.draw.line(self.screen, theme.chart_axis, (plot.left, plot.bottom), (plot.right, plot.bottom), 1)
        ax = micro_font.render("Effective temperature T_eff (K) — hotter →  ← cooler", True, theme.text_tertiary)
        self.screen.blit(ax, ax.get_rect(midtop=(plot.centerx, plot.bottom + 16)))
        ay = pygame.transform.rotate(micro_font.render("log₁₀ (L / Lsun)", True, theme.text_tertiary), 90)
        self.screen.blit(ay, ay.get_rect(center=(chart.left + 14, plot.centery)))

        # Main-sequence locus (soft band)
        ms_pts = [(xpos(t), ypos(l)) for t, l in sd.MAIN_SEQUENCE_LOCUS]
        band_surf = pygame.Surface(plot.size, pygame.SRCALPHA)
        pygame.draw.lines(band_surf, (*theme.chart_ms_band, 70), False, [(x - plot.left, y - plot.top) for x, y in ms_pts], 14)
        self.screen.blit(band_surf, plot.topleft)
        pygame.draw.lines(self.screen, theme.chart_ms_band, False, ms_pts, 1)
        for text, t, l in sd.HR_REGION_LABELS:
            lab = micro_font.render(text, True, theme.text_tertiary)
            self.screen.blit(lab, lab.get_rect(center=(xpos(t), ypos(l))))

        # Schematic evolutionary track for this mass + current-age marker
        anchor, track = self._evolution_track(d.mass_solar)
        track_pts = [(xpos(s.temp_k), ypos(s.log_l)) for s in track if s.temp_k >= 600]
        if len(track_pts) > 1:
            _dashed_polyline(self.screen, theme.chart_track, track_pts, 3, 4, 1)
        phase_now = sd.phase_from_age(d.age_gyr * 1e9, anchor)
        state_now = sd.compute_star_state(phase_now, anchor)
        mx, my = xpos(state_now.temp_k), ypos(state_now.log_l)
        pygame.draw.circle(self.screen, theme.chart_track, (int(mx), int(my)), 4, 1)

        # Markers: selected star first so reference labels can dodge it.
        log_l = math.log10(d.luminosity_solar)
        px, py = xpos(d.t_eff_k), ypos(log_l)
        placed_labels: List[pygame.Rect] = []

        def place_label(surf: pygame.Surface, ax: float, ay: float) -> None:
            r = surf.get_rect(midleft=(ax + 8, ay))
            candidates = (r, r.move(0, -14), r.move(0, 14), r.move(-r.width - 16, 0), r.move(0, -28), r.move(0, 28))
            for c in candidates:
                if plot.contains(c) and not any(c.colliderect(o) for o in placed_labels):
                    r = c
                    break
            placed_labels.append(r)
            self.screen.blit(surf, r)

        halo = pygame.Surface((36, 36), pygame.SRCALPHA)
        pygame.draw.circle(halo, (*theme.selection_ring, 60), (18, 18), 16)
        self.screen.blit(halo, (px - 18, py - 18))
        pygame.draw.circle(self.screen, d.color_rgb, (int(px), int(py)), 7)
        pygame.draw.circle(self.screen, theme.selection_ring, (int(px), int(py)), 9, 2)
        name_s = small_font.render(str(star.get("name", "Star")), True, theme.text_primary)
        name_r = name_s.get_rect(midleft=(px + 14, py))
        placed_labels.append(name_r.inflate(4, 4))
        placed_labels.append(pygame.Rect(px - 10, py - 10, 20, 20))
        self.screen.blit(name_s, name_r)

        # Sun reference
        sx, sy = xpos(sd.T_SUN_K), ypos(0.0)
        pygame.draw.circle(self.screen, theme.chart_reference, (int(sx), int(sy)), 5, 1)
        pygame.draw.circle(self.screen, theme.chart_reference, (int(sx), int(sy)), 1)
        if abs(d.t_eff_k - sd.T_SUN_K) > 60 or abs(log_l) > 0.08:
            place_label(micro_font.render("Sun", True, theme.chart_reference), sx, sy)

        # Other stars in the sandbox
        if self.hr_show_others:
            for other in self._other_stars(star):
                try:
                    ot, ol = float(other.get("temperature", 5778)), math.log10(max(1e-9, float(other.get("luminosity", 1.0))))
                except Exception:
                    continue
                ox, oy = xpos(ot), ypos(ol)
                pygame.draw.circle(self.screen, sd.temperature_to_rgb(ot), (int(ox), int(oy)), 4)
                pygame.draw.circle(self.screen, theme.panel_border, (int(ox), int(oy)), 4, 1)
                place_label(micro_font.render(str(other.get("name", "Star")), True, theme.text_tertiary), ox, oy)

        # Side readouts
        sx0 = chart.right + 12
        y = area.top
        self._readout(sx0, y, "log₁₀ T_eff", f"{math.log10(d.t_eff_k):.3f}  ({d.t_eff_k:,.0f} K)", width=side_w); y += 40
        self._readout(sx0, y, "log₁₀ L / Lsun", f"{log_l:+.3f}  ({_fmt(d.luminosity_solar)} Lsun)", width=side_w); y += 40
        self._readout(sx0, y, "Absolute bolometric magnitude", f"M_bol = {d.abs_bol_magnitude:+.2f}", width=side_w); y += 40
        # Offset from the main-sequence locus at this temperature
        ms_l = self._ms_log_l_at(d.t_eff_k)
        delta = log_l - ms_l
        if abs(delta) < 0.5:
            pos_txt, pos_col = "On the main sequence", theme.success_soft
        elif delta > 0:
            pos_txt, pos_col = "Above the main sequence (evolved / giant)", theme.warning_soft
        else:
            pos_txt, pos_col = "Below the main sequence (compact)", theme.accent_soft
        self._readout(sx0, y, "Δ log L from MS locus", f"{delta:+.2f} dex", width=side_w); y += 40
        y = self._wrapped(pos_txt, small_font, pos_col, sx0, y, side_w) + 8
        ui_theme.divider(self.screen, sx0, y, side_w, theme); y += 10
        self._readout(sx0, y, "Model track", f"{anchor.name} · → {anchor.endpoint}", width=side_w); y += 40
        self._readout(sx0, y, "Age marker on track", f"{d.age_gyr:.2f} Gyr · {sd.phase_label(phase_now, anchor.endpoint, sd.has_giant_branch(anchor)).split(' (')[0]}", width=side_w); y += 44

        # Legend
        def legend(yy: int, draw_marker, text: str) -> int:
            draw_marker(sx0 + 8, yy + 7)
            self.screen.blit(micro_font.render(text, True, theme.text_secondary), (sx0 + 20, yy))
            return yy + 16
        y = legend(y, lambda x, yy: pygame.draw.line(self.screen, theme.chart_ms_band, (x - 7, yy), (x + 7, yy), 3), "Main-sequence locus")
        y = legend(y, lambda x, yy: _dashed_line(self.screen, theme.chart_track, (x - 7, yy), (x + 7, yy), 3, 3), "Schematic track for this mass")
        y = legend(y, lambda x, yy: pygame.draw.circle(self.screen, theme.chart_reference, (x, yy), 4, 1), "Sun (5,778 K, 1 Lsun)")
        y = legend(y, lambda x, yy: pygame.draw.circle(self.screen, theme.selection_ring, (x, yy), 5, 2), "Selected star (stated T, L)")
        y += 6
        # Toggle for other stars
        self.hr_others_rect = pygame.Rect(sx0, y, side_w, 24)
        ui_theme.draw_button(self.screen, self.hr_others_rect,
                             "Hide other stars" if self.hr_show_others else "Show other stars", micro_font,
                             kind="ghost", hover=self.hr_others_rect.collidepoint(self.viz._mouse_pos()), theme=theme)
        y += 32
        self._wrapped("Track uses log-mass interpolated MIST/Padova-style checkpoints and is illustrative; "
                      "the star marker uses the values set in the panel.", micro_font, theme.text_tertiary, sx0, y, side_w)

    @staticmethod
    def _ms_log_l_at(t_eff: float) -> float:
        pts = sorted(sd.MAIN_SEQUENCE_LOCUS)  # ascending T
        if t_eff <= pts[0][0]:
            return pts[0][1]
        if t_eff >= pts[-1][0]:
            return pts[-1][1]
        for (t0, l0), (t1, l1) in zip(pts, pts[1:]):
            if t0 <= t_eff <= t1:
                f = (math.log10(t_eff) - math.log10(t0)) / (math.log10(t1) - math.log10(t0))
                return l0 + (l1 - l0) * f
        return 0.0

    # -------------------------------------------------------------- evolution
    def _render_evolution(self, area: pygame.Rect, star: Dict[str, Any], d: sd.DerivedStarQuantities) -> None:
        theme = self.theme
        title_font, label_font, small_font, micro_font, _ = self._fonts()
        anchor, _ = self._evolution_track(d.mass_solar)
        age_yr = d.age_gyr * 1e9
        phase = sd.phase_from_age(age_yr, anchor)
        beyond = age_yr > anchor.t_total
        state = sd.compute_star_state(phase, anchor)

        # Timeline ------------------------------------------------------------
        bar = pygame.Rect(area.left, area.top + 18, area.width, 22)
        segs = (
            ("Pre-main sequence", 0.0, sd.PHASE_PROTO_END, anchor.t_proto, theme.chart_track),
            ("Main sequence", sd.PHASE_PROTO_END, sd.PHASE_MS_END, anchor.t_ms, theme.accent),
            ("Giant branch" if sd.has_giant_branch(anchor) else "Contraction", sd.PHASE_MS_END, sd.PHASE_GIANT_END, anchor.t_giant, theme.warning),
            (anchor.endpoint, sd.PHASE_GIANT_END, 1.0, anchor.t_total - anchor.t_proto - anchor.t_ms - anchor.t_giant, theme.text_tertiary),
        )
        hdr = micro_font.render("LIFECYCLE TIMELINE  ·  segment widths are schematic, not proportional to duration", True, theme.text_tertiary)
        self.screen.blit(hdr, (area.left, area.top))
        for name, p0, p1, dur, col in segs:
            r = pygame.Rect(bar.left + int(p0 * bar.width), bar.top, max(2, int((p1 - p0) * bar.width) - 2), bar.height)
            fill = _mix(theme.field_bg, col, 0.35)
            pygame.draw.rect(self.screen, fill, r, border_radius=3)
            pygame.draw.rect(self.screen, _mix(fill, col, 0.5), r, 1, border_radius=3)
            if r.width > 60:
                nm = micro_font.render(name, True, theme.text_primary)
                self.screen.blit(nm, nm.get_rect(midleft=(r.left + 8, r.centery)))
            dur_txt = sd.format_years(dur) if name != anchor.endpoint else "cooling / final state"
            dl = micro_font.render(dur_txt, True, theme.text_secondary)
            anchor_x = r.centerx if r.width > 60 else r.left
            dl_rect = dl.get_rect(midtop=(anchor_x, bar.bottom + 4))
            dl_rect.left = max(area.left, min(dl_rect.left, area.right - dl_rect.width))
            self.screen.blit(dl, dl_rect)
        # Age marker
        mx = bar.left + int(min(phase, 1.0) * bar.width)
        pygame.draw.polygon(self.screen, theme.text_primary, [(mx, bar.top - 2), (mx - 6, bar.top - 10), (mx + 6, bar.top - 10)])
        pygame.draw.line(self.screen, theme.text_primary, (mx, bar.top), (mx, bar.bottom), 2)
        age_lab = small_font.render(f"now · {d.age_gyr:.2f} Gyr" + ("  (beyond model lifetime)" if beyond else ""), True,
                                    theme.warning_soft if beyond else theme.text_primary)
        self.screen.blit(age_lab, (min(max(mx - 40, area.left), area.right - age_lab.get_width()), bar.bottom + 22))

        # Two columns ---------------------------------------------------------
        col_top = bar.bottom + 50
        col_w = (area.width - 24) // 2
        lx = area.left
        rx = area.left + col_w + 24
        y = col_top

        # Left: stage + destiny + stats
        ui_theme.draw_section_label(self.screen, micro_font, "CURRENT STAGE", (lx, y), theme=theme); y += 18
        stage = sd.phase_label(phase, anchor.endpoint, sd.has_giant_branch(anchor))
        self.screen.blit(label_font.render(stage, True, theme.text_primary), (lx, y)); y += 22
        if sd.PHASE_PROTO_END <= phase < sd.PHASE_MS_END and not beyond:
            frac = (age_yr - anchor.t_proto) / max(anchor.t_ms, 1.0)
            self.screen.blit(small_font.render(f"{frac * 100:.0f}% of the main-sequence lifetime elapsed", True, theme.text_secondary), (lx, y)); y += 18
        blurb_key = "proto" if phase < sd.PHASE_PROTO_END else "ms" if phase < sd.PHASE_MS_END else ("giant" if sd.has_giant_branch(anchor) else "nogiant") if phase < sd.PHASE_GIANT_END else anchor.endpoint
        y = self._wrapped(sd.PHASE_BLURBS.get(blurb_key, ""), small_font, theme.text_secondary, lx, y + 2, col_w) + 8

        ui_theme.draw_section_label(self.screen, micro_font, "FINAL STATE", (lx, y), theme=theme); y += 18
        self.screen.blit(label_font.render(anchor.endpoint, True, theme.text_primary), (lx, y)); y += 22
        y = self._wrapped(sd.PHASE_BLURBS.get(anchor.endpoint, ""), small_font, theme.text_secondary, lx, y, col_w) + 8

        ui_theme.draw_section_label(self.screen, micro_font, "MODEL LIFETIMES", (lx, y), theme=theme); y += 18
        rows = [
            ("Main-sequence lifetime (track)", sd.format_years(anchor.t_ms)),
            ("τ_MS ≈ 10 Gyr · M^-2.5 (AIET heuristic)", f"{d.ms_lifetime_gyr:.2f} Gyr" if d.ms_lifetime_gyr < 1000 else f"{d.ms_lifetime_gyr:,.0f} Gyr"),
            ("Total lifetime to remnant (track)", sd.format_years(anchor.t_total)),
            ("Radius now (track) / stated", f"{_fmt(state.radius_solar)} / {_fmt(d.radius_solar)} Rsun"),
            ("T_eff now (track) / stated", f"{state.temp_k:,.0f} / {d.t_eff_k:,.0f} K"),
            ("Peak giant radius (track)", f"{_fmt(anchor.r_giant)} Rsun" if sd.has_giant_branch(anchor) else "none (M < 0.25 Msun)"),
        ]
        for lab, val in rows:
            self.screen.blit(small_font.render(lab, True, theme.text_secondary), (lx, y))
            vs = small_font.render(val, True, theme.text_primary)
            self.screen.blit(vs, vs.get_rect(topright=(lx + col_w, y)))
            y += 18

        # Right: interior structure schematic
        y2 = col_top
        ui_theme.draw_section_label(self.screen, micro_font, "INTERIOR STRUCTURE (SCHEMATIC)", (rx, y2), theme=theme); y2 += 22
        layers, note = sd.interior_structure(d.mass_solar, phase, anchor.endpoint)
        radius_px = 88
        cx, cy = rx + radius_px + 4, y2 + radius_px + 4
        kind_colors = {
            "core": (255, 244, 214), "radiative": (222, 150, 60), "convective": (196, 88, 44),
            "shell": (80, 160, 230), "envelope": (196, 88, 44), "surface": d.color_rgb,
        }
        for layer in reversed(layers):
            col = d.color_rgb if layer.kind == "surface" else kind_colors.get(layer.kind, theme.panel_elevated)
            rr = max(2, int(radius_px * layer.frac_radius))
            pygame.draw.circle(self.screen, col, (cx, cy), rr)
            pygame.draw.circle(self.screen, _mix(col, theme.field_bg, 0.45), (cx, cy), rr, 1)
        pygame.draw.line(self.screen, theme.field_bg, (cx, cy), (cx + radius_px, cy), 1)
        pygame.draw.line(self.screen, theme.field_bg, (cx, cy), (cx, cy - radius_px), 1)
        # Layer legend (wrapped to the remaining column width)
        ly = y2
        lxx = cx + radius_px + 16
        legend_w = max(80, rx + col_w - lxx)
        for layer in layers:
            col = d.color_rgb if layer.kind == "surface" else kind_colors.get(layer.kind, theme.panel_elevated)
            pygame.draw.rect(self.screen, col, pygame.Rect(lxx, ly + 3, 10, 10), border_radius=2)
            ly = self._wrapped(layer.label, small_font, theme.text_primary, lxx + 16, ly, legend_w - 16, 15)
            self.screen.blit(micro_font.render(f"to {layer.frac_radius * 100:.0f}% of radius", True, theme.text_tertiary), (lxx + 16, ly))
            ly += 18
        y2 = max(cy + radius_px + 10, ly)
        y2 = self._wrapped(note, small_font, theme.text_secondary, rx, y2, col_w)

        # Track profiles under both columns: radius and temperature over the lifecycle
        top = max(y, y2) + 14
        if area.bottom - top > 110:
            _, track = self._evolution_track(d.mass_solar)
            half = (area.width - 16) // 2
            charts = (
                ("RADIUS ALONG THE TRACK (Rsun, log)", [math.log10(max(1e-3, st.radius_solar)) for st in track],
                 math.log10(max(1e-3, d.radius_solar)), lambda v: f"{10 ** v:.2g}"),
                ("T_EFF ALONG THE TRACK (K, log)", [math.log10(max(300.0, st.temp_k)) for st in track],
                 math.log10(d.t_eff_k), lambda v: f"{10 ** v:,.0f}"),
            )
            for i, (title, series, stated, fmt) in enumerate(charts):
                cr = pygame.Rect(area.left + i * (half + 16), top, half, area.bottom - top)
                self.screen.blit(micro_font.render(title, True, theme.text_tertiary), (cr.left, cr.top))
                frame = pygame.Rect(cr.left, cr.top + 16, cr.width, cr.height - 16)
                self._chart_frame(frame)
                plot = pygame.Rect(frame.left + 40, frame.top + 8, frame.width - 48, frame.height - 22)
                lo, hi = min(series + [stated]), max(series + [stated])
                if hi - lo < 0.2:
                    lo, hi = lo - 0.1, hi + 0.1
                pad_v = (hi - lo) * 0.08
                lo, hi = lo - pad_v, hi + pad_v

                def px_(ph: float) -> float:
                    return plot.left + ph * plot.width

                def py_(v: float) -> float:
                    return plot.bottom - (v - lo) / (hi - lo) * plot.height

                for name, p0, p1, _dur, col in segs:
                    seg_r = pygame.Rect(int(px_(p0)), plot.top, max(1, int((p1 - p0) * plot.width)), plot.height)
                    sh = pygame.Surface(seg_r.size, pygame.SRCALPHA)
                    sh.fill((*col, 18))
                    self.screen.blit(sh, seg_r.topleft)
                for k in range(3):
                    v = lo + (hi - lo) * (k + 0.5) / 3
                    gy = py_(v)
                    pygame.draw.line(self.screen, theme.chart_grid, (plot.left, gy), (plot.right, gy), 1)
                    tl = micro_font.render(fmt(v), True, theme.chart_axis_text)
                    self.screen.blit(tl, tl.get_rect(midright=(plot.left - 4, gy)))
                pygame.draw.line(self.screen, theme.chart_axis, (plot.left, plot.top), (plot.left, plot.bottom), 1)
                pygame.draw.line(self.screen, theme.chart_axis, (plot.left, plot.bottom), (plot.right, plot.bottom), 1)
                pts = [(px_(st.phase), py_(v)) for st, v in zip(track, series)]
                if len(pts) > 1:
                    pygame.draw.lines(self.screen, theme.chart_track, False, pts, 2)
                # stated value + current age
                _dashed_line(self.screen, d.color_rgb, (plot.left, py_(stated)), (plot.right, py_(stated)), 3, 3)
                ax_ = px_(min(phase, 1.0))
                pygame.draw.line(self.screen, theme.text_primary, (ax_, plot.top), (ax_, plot.bottom), 1)
                xl = micro_font.render("lifecycle phase →", True, theme.chart_axis_text)
                self.screen.blit(xl, xl.get_rect(midtop=(plot.centerx, plot.bottom + 2)))
                sl = micro_font.render(f"stated {fmt(stated)}", True, d.color_rgb)
                self.screen.blit(sl, sl.get_rect(topright=(plot.right - 2, plot.top + 2)))

    # ----------------------------------------------------------- HZ over time
    def _planets_of(self, star: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = []
        getter = getattr(self.viz, "_get_parent_star", None)
        for b in getattr(self.viz, "placed_bodies", []):
            if b.get("type") != "planet" or b.get("is_destroyed"):
                continue
            host = None
            if callable(getter):
                try:
                    host = getter(b)
                except Exception:
                    host = None
            if host is star or (host is None and b.get("parent_id") == star.get("id")):
                out.append(b)
        out.sort(key=lambda p: float(p.get("semiMajorAxis") or 0.0))
        return out

    def _hz_samples(self, mass: float) -> Tuple[sd.EvolutionAnchor, List[sd.HZTimeSample]]:
        key = round(mass, 4)
        if getattr(self, "_hz_cache_key", None) != key:
            self._hz_cache_key = key
            self._hz_cache = sd.hz_over_lifetime(sd.interpolate_evolution(mass), 360)
        return sd.interpolate_evolution(mass), self._hz_cache

    def _render_hz_time(self, area: pygame.Rect, star: Dict[str, Any], d: sd.DerivedStarQuantities) -> None:
        theme = self.theme
        title_font, label_font, small_font, micro_font, _ = self._fonts()
        anchor, samples = self._hz_samples(d.mass_solar)
        planets = self._planets_of(star)
        side_w = 236
        chart = pygame.Rect(area.left, area.top, area.width - side_w - 12, area.height)
        self._chart_frame(chart)
        plot = pygame.Rect(chart.left + 52, chart.top + 30, chart.width - 68, chart.height - 66)

        if not samples:
            self._wrapped("Habitable-zone module unavailable (src/physics/kopparapu_hz.py could not be imported).",
                          small_font, theme.warning_soft, plot.left, plot.top, plot.width)
            return

        age_now = d.age_gyr * 1e9
        t_total = max(anchor.t_total, age_now * 1.05, 1.0)

        # Axis ranges: linear age, log distance covering the HZ band and every planet.
        a_vals = [s.inner_au for s in samples] + [s.outer_au for s in samples]
        a_vals += [float(p.get("semiMajorAxis") or 0.0) for p in planets if float(p.get("semiMajorAxis") or 0.0) > 0]
        if d.hz_inner_au and d.hz_outer_au:
            a_vals += [d.hz_inner_au, d.hz_outer_au]
        a_vals = [v for v in a_vals if v > 0]
        lo_a = max(1e-3, min(a_vals) / 1.6)
        hi_a = min(1e4, max(a_vals) * 1.6)
        if hi_a / lo_a < 10:
            lo_a, hi_a = lo_a / 2, hi_a * 2
        llo, lhi = math.log10(lo_a), math.log10(hi_a)

        def xpos(age: float) -> float:
            return plot.left + max(0.0, min(1.0, age / t_total)) * plot.width

        def ypos(a: float) -> float:
            a = max(lo_a, min(hi_a, a))
            return plot.bottom - (math.log10(a) - llo) / (lhi - llo) * plot.height

        # Phase shading along the age axis
        phase_segs = (
            (0.0, sd.PHASE_PROTO_END, theme.chart_track),
            (sd.PHASE_PROTO_END, sd.PHASE_MS_END, theme.accent),
            (sd.PHASE_MS_END, sd.PHASE_GIANT_END, theme.warning),
            (sd.PHASE_GIANT_END, 1.0, theme.text_tertiary),
        )
        for p0, p1, col in phase_segs:
            x0 = xpos(sd.compute_star_state(p0, anchor).age_years)
            x1 = xpos(sd.compute_star_state(min(p1, 0.9999), anchor).age_years)
            if x1 - x0 >= 1:
                sh = pygame.Surface((int(x1 - x0), plot.height), pygame.SRCALPHA)
                sh.fill((*col, 14))
                self.screen.blit(sh, (x0, plot.top))
                if x1 - x0 > 70:
                    name = {0.0: "Pre-MS", sd.PHASE_PROTO_END: "Main sequence",
                            sd.PHASE_MS_END: "Giant branch" if sd.has_giant_branch(anchor) else "Contraction",
                            sd.PHASE_GIANT_END: anchor.endpoint}[p0]
                    lab = micro_font.render(name.upper(), True, theme.text_tertiary)
                    self.screen.blit(lab, lab.get_rect(midtop=((x0 + x1) / 2, chart.top + 8)))

        # Grid + ticks
        n_ticks = 6
        for i in range(n_ticks + 1):
            age = t_total * i / n_ticks
            gx = xpos(age)
            pygame.draw.line(self.screen, theme.chart_grid, (gx, plot.top), (gx, plot.bottom), 1)
            tl = micro_font.render(sd.format_years(age) if age > 0 else "0", True, theme.chart_axis_text)
            self.screen.blit(tl, tl.get_rect(midtop=(gx, plot.bottom + 4)))
        dec = int(math.floor(llo))
        while dec <= math.ceil(lhi):
            for m in (1, 2, 5):
                a = m * 10.0 ** dec
                if lo_a <= a <= hi_a:
                    gy = ypos(a)
                    pygame.draw.line(self.screen, theme.chart_grid, (plot.left, gy), (plot.right, gy), 1)
                    if m == 1 or (lhi - llo) < 2.5:
                        tl = micro_font.render(f"{a:g}", True, theme.chart_axis_text)
                        self.screen.blit(tl, tl.get_rect(midright=(plot.left - 6, gy)))
            dec += 1
        pygame.draw.line(self.screen, theme.chart_axis, (plot.left, plot.top), (plot.left, plot.bottom), 1)
        pygame.draw.line(self.screen, theme.chart_axis, (plot.left, plot.bottom), (plot.right, plot.bottom), 1)
        ax = micro_font.render("Stellar age (schematic track)", True, theme.text_tertiary)
        self.screen.blit(ax, ax.get_rect(midtop=(plot.centerx, plot.bottom + 16)))
        ay = pygame.transform.rotate(micro_font.render("Orbital distance (AU, log)", True, theme.text_tertiary), 90)
        self.screen.blit(ay, ay.get_rect(center=(chart.left + 14, plot.centery)))

        # HZ band: filled polygon between inner and outer boundaries; hatched where T_eff is
        # outside Kopparapu's validity range.
        band = pygame.Surface(plot.size, pygame.SRCALPHA)
        top_pts = [(xpos(s.age_years) - plot.left, ypos(s.outer_au) - plot.top) for s in samples]
        bot_pts = [(xpos(s.age_years) - plot.left, ypos(s.inner_au) - plot.top) for s in reversed(samples)]
        if len(top_pts) > 2:
            pygame.draw.polygon(band, (*theme.chart_hz_band, 70), top_pts + bot_pts)
        # Hatch invalid segments
        for s0, s1 in zip(samples, samples[1:]):
            if s0.teff_in_range and s1.teff_in_range:
                continue
            x0, x1 = xpos(s0.age_years) - plot.left, xpos(s1.age_years) - plot.left
            if x1 - x0 < 0.5:
                x1 = x0 + 1
            y_top = min(ypos(s0.outer_au), ypos(s1.outer_au)) - plot.top
            y_bot = max(ypos(s0.inner_au), ypos(s1.inner_au)) - plot.top
            seg = pygame.Rect(int(x0), int(y_top), max(1, int(x1 - x0) + 1), max(1, int(y_bot - y_top)))
            pygame.draw.rect(band, (*theme.warning, 34), seg)
        self.screen.blit(band, plot.topleft)
        outer_line = [(xpos(s.age_years), ypos(s.outer_au)) for s in samples]
        inner_line = [(xpos(s.age_years), ypos(s.inner_au)) for s in samples]
        if len(outer_line) > 1:
            pygame.draw.lines(self.screen, theme.chart_hz_band, False, outer_line, 1)
            pygame.draw.lines(self.screen, theme.chart_hz_band, False, inner_line, 1)

        # Planets: horizontal orbit lines, thick + green where inside the band
        min_dur = 0.002 * t_total
        planet_rows: List[Tuple[Dict[str, Any], float, List[Tuple[float, float]], Optional[bool]]] = []
        for p in planets:
            a = float(p.get("semiMajorAxis") or 0.0)
            if a <= 0:
                continue
            py = ypos(a)
            pygame.draw.line(self.screen, theme.chart_planet, (plot.left, py), (plot.right, py), 1)
            intervals = sd.hz_intervals_for_orbit(samples, a, min_dur)
            for t0, t1 in intervals:
                pygame.draw.line(self.screen, theme.chart_hz_band, (xpos(t0), py), (xpos(t1), py), 4)
            now_in: Optional[bool] = None
            if d.hz_inner_au is not None and d.hz_outer_au is not None:
                now_in = d.hz_inner_au <= a <= d.hz_outer_au
            planet_rows.append((p, a, intervals, now_in))
            nm = micro_font.render(str(p.get("display_name") or p.get("name") or "Planet"), True, theme.text_secondary)
            self.screen.blit(nm, nm.get_rect(bottomright=(plot.right - 4, py - 1)))

        # Current age marker + stated-value HZ bracket (uses the panel's L and T, i.e. the same
        # numbers the sandbox HZ ring uses) so the track/stated mismatch is visible.
        ax_ = xpos(age_now)
        pygame.draw.line(self.screen, theme.text_primary, (ax_, plot.top), (ax_, plot.bottom), 1)
        now_lab = micro_font.render(f"now · {d.age_gyr:.2f} Gyr", True, theme.text_primary)
        nl_rect = now_lab.get_rect(topleft=(ax_ + 5, plot.top + 4))
        if nl_rect.right > plot.right - 2:
            nl_rect = now_lab.get_rect(topright=(ax_ - 5, plot.top + 4))
        bg = pygame.Surface(nl_rect.inflate(6, 2).size, pygame.SRCALPHA)
        bg.fill((*theme.field_bg, 200))
        self.screen.blit(bg, nl_rect.inflate(6, 2).topleft)
        self.screen.blit(now_lab, nl_rect)
        if d.hz_inner_au and d.hz_outer_au:
            y0, y1 = ypos(d.hz_outer_au), ypos(d.hz_inner_au)
            pygame.draw.line(self.screen, d.color_rgb, (ax_, y0), (ax_, y1), 3)
            for yy in (y0, y1):
                pygame.draw.line(self.screen, d.color_rgb, (ax_ - 5, yy), (ax_ + 5, yy), 2)

        # Side column ---------------------------------------------------------
        sx0 = chart.right + 12
        y = area.top
        ui_theme.draw_section_label(self.screen, micro_font, "HZ NOW (STATED L, T_EFF)", (sx0, y), theme=theme); y += 18
        if d.hz_inner_au and d.hz_outer_au:
            self.screen.blit(small_font.render(f"{d.hz_inner_au:.3f} – {d.hz_outer_au:.3f} AU", True, theme.text_primary), (sx0, y)); y += 18
        else:
            self.screen.blit(small_font.render("—", True, theme.text_primary), (sx0, y)); y += 18
        # Track HZ at this age for comparison
        near = min(samples, key=lambda s: abs(s.age_years - age_now))
        self.screen.blit(micro_font.render(f"track at this age: {near.inner_au:.3f} – {near.outer_au:.3f} AU", True, theme.text_tertiary), (sx0, y)); y += 18
        ui_theme.divider(self.screen, sx0, y, side_w, theme); y += 10

        ui_theme.draw_section_label(self.screen, micro_font, "PLANETS ON THIS TRACK", (sx0, y), theme=theme); y += 18
        if not planet_rows:
            y = self._wrapped("No planets orbit this star yet. Add one and its orbit will appear as a line across the band.",
                              small_font, theme.text_secondary, sx0, y, side_w) + 6
        max_rows = max(1, (area.bottom - y - 150) // 50)
        for p, a, intervals, now_in in planet_rows[:max_rows]:
            name = str(p.get("display_name") or p.get("name") or "Planet")
            col = theme.success_soft if now_in else (theme.text_secondary if now_in is None else theme.warning_soft)
            head = small_font.render(f"{name} · {a:.3g} AU", True, theme.text_primary)
            self.screen.blit(head, (sx0, y))
            status = "inside HZ now" if now_in else ("outside HZ now" if now_in is not None else "")
            st = micro_font.render(status, True, col)
            self.screen.blit(st, st.get_rect(topright=(sx0 + side_w, y + 2)))
            y += 17
            if intervals:
                total_in = sum(t1 - t0 for t0, t1 in intervals)
                spans = "; ".join(f"{sd.format_years(t0)} → {sd.format_years(t1)}" for t0, t1 in intervals[:2])
                if len(intervals) > 2:
                    spans += f"; +{len(intervals) - 2} more"
                y = self._wrapped(f"In HZ: {spans}", micro_font, theme.text_secondary, sx0, y, side_w)
                self.screen.blit(micro_font.render(f"total {sd.format_years(total_in)} of {sd.format_years(anchor.t_total)} lifetime", True, theme.text_tertiary), (sx0, y)); y += 14
            else:
                self.screen.blit(micro_font.render("never inside the HZ on this track", True, theme.text_tertiary), (sx0, y)); y += 14
            y += 6
        if len(planet_rows) > max_rows:
            self.screen.blit(micro_font.render(f"+{len(planet_rows) - max_rows} more planets (see chart)", True, theme.text_tertiary), (sx0, y)); y += 16
        y += 4
        ui_theme.divider(self.screen, sx0, y, side_w, theme); y += 10

        # Legend
        def legend(yy: int, draw_marker, text: str) -> int:
            draw_marker(sx0 + 8, yy + 7)
            self.screen.blit(micro_font.render(text, True, theme.text_secondary), (sx0 + 20, yy))
            return yy + 16
        y = legend(y, lambda x, yy: pygame.draw.rect(self.screen, _mix(theme.field_bg, theme.chart_hz_band, 0.45), pygame.Rect(x - 7, yy - 5, 14, 10)), "Conservative HZ (Kopparapu 2013)")
        y = legend(y, lambda x, yy: pygame.draw.rect(self.screen, _mix(theme.field_bg, theme.warning, 0.4), pygame.Rect(x - 7, yy - 5, 14, 10)), "T_eff outside 2600–7200 K fit range")
        y = legend(y, lambda x, yy: pygame.draw.line(self.screen, theme.chart_hz_band, (x - 7, yy), (x + 7, yy), 4), "Orbit inside the HZ")
        y = legend(y, lambda x, yy: pygame.draw.line(self.screen, d.color_rgb, (x, yy - 6), (x, yy + 6), 3), "HZ from stated L, T (sandbox ring)")
        y += 4
        self._wrapped("Same Kopparapu boundaries as the sandbox ring, evaluated along the schematic L(t), T(t) track from the "
                      "Evolution tab. The track is illustrative; where its T_eff leaves the fit's validity range the band "
                      "uses clamped T_eff and is hatched.", micro_font, theme.text_tertiary, sx0, y, side_w)

    # ------------------------------------------------------------- properties
    def _render_properties(self, area: pygame.Rect, star: Dict[str, Any], d: sd.DerivedStarQuantities) -> None:
        theme = self.theme
        title_font, label_font, small_font, micro_font, _ = self._fonts()
        kind, prov_label = self._provenance(star)
        col_w = area.width
        y = area.top

        def header(text: str, yy: int) -> int:
            ui_theme.draw_section_label(self.screen, micro_font, text, (area.left, yy), theme=theme)
            return yy + 20

        def row(yy: int, label: str, value: str, source: str, tag: Optional[str] = None, flag: Optional[Color] = None) -> int:
            self.screen.blit(small_font.render(label, True, theme.text_secondary), (area.left, yy))
            vs = small_font.render(value, True, flag or theme.text_primary)
            self.screen.blit(vs, vs.get_rect(topright=(area.left + 430, yy)))
            max_src_w = area.right - (area.left + 446) - (micro_font.size(tag)[0] + 24 if tag else 0)
            src = source
            while micro_font.size(src)[0] > max_src_w and len(src) > 6:
                src = src[:-4].rstrip(" ·") + "…"
            self.screen.blit(micro_font.render(src, True, theme.text_tertiary), (area.left + 446, yy + 2))
            if tag:
                ui_theme.draw_tag(self.screen, micro_font, tag, (area.left + 446 + micro_font.size(src)[0] + 8, yy), kind=kind if tag == prov_label else "assumed", theme=theme)
            return yy + 19

        y = header("STATED PARAMETERS", y)
        y = row(y, "Mass", f"{_fmt(d.mass_solar)} Msun", "customization panel", prov_label)
        y = row(y, "Radius", f"{_fmt(d.radius_solar)} Rsun", "customization panel", prov_label)
        y = row(y, "Effective temperature", f"{d.t_eff_k:,.0f} K", "customization panel", prov_label)
        y = row(y, "Luminosity", f"{_fmt(d.luminosity_solar)} Lsun", "customization panel", prov_label)
        y = row(y, "Age", f"{d.age_gyr:.2f} Gyr", "customization panel", "ASSUMED" if str(star.get("name")) != "Sun" else prov_label)
        if d.metallicity_feh is not None:
            y = row(y, "Metallicity [Fe/H]", f"{d.metallicity_feh:+.2f} dex", "customization panel", prov_label)
        y = row(y, "Spectral class (label)", str(star.get("spectral_class") or "—"), "preset label", prov_label)
        y += 6
        ui_theme.divider(self.screen, area.left, y, col_w, theme); y += 10

        y = header("DERIVED (FROM STATED VALUES)", y)
        l_delta = (d.luminosity_sb_solar / d.luminosity_solar - 1.0) * 100.0
        consistent = abs(l_delta) <= 15.0
        y = row(y, "Luminosity from R and T", f"{_fmt(d.luminosity_sb_solar)} Lsun  ({l_delta:+.0f}% vs stated)",
                "L = R² (T/5778 K)⁴ · Stefan–Boltzmann", None, theme.success_soft if consistent else theme.warning_soft)
        y = row(y, "Radius from L and T", f"{_fmt(d.radius_sb_solar)} Rsun", "R = √L / (T/5778 K)²")
        y = row(y, "Surface gravity", f"log g = {d.log_g_cgs:.2f} (cgs)", "log g_sun + log(M / R²), g_sun = 27,400 cm s⁻²")
        y = row(y, "Mean density", f"{_fmt(d.mean_density_g_cm3)} g cm⁻³", "ρ_sun · M / R³, ρ_sun = 1.408 g cm⁻³")
        y = row(y, "Escape velocity", f"{d.escape_velocity_km_s:,.0f} km s⁻¹", "v_sun √(M / R), v_sun = 617.7 km s⁻¹")
        y = row(y, "Absolute bolometric magnitude", f"{d.abs_bol_magnitude:+.2f}", "M_bol = 4.74 − 2.5 log L")
        y = row(y, "Peak wavelength (Wien)", f"{d.peak_wavelength_nm:,.0f} nm", "λmax = 2.898×10⁶ nm·K / T")
        cls = sd.spectral_class_from_temperature(d.t_eff_k)
        label_letter = (str(star.get("spectral_class") or "")[:1]).upper()
        match = (not label_letter) or label_letter == cls.letter
        y = row(y, "Spectral class from T_eff", cls.letter + ("" if match else f"  (label says {label_letter})"),
                f"Harvard temperature cuts · {cls.signature}", None, theme.text_primary if match else theme.warning_soft)
        y = row(y, "Main-sequence lifetime", f"{d.ms_lifetime_gyr:,.1f} Gyr" if d.ms_lifetime_gyr < 1e4 else f"{d.ms_lifetime_gyr:.2e} Gyr",
                "τ ≈ 10 Gyr · (M/Msun)^-2.5", None, theme.warning_soft if d.age_gyr > d.ms_lifetime_gyr else None)
        if d.hz_inner_au is not None and d.hz_outer_au is not None:
            y = row(y, "Habitable zone (conservative)", f"{d.hz_inner_au:.3f} – {d.hz_outer_au:.3f} AU",
                    "Kopparapu et al. 2013 · recent Venus / early Mars limits")
        y += 6
        ui_theme.divider(self.screen, area.left, y, col_w, theme); y += 10

        # Compare to Sun bars
        y = header("RELATIVE TO THE SUN (log scale, Sun = 1)", y)
        bars = [
            ("Mass", d.mass_solar, 0.05, 100.0),
            ("Radius", d.radius_solar, 0.05, 100.0),
            ("T_eff", d.t_eff_k / sd.T_SUN_K, 0.3, 10.0),
            ("Luminosity", d.luminosity_solar, 1e-4, 1e6),
        ]
        bar_x = area.left + 90
        bar_w = min(460, col_w - 200)
        for label, val, lo, hi in bars:
            self.screen.blit(small_font.render(label, True, theme.text_secondary), (area.left, y))
            track = pygame.Rect(bar_x, y + 4, bar_w, 10)
            pygame.draw.rect(self.screen, theme.field_bg, track, border_radius=5)
            pygame.draw.rect(self.screen, theme.panel_border, track, 1, border_radius=5)
            f = (math.log10(max(lo, min(hi, val))) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
            f_sun = (0.0 - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
            x_val = track.left + int(f * track.width)
            x_sun = track.left + int(f_sun * track.width)
            fill_l, fill_r = sorted((x_val, x_sun))
            pygame.draw.rect(self.screen, _mix(theme.accent, theme.field_bg, 0.2), pygame.Rect(fill_l, track.top + 1, max(2, fill_r - fill_l), track.height - 2), border_radius=4)
            pygame.draw.line(self.screen, theme.chart_reference, (x_sun, track.top - 3), (x_sun, track.bottom + 3), 1)
            pygame.draw.circle(self.screen, d.color_rgb, (x_val, track.centery), 5)
            vs = small_font.render(f"{_fmt(val)}×", True, theme.text_primary)
            self.screen.blit(vs, (track.right + 10, y))
            y += 24
        pygame.draw.line(self.screen, theme.chart_reference, (area.left, y + 7), (area.left, y + 13), 1)
        self.screen.blit(micro_font.render("tick marks the Sun's value on each bar", True, theme.text_tertiary), (area.left + 8, y + 2))
