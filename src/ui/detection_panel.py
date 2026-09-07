"""
Detection panel: how would astronomers actually find the selected planet?

Three instrument views for the selected planet and its host star:
  * Radial velocity — the star's reflex wobble, its Keplerian RV curve and the
    (exaggerated-for-display) Doppler shift of absorption lines. Ported from the
    Hubble Doppler Sonifier's line-shift + sonification ideas.
  * Transit — orbit projected at a chosen inclination (Orbital Inclination widget
    geometry, astronomical convention: 90 deg = edge-on), close-up of the disc
    crossing and the resulting light curve.
  * Compare — every planet of the host star side by side across RV, transit,
    astrometry and reflected-light contrast, with order-of-magnitude thresholds.

Presentation-only. Reads the sandbox dicts every frame so edits in the
customization panel update the charts live. All numbers come from
src/science/detection_methods.py; nothing here feeds back into the simulation.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pygame

from src.science import detection_methods as dm
from src.science import stellar_data as sd
from src.ui import theme as ui_theme
from src.ui.star_data_panel import _dashed_line, _dashed_polyline, _fmt, _mix, _wavelength_to_rgb
from src.ui.theme import THEME

try:  # numpy is already an AIET dependency; only the sonification needs it here
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


TABS: Sequence[Tuple[str, str]] = (
    ("rv", "Radial velocity"),
    ("transit", "Transit"),
    ("compare", "Compare methods"),
)

Color = Tuple[int, int, int]
ANIM_PERIOD_S = 12.0        # one displayed orbit every 12 s regardless of the real period
SOUND_SECONDS = 4.0


def _parse_color(value: Any, fallback: Color) -> Color:
    try:
        if isinstance(value, (tuple, list)) and len(value) >= 3:
            return (int(value[0]), int(value[1]), int(value[2]))
        if isinstance(value, str) and value:
            c = pygame.Color(value)
            return (c.r, c.g, c.b)
    except Exception:
        pass
    return fallback


class DetectionPanel:
    """Tabbed instrument panel for the selected planet. Non-modal, like StarDataPanel: it sits
    over the sandbox viewport and leaves the customization panel interactive."""

    def __init__(self, visualizer: Any):
        self.viz = visualizer
        self.visible = False
        self.tab = "rv"
        self.planet_id: Optional[str] = None
        self.panel_rect: Optional[pygame.Rect] = None
        self.close_rect: Optional[pygame.Rect] = None
        self.save_rect: Optional[pygame.Rect] = None
        self.tab_rects: Dict[str, pygame.Rect] = {}
        self.slider_rect: Optional[pygame.Rect] = None
        self.slider_dragging = False
        self.anim_rect: Optional[pygame.Rect] = None
        self.sound_rect: Optional[pygame.Rect] = None
        self.edge_on_rect: Optional[pygame.Rect] = None
        self.inclination_deg = 90.0
        self.animate = True
        self._anim_t0 = time.time()
        self._anim_frozen_phase: Optional[float] = None
        self._opened_at = 0.0
        self._sound: Optional[pygame.mixer.Sound] = None
        self._sound_note = ""
        self._sound_until = 0.0

    # ------------------------------------------------------------------ state
    @property
    def screen(self) -> pygame.Surface:
        return self.viz.screen

    @property
    def theme(self):
        return getattr(self.viz, "theme", THEME)

    def open(self, planet: Optional[Dict[str, Any]], tab: Optional[str] = None) -> None:
        if not planet or planet.get("type") != "planet":
            return
        if tab:
            self.tab = tab
        if planet.get("id") != self.planet_id:
            self._anim_t0 = time.time()
        self.planet_id = planet.get("id")
        if not self.visible:
            self._opened_at = time.time()
        self.visible = True

    def close(self) -> None:
        self.visible = False
        self.slider_dragging = False

    def _current_planet(self) -> Optional[Dict[str, Any]]:
        body = getattr(self.viz, "selected_body", None)
        if body and body.get("type") == "planet" and not body.get("is_destroyed"):
            return body
        return None

    def _host_star(self, planet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        getter = getattr(self.viz, "_get_parent_star", None)
        star = None
        if callable(getter):
            try:
                star = getter(planet)
            except Exception:
                star = None
        if star is None:
            stars = [b for b in getattr(self.viz, "placed_bodies", []) if b.get("type") == "star" and not b.get("is_destroyed")]
            star = stars[0] if stars else None
        return star

    def _siblings(self, star: Dict[str, Any]) -> List[Dict[str, Any]]:
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

    def _phase(self) -> float:
        """Displayed orbital phase (mean anomaly / 2pi), 0 = periastron = mid-transit."""
        if not self.animate and self._anim_frozen_phase is not None:
            return self._anim_frozen_phase
        return ((time.time() - self._anim_t0) / ANIM_PERIOD_S) % 1.0

    def _toggle_animation(self) -> None:
        if self.animate:
            self._anim_frozen_phase = self._phase()
            self.animate = False
        else:
            # resume from the frozen phase
            frozen = self._anim_frozen_phase or 0.0
            self._anim_t0 = time.time() - frozen * ANIM_PERIOD_S
            self._anim_frozen_phase = None
            self.animate = True

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

    def _text_input_active(self) -> bool:
        v = self.viz
        if any(getattr(v, f, False) for f in ("rename_edit_active", "show_rename_edit", "custom_modal_visible", "show_custom_modal")):
            return True
        return any(name.endswith("_input_active") and bool(val) for name, val in vars(v).items())

    def _set_inclination_from_x(self, x: float) -> None:
        if not self.slider_rect:
            return
        f = (x - self.slider_rect.left) / max(1, self.slider_rect.width)
        self.inclination_deg = round(max(0.0, min(1.0, f)) * 90.0 * 2) / 2.0

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Returns True if the event was consumed."""
        if self._input_blocked():
            self.slider_dragging = False
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            launcher = getattr(self.viz, "detection_tab_button_rects", None) or {}
            planet = self._current_planet()
            if planet and getattr(self.viz, "show_customization_panel", False):
                for tab_id, rect in launcher.items():
                    if rect and rect.collidepoint(pos):
                        if self.visible and self.tab == tab_id:
                            self.close()
                        else:
                            self.open(planet, tab_id)
                        return True
            if not self.visible or not self.panel_rect:
                return False
            if self.close_rect and self.close_rect.collidepoint(pos):
                self.close()
                return True
            if self.save_rect and self.save_rect.collidepoint(pos):
                self._save_png()
                return True
            if self.slider_rect and self.slider_rect.inflate(6, 14).collidepoint(pos):
                self.slider_dragging = True
                self._set_inclination_from_x(pos[0])
                return True
            if self.edge_on_rect and self.edge_on_rect.collidepoint(pos):
                self.inclination_deg = 90.0
                return True
            if self.anim_rect and self.anim_rect.collidepoint(pos):
                self._toggle_animation()
                return True
            if self.sound_rect and self.sound_rect.collidepoint(pos):
                self._play_sound()
                return True
            for tab_id, rect in self.tab_rects.items():
                if rect.collidepoint(pos):
                    self.tab = tab_id
                    return True
            if self.panel_rect.collidepoint(pos):
                return True
            return False
        if event.type == pygame.MOUSEMOTION and self.slider_dragging:
            self._set_inclination_from_x(event.pos[0])
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.slider_dragging:
            self.slider_dragging = False
            return True
        if self.visible and self.panel_rect and event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEWHEEL):
            pos = getattr(event, "pos", None) or self.viz._mouse_pos()
            if self.panel_rect.collidepoint(pos):
                return True
        if self.visible and event.type == pygame.KEYDOWN and not self._text_input_active():
            hotkeys = (pygame.K_1, pygame.K_2, pygame.K_3)
            if event.key in hotkeys:
                self.tab = TABS[hotkeys.index(event.key)][0]
                return True
            if event.key == pygame.K_SPACE and self.panel_rect and self.panel_rect.collidepoint(self.viz._mouse_pos()):
                self._toggle_animation()
                return True
        return False

    def _save_png(self) -> None:
        if not self.panel_rect:
            return
        try:
            out_dir = self.viz.create_export_directory()
        except Exception:
            out_dir = "exports"
            os.makedirs(out_dir, exist_ok=True)
        planet = self._current_planet() or {}
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(planet.get("name", "planet")))
        path = os.path.join(out_dir, f"detection_{safe}_{self.tab}_{time.strftime('%Y%m%d_%H%M%S')}.png")
        try:
            sub = self.screen.subsurface(self.panel_rect.clip(self.screen.get_rect()))
            pygame.image.save(sub.copy(), path)
            if hasattr(self.viz, "_show_export_toast"):
                self.viz._show_export_toast(path)
        except Exception as exc:  # pragma: no cover
            print(f"[Detection] PNG export failed: {exc}")

    # ----------------------------------------------------------------- audio
    def _ensure_mixer(self) -> bool:
        if np is None:
            return False
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
            return bool(pygame.mixer.get_init())
        except Exception as exc:
            print(f"[Detection] audio unavailable: {exc}")
            return False

    def _make_sound(self, samples) -> Optional[pygame.mixer.Sound]:
        """samples: float array in [-1, 1] → pygame Sound matching the mixer format."""
        try:
            init = pygame.mixer.get_init()
            if not init:
                return None
            _, _, channels = init
            pcm = (np.clip(samples, -1.0, 1.0) * 32000.0).astype(np.int16)
            if channels >= 2:
                pcm = np.repeat(pcm[:, None], channels, axis=1)
            return pygame.sndarray.make_sound(np.ascontiguousarray(pcm))
        except Exception as exc:
            print(f"[Detection] could not build sound: {exc}")
            return None

    def _play_sound(self) -> None:
        planet = self._current_planet()
        star = self._host_star(planet) if planet else None
        if not planet or not star or not self._ensure_mixer():
            self._sound_note = "audio unavailable"
            self._sound_until = time.time() + 3
            return
        s = dm.summarize_planet(planet, star, self.inclination_deg)
        sr = pygame.mixer.get_init()[0]
        n = int(SOUND_SECONDS * sr)
        t = np.arange(n) / sr
        base_hz = 440.0
        if self.tab == "transit":
            # Amplitude follows the light curve; the dip is scaled so it is clearly audible.
            pts, half_w = dm.transit_light_curve_hours(s.period_days, s.a_au, float(star.get("radius", 1.0) or 1.0),
                                                       s.radius_earth, s.inc_deg, s.eccentricity, n=400)
            depth = max(1e-12, 1.0 - min(f for _, f in pts))
            xs = np.array([p[0] for p in pts]) / half_w  # -1..1
            fl = np.array([p[1] for p in pts])
            u = np.interp(np.linspace(-1, 1, n), xs, fl)
            amp = 1.0 - (1.0 - u) / depth * 0.7 if s.transits else np.ones(n)
            env = np.minimum(1.0, np.minimum(t / 0.05, (SOUND_SECONDS - t) / 0.15))
            samples = 0.35 * amp * env * np.sin(2 * np.pi * base_hz * t)
            self._sound_note = ("dip exaggerated to −70% loudness" if s.transits else "no transit at this inclination: steady tone")
        else:
            # Pitch follows v_r over one orbit; K is scaled to ±2 semitones (real shift ≈ K/c is inaudible).
            if s.k_m_s <= 0:
                self._sound_note = "no wobble (planet mass = 0)"
                self._sound_until = time.time() + 3
                return
            phases = np.linspace(0, 1, 400)
            v = np.array([dm.rv_at_mean_anomaly(2 * math.pi * p, s.k_m_s, s.eccentricity) for p in phases])
            v_t = np.interp((t / SOUND_SECONDS) % 1.0, phases, v)
            gain = (2 ** (2 / 12) - 1) / max(1e-9, s.k_m_s)
            freq = base_hz * (1.0 + gain * v_t)
            phase = np.cumsum(2 * np.pi * freq / sr)
            env = np.minimum(1.0, np.minimum(t / 0.05, (SOUND_SECONDS - t) / 0.15))
            samples = 0.35 * env * np.sin(phase)
            self._sound_note = f"pitch exaggerated ×{gain * dm.C_M_S:.1e} (K → ±2 semitones)"
        snd = self._make_sound(samples)
        if snd is None:
            self._sound_note = "audio unavailable"
        else:
            try:
                snd.play()
                self._sound = snd
            except Exception as exc:
                self._sound_note = f"audio error: {exc}"
        self._sound_until = time.time() + SOUND_SECONDS + 1.0

    # ----------------------------------------------------------------- render
    def _fonts(self):
        v = self.viz
        return getattr(v, "headline_font", v.subtitle_font), v.tab_font, v.tiny_font, v.micro_font, v.font

    def render(self) -> None:
        if not self.visible:
            return
        planet = self._current_planet()
        if planet is None or not getattr(self.viz, "show_customization_panel", False):
            self.visible = False
            return
        star = self._host_star(planet)
        self.planet_id = planet.get("id")
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
        p_color = _parse_color(planet.get("base_color"), theme.chart_planet)
        swatch_c = (rect.left + pad + 9, rect.top + pad + 9)
        pygame.draw.circle(self.screen, p_color, swatch_c, 9)
        pygame.draw.circle(self.screen, theme.panel_border, swatch_c, 9, 1)
        close_size = 22
        self.close_rect = pygame.Rect(rect.right - close_size - 12, rect.top + 12, close_size, close_size)
        ui_theme.draw_close_x(self.screen, self.close_rect, hover=self.close_rect.collidepoint(mouse), theme=theme)
        self.save_rect = pygame.Rect(self.close_rect.left - 10 - 92, rect.top + 11, 92, 26)
        ui_theme.draw_button(self.screen, self.save_rect, "Save PNG", small_font, kind="secondary",
                             hover=self.save_rect.collidepoint(mouse), theme=theme)

        p_name = str(planet.get("display_name") or planet.get("name") or "Planet")
        s_name = str(star.get("name", "star")) if star else "no star"
        inc_tag = f"i = {self.inclination_deg:.1f}°" + (" · EDGE-ON" if self.inclination_deg >= 89.5 else "")
        geo_tag = "OBSERVER GEOMETRY ASSUMED"
        avail = self.save_rect.left - 12 - (swatch_c[0] + 18)
        tags_w = micro_font.size(inc_tag)[0] + 16 + 6 + micro_font.size(geo_tag)[0] + 16 + 10
        title_txt = f"Detection · {p_name} around {s_name}"
        if title_font.size(title_txt)[0] + tags_w > avail:
            title_txt = f"Detection · {p_name}"
        show_geo = title_font.size(title_txt)[0] + tags_w <= avail
        while title_font.size(title_txt)[0] + (tags_w if show_geo else micro_font.size(inc_tag)[0] + 26) > avail and len(title_txt) > 14:
            title_txt = title_txt[:-2].rstrip() + "…"
        title = title_font.render(title_txt, True, theme.text_primary)
        title_rect = title.get_rect(midleft=(swatch_c[0] + 18, swatch_c[1]))
        self.screen.blit(title, title_rect)
        tag_rect = ui_theme.draw_tag(self.screen, micro_font, inc_tag, (title_rect.right + 10, title_rect.centery - 8),
                                     kind="assumed", theme=theme)
        if show_geo:
            ui_theme.draw_tag(self.screen, micro_font, geo_tag, (tag_rect.right + 6, title_rect.centery - 8),
                              kind="assumed", theme=theme)

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

        # Shared controls row: inclination slider, edge-on preset, animation toggle, sound -----
        ctrl_y = tab_y + tab_h + 12
        ctrl_h = 26
        ctrl_left = rect.left + pad
        ctrl_right = rect.right - pad
        self.sound_rect = pygame.Rect(ctrl_right - 104, ctrl_y, 104, ctrl_h)
        ui_theme.draw_button(self.screen, self.sound_rect, "Play sound", small_font, kind="ghost",
                             hover=self.sound_rect.collidepoint(mouse), theme=theme)
        self.anim_rect = pygame.Rect(self.sound_rect.left - 6 - 84, ctrl_y, 84, ctrl_h)
        ui_theme.draw_button(self.screen, self.anim_rect, "Pause" if self.animate else "Animate", small_font, kind="ghost",
                             hover=self.anim_rect.collidepoint(mouse), theme=theme)
        self.edge_on_rect = pygame.Rect(self.anim_rect.left - 6 - 76, ctrl_y, 76, ctrl_h)
        ui_theme.draw_button(self.screen, self.edge_on_rect, "Edge-on", small_font, kind="ghost",
                             hover=self.edge_on_rect.collidepoint(mouse), active=self.inclination_deg >= 89.5, theme=theme)
        lab = micro_font.render("ORBITAL INCLINATION", True, theme.text_tertiary)
        self.screen.blit(lab, lab.get_rect(midleft=(ctrl_left, ctrl_y + 7)))
        val = small_font.render(f"i = {self.inclination_deg:.1f}°", True, theme.text_primary)
        self.screen.blit(val, val.get_rect(midleft=(ctrl_left, ctrl_y + 19)))
        s_left = ctrl_left + max(lab.get_width(), val.get_width()) + 14
        lo_lab = micro_font.render("0° face-on", True, theme.text_tertiary)
        hi_lab = micro_font.render("90° edge-on", True, theme.text_tertiary)
        self.screen.blit(lo_lab, lo_lab.get_rect(midleft=(s_left, ctrl_y + ctrl_h // 2)))
        s_left += lo_lab.get_width() + 8
        s_right = self.edge_on_rect.left - 14 - hi_lab.get_width() - 8
        self.screen.blit(hi_lab, hi_lab.get_rect(midleft=(s_right + 8, ctrl_y + ctrl_h // 2)))
        self.slider_rect = pygame.Rect(s_left, ctrl_y + ctrl_h // 2 - 3, max(60, s_right - s_left), 6)
        self._draw_slider(self.slider_rect, self.inclination_deg / 90.0, mouse)
        if time.time() < self._sound_until and self._sound_note:
            note = micro_font.render(self._sound_note, True, theme.accent_soft)
            self.screen.blit(note, note.get_rect(topright=(ctrl_right, ctrl_y + ctrl_h + 2)))

        ui_theme.divider(self.screen, rect.left + pad, ctrl_y + ctrl_h + 14, rect.width - 2 * pad, theme)
        content_top = ctrl_y + ctrl_h + 24
        content = pygame.Rect(rect.left + pad, content_top, rect.width - 2 * pad, rect.bottom - pad - content_top)
        prev_clip = self.screen.get_clip()
        self.screen.set_clip(content.inflate(4, 4))
        try:
            if star is None:
                self._wrapped("This planet has no host star in the sandbox, so no detection signal can be computed. "
                              "Add a star and place the planet in orbit around it.", small_font, theme.warning_soft,
                              content.left, content.top, content.width)
                return
            summary = dm.summarize_planet(planet, star, self.inclination_deg)
            if self.tab == "rv":
                self._render_rv(content, planet, star, summary)
            elif self.tab == "transit":
                self._render_transit(content, planet, star, summary)
            else:
                self._render_compare(content, planet, star, summary)
        finally:
            self.screen.set_clip(prev_clip)

    # --------------------------------------------------------------- helpers
    def _draw_slider(self, track: pygame.Rect, f: float, mouse: Tuple[int, int]) -> None:
        theme = self.theme
        pygame.draw.rect(self.screen, theme.field_bg, track, border_radius=3)
        pygame.draw.rect(self.screen, theme.panel_border, track, 1, border_radius=3)
        fill = pygame.Rect(track.left, track.top, int(track.width * f), track.height)
        if fill.width > 0:
            pygame.draw.rect(self.screen, _mix(theme.accent, theme.field_bg, 0.25), fill, border_radius=3)
        kx = track.left + int(track.width * f)
        hover = self.slider_dragging or track.inflate(6, 14).collidepoint(mouse)
        pygame.draw.circle(self.screen, theme.text_primary, (kx, track.centery), 8 if hover else 7)
        pygame.draw.circle(self.screen, theme.accent if hover else theme.panel_border, (kx, track.centery), 8 if hover else 7, 2)

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

    def _tone_color(self, tone: str) -> Color:
        theme = self.theme
        return {"good": theme.success_soft, "ok": theme.warning_soft}.get(tone, theme.danger_soft)

    def _log_scale_bar(self, x: int, y: int, w: int, lo: float, hi: float, value: float, value_label: str,
                       thresholds: Sequence[Tuple[float, str]], references: Sequence[Tuple[str, float]],
                       fmt) -> int:
        """Horizontal log-scale comparison bar. Returns the y below the drawn block."""
        theme = self.theme
        _, _, small_font, micro_font, _ = self._fonts()
        llo, lhi = math.log10(lo), math.log10(hi)

        def xp(val: float) -> int:
            val = max(lo, min(hi, val))
            return x + int((math.log10(val) - llo) / (lhi - llo) * w)

        track = pygame.Rect(x, y + 58, w, 8)
        pygame.draw.rect(self.screen, theme.field_bg, track, border_radius=4)
        pygame.draw.rect(self.screen, theme.panel_border, track, 1, border_radius=4)
        # Decade ticks
        d = int(math.ceil(llo))
        while d <= math.floor(lhi):
            tx = xp(10 ** d)
            pygame.draw.line(self.screen, theme.chart_axis, (tx, track.bottom + 1), (tx, track.bottom + 5), 1)
            tl = micro_font.render(fmt(10 ** d), True, theme.chart_axis_text)
            self.screen.blit(tl, tl.get_rect(midtop=(tx, track.bottom + 6)))
            d += 1
        # Thresholds (dashed) with labels on two staggered rows above the track
        for i, (thr, name) in enumerate(thresholds):
            tx = xp(thr)
            _dashed_line(self.screen, theme.chart_reference, (tx, track.top - 4), (tx, track.bottom + 4), 3, 3)
            tl = micro_font.render(name, True, theme.text_tertiary)
            row = i % 2
            r = tl.get_rect(midbottom=(tx, track.top - 6 - row * 12))
            r.left = max(x, min(r.left, x + w - r.width))
            self.screen.blit(tl, r)
        # References below the decade labels
        for i, (name, val) in enumerate(references):
            rx = xp(val)
            pygame.draw.circle(self.screen, theme.chart_reference, (rx, track.centery), 3)
            tl = micro_font.render(name, True, theme.text_secondary)
            r = tl.get_rect(midtop=(rx, track.bottom + 20 + (i % 2) * 12))
            r.left = max(x, min(r.left, x + w - r.width))
            self.screen.blit(tl, r)
        # Value: label on its own row at the top, connector down to the marker
        vx = xp(value)
        clipped = value < lo or value > hi
        pygame.draw.circle(self.screen, theme.accent, (vx, track.centery), 7)
        pygame.draw.circle(self.screen, theme.text_primary, (vx, track.centery), 7, 2)
        vl = small_font.render(value_label + (" (off scale)" if clipped else ""), True, theme.text_primary)
        r = vl.get_rect(topleft=(x, y + 14))
        r.centerx = vx
        r.left = max(x, min(r.left, x + w - r.width))
        self.screen.blit(vl, r)
        pygame.draw.line(self.screen, theme.accent, (vx, r.bottom + 1), (vx, track.top - 1), 1)
        return track.bottom + 46

    # ------------------------------------------------------------- RV tab
    def _render_rv(self, area: pygame.Rect, planet: Dict[str, Any], star: Dict[str, Any], s: dm.DetectionSummary) -> None:
        theme = self.theme
        title_font, label_font, small_font, micro_font, _ = self._fonts()
        side_w = 280
        chart_w = area.width - side_w - 12
        phase_now = self._phase()
        v_now = dm.rv_at_mean_anomaly(2 * math.pi * phase_now, s.k_m_s, s.eccentricity)
        star_col = sd.temperature_to_rgb(float(star.get("temperature", 5778.0) or 5778.0))

        # RV curve ------------------------------------------------------------
        curve_h = int(area.height * 0.50)
        chart = pygame.Rect(area.left, area.top, chart_w, curve_h)
        self._chart_frame(chart)
        plot = pygame.Rect(chart.left + 56, chart.top + 26, chart.width - 72, chart.height - 58)
        k_axis = max(s.k_m_s * 1.25, 1e-6)
        if s.eccentricity > 0:
            k_axis = max(k_axis, abs(s.k_m_s * (1 + s.eccentricity)) * 1.1)

        def xp(ph: float) -> float:
            return plot.left + ph * plot.width

        def yp(vv: float) -> float:
            return plot.centery - (vv / k_axis) * (plot.height / 2)

        hdr = micro_font.render("STELLAR RADIAL VELOCITY OVER ONE ORBIT  ·  positive = receding (redshift)", True, theme.text_tertiary)
        self.screen.blit(hdr, (chart.left + 10, chart.top + 8))
        for f in (0.25, 0.5, 0.75):
            gx = xp(f)
            pygame.draw.line(self.screen, theme.chart_grid, (gx, plot.top), (gx, plot.bottom), 1)
        for frac in (-1.0, -0.5, 0.5, 1.0):
            gy = yp(frac * k_axis)
            pygame.draw.line(self.screen, theme.chart_grid, (plot.left, gy), (plot.right, gy), 1)
            tl = micro_font.render(dm.format_velocity(frac * k_axis) if s.k_m_s > 0 else "0", True, theme.chart_axis_text)
            self.screen.blit(tl, tl.get_rect(midright=(plot.left - 6, gy)))
        _dashed_line(self.screen, theme.chart_axis, (plot.left, plot.centery), (plot.right, plot.centery), 4, 4)
        zl = micro_font.render("0", True, theme.chart_axis_text)
        self.screen.blit(zl, zl.get_rect(midright=(plot.left - 6, plot.centery)))
        pygame.draw.line(self.screen, theme.chart_axis, (plot.left, plot.top), (plot.left, plot.bottom), 1)
        for f, txt in ((0.0, "0"), (0.5, "0.5"), (1.0, "1 orbit")):
            tl = micro_font.render(txt, True, theme.chart_axis_text)
            tr = tl.get_rect(midtop=(xp(f), plot.bottom + 4))
            tr.right = min(tr.right, chart.right - 4)
            self.screen.blit(tl, tr)
        ax = micro_font.render(f"Orbital phase (period {s.period_days:,.1f} d)", True, theme.text_tertiary)
        self.screen.blit(ax, ax.get_rect(midtop=(plot.centerx, plot.bottom + 16)))
        ay = pygame.transform.rotate(micro_font.render("v_r (m/s)", True, theme.text_tertiary), 90)
        self.screen.blit(ay, ay.get_rect(center=(chart.left + 12, plot.centery)))

        if s.k_m_s > 0:
            pts = dm.rv_curve(s.k_m_s, s.eccentricity, 90.0, 240)
            red = [(xp(p), yp(max(0.0, vv))) for p, vv in pts]
            blue = [(xp(p), yp(min(0.0, vv))) for p, vv in pts]
            fill = pygame.Surface((plot.width + 1, plot.height + 1), pygame.SRCALPHA)
            pygame.draw.polygon(fill, (*theme.chart_redshift, 40), [(px - plot.left, py - plot.top) for px, py in ([(plot.left, plot.centery)] + red + [(plot.right, plot.centery)])])
            pygame.draw.polygon(fill, (*theme.chart_blueshift, 40), [(px - plot.left, py - plot.top) for px, py in ([(plot.left, plot.centery)] + blue + [(plot.right, plot.centery)])])
            self.screen.blit(fill, plot.topleft)
            line = [(xp(p), yp(vv)) for p, vv in pts]
            # color the polyline by sign
            for (x0, y0), (x1, y1) in zip(line, line[1:]):
                col = theme.chart_redshift if (y0 + y1) / 2 < plot.centery else theme.chart_blueshift
                pygame.draw.line(self.screen, col, (x0, y0), (x1, y1), 2)
            # K annotation
            _dashed_line(self.screen, theme.chart_reference, (plot.left, yp(s.k_m_s)), (plot.right, yp(s.k_m_s)), 3, 4)
            kl = micro_font.render(f"K = {dm.format_velocity(s.k_m_s)}", True, theme.text_secondary)
            self.screen.blit(kl, kl.get_rect(bottomright=(plot.right - 4, yp(s.k_m_s) - 2)))
            # current phase marker
            mx, my = xp(phase_now), yp(v_now)
            pygame.draw.line(self.screen, theme.text_primary, (mx, plot.top), (mx, plot.bottom), 1)
            pygame.draw.circle(self.screen, theme.text_primary, (int(mx), int(my)), 5)
            pygame.draw.circle(self.screen, star_col, (int(mx), int(my)), 3)
        else:
            self._wrapped("Planet mass is zero — no reflex motion.", small_font, theme.text_secondary, plot.left, plot.centery - 8, plot.width)

        # Doppler strip -------------------------------------------------------
        strip_top = chart.bottom + 12
        strip_box = pygame.Rect(area.left, strip_top, chart_w, area.bottom - strip_top)
        self._chart_frame(strip_box)
        hdr = micro_font.render("ABSORPTION LINES · rest (grey) vs. shifted by the star's current v_r", True, theme.text_tertiary)
        self.screen.blit(hdr, (strip_box.left + 10, strip_box.top + 8))
        rainbow = pygame.Rect(strip_box.left + 16, strip_box.top + 46, strip_box.width - 32, 30)
        lam0, lam1 = 380.0, 750.0

        def lx(lam: float) -> float:
            return rainbow.left + (lam - lam0) / (lam1 - lam0) * rainbow.width

        for px in range(rainbow.left, rainbow.right + 1):
            lam = lam0 + (lam1 - lam0) * (px - rainbow.left) / max(1, rainbow.width)
            pygame.draw.line(self.screen, _wavelength_to_rgb(lam), (px, rainbow.top), (px, rainbow.bottom))
        pygame.draw.rect(self.screen, theme.panel_border, rainbow, 1)
        # Exaggeration: K maps to 14 px so the motion is visible; report the factor honestly.
        k_shift_nm = dm.doppler_shift_nm(dm.H_ALPHA_NM, max(s.k_m_s, 1e-9))
        px_per_nm = rainbow.width / (lam1 - lam0)
        exag = (14.0 / px_per_nm) / max(k_shift_nm, 1e-12) if s.k_m_s > 0 else 1.0
        for lam, name in dm.DOPPLER_LINES:
            rx = lx(lam)
            pygame.draw.line(self.screen, _mix(theme.field_bg, theme.text_primary, 0.35), (rx, rainbow.top + 1), (rx, rainbow.bottom - 1), 1)
            shift = dm.doppler_shift_nm(lam, v_now) * exag
            sx = max(rainbow.left, min(rainbow.right, lx(lam + shift)))
            col = theme.chart_redshift if v_now > 0 else theme.chart_blueshift
            pygame.draw.line(self.screen, col, (sx, rainbow.top - 6), (sx, rainbow.bottom + 6), 2)
            nl = micro_font.render(name, True, theme.text_secondary)
            self.screen.blit(nl, nl.get_rect(midbottom=(rx, rainbow.top - 8)))
        for lam in (400, 450, 500, 550, 600, 650, 700, 750):
            tl = micro_font.render(str(lam), True, theme.chart_axis_text)
            self.screen.blit(tl, tl.get_rect(midtop=(lx(lam), rainbow.bottom + 8)))
        info_y = rainbow.bottom + 24
        dl = dm.doppler_shift_nm(dm.H_ALPHA_NM, v_now)
        line1 = (f"now: v_r = {dm.format_velocity(v_now)}  →  Δλ(Hα) = {dl:+.2e} nm  (z = {dm.redshift_z(v_now):+.2e}). "
                 f"Display shift exaggerated ×{exag:.1e}; the true shift is far below one pixel.")
        info_y = self._wrapped(line1, micro_font, theme.text_secondary, strip_box.left + 16, info_y, strip_box.width - 32)
        self._wrapped("Method: measure the star's spectrum repeatedly; a periodic line shift reveals an unseen companion. "
                      "Only M_p·sin i is recovered — the inclination stays unknown unless the planet also transits.",
                      micro_font, theme.text_tertiary, strip_box.left + 16, info_y + 2, strip_box.width - 32)

        # Side readouts -------------------------------------------------------
        sx0 = area.left + chart_w + 12
        y = area.top
        self._readout(sx0, y, "Semi-amplitude K at this i", dm.format_velocity(s.k_m_s), theme.accent_soft, side_w); y += 38
        self._readout(sx0, y, "K if edge-on (sin i = 1)", dm.format_velocity(s.k_edge_on_m_s), None, side_w); y += 38
        msini = s.mass_earth * math.sin(math.radians(s.inc_deg))
        self._readout(sx0, y, "Recovered M_p sin i", f"{_fmt(msini)} Mearth  (true {_fmt(s.mass_earth)} Mearth)", None, side_w); y += 38
        self._readout(sx0, y, "Period · eccentricity", f"{s.period_days:,.1f} d · e = {s.eccentricity:.3f}", None, side_w); y += 38
        m_star_kg = float(star.get("mass", 1.0) or 1.0) * dm.M_SUN_KG
        a_star_km = s.a_au * s.mass_earth * dm.M_EARTH_KG / (m_star_kg + s.mass_earth * dm.M_EARTH_KG) * dm.AU_M / 1000.0
        self._readout(sx0, y, "Star's orbit about barycentre", f"{a_star_km:,.0f} km  ({a_star_km / (float(star.get('radius', 1.0) or 1.0) * dm.R_SUN_M / 1000.0):.3g} R★)", None, side_w); y += 40
        label, tone = dm.rv_detectability(s.k_m_s)
        y = self._wrapped(label, small_font, self._tone_color(tone), sx0, y, side_w) + 6
        ui_theme.divider(self.screen, sx0, y, side_w, theme); y += 8
        ui_theme.draw_section_label(self.screen, micro_font, "K VS. SPECTROGRAPH PRECISION (LOG SCALE)", (sx0, y), theme=theme); y += 6
        y = self._log_scale_bar(sx0 + 6, y, side_w - 12, 0.01, 300.0, s.k_m_s, dm.format_velocity(s.k_m_s),
                                ((0.1, "ESPRESSO"), (1.0, "HARPS"), (3.0, "HIRES")),
                                (("Earth", 0.089), ("Jupiter", 12.5), ("51 Peg b", 56.0)),
                                lambda v: f"{v:g}")
        self._wrapped("Formula: K = (2πG/P)^⅓ · M_p sin i / (M★+M_p)^⅔ / √(1−e²). Argument of periastron assumed 90°. "
                      "Thresholds are order-of-magnitude instrument precisions.", micro_font, theme.text_tertiary, sx0, y, side_w)

    # -------------------------------------------------------------- Transit tab
    def _render_transit(self, area: pygame.Rect, planet: Dict[str, Any], star: Dict[str, Any], s: dm.DetectionSummary) -> None:
        theme = self.theme
        title_font, label_font, small_font, micro_font, _ = self._fonts()
        side_w = 280
        chart_w = area.width - side_w - 12
        phase_now = self._phase()
        r_star = max(1e-3, float(star.get("radius", 1.0) or 1.0))
        star_col = sd.temperature_to_rgb(float(star.get("temperature", 5778.0) or 5778.0))
        p_col = _parse_color(planet.get("base_color"), theme.chart_planet)
        ar = dm.a_over_rstar(s.a_au, r_star)
        k = dm.radius_ratio(s.radius_earth, r_star)
        mean_anom = 2 * math.pi * phase_now
        px_, py_, pz_, _ = dm.sky_position(mean_anom, ar, s.inc_deg, s.eccentricity)

        top_h = int(area.height * 0.52)
        orbit_box = pygame.Rect(area.left, area.top, (chart_w - 12) // 2, top_h)
        close_box = pygame.Rect(orbit_box.right + 12, area.top, chart_w - orbit_box.width - 12, top_h)
        self._chart_frame(orbit_box)
        self._chart_frame(close_box)

        # Orbit view (whole orbit fits) ---------------------------------------
        hdr = micro_font.render("ORBIT AS SEEN BY THE OBSERVER", True, theme.text_tertiary)
        self.screen.blit(hdr, (orbit_box.left + 10, orbit_box.top + 8))
        oc = (orbit_box.centerx, orbit_box.centery + 8)
        orbit_r_px = min(orbit_box.width, orbit_box.height - 30) * 0.42
        scale_o = orbit_r_px / (ar * (1 + s.eccentricity))
        star_px_o = max(3, int(scale_o))  # star radius in px (1 R* = scale_o px), min 3 px

        def to_screen_o(x: float, y: float) -> Tuple[float, float]:
            return oc[0] + x * scale_o, oc[1] - y * scale_o

        pts_front, pts_back = [], []
        n = 180
        for i in range(n + 1):
            x, y, z, _ = dm.sky_position(2 * math.pi * i / n, ar, s.inc_deg, s.eccentricity)
            (pts_front if z > 0 else pts_back).append((to_screen_o(x, y), i))
        # draw back half dashed, front half solid, keeping contiguous runs
        def runs(pts):
            out, cur, last_i = [], [], None
            for p, i in pts:
                if last_i is not None and i != last_i + 1:
                    out.append(cur); cur = []
                cur.append(p); last_i = i
            if cur:
                out.append(cur)
            return out
        for r in runs(pts_back):
            if len(r) > 1:
                _dashed_polyline(self.screen, _mix(theme.chart_ms_band, theme.field_bg, 0.2), r, 4, 4, 1)
        planet_o = to_screen_o(px_, py_)
        if pz_ <= 0:
            pygame.draw.circle(self.screen, _mix(p_col, theme.field_bg, 0.5), (int(planet_o[0]), int(planet_o[1])), 4)
        glow = pygame.Surface((star_px_o * 4 + 8, star_px_o * 4 + 8), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*star_col, 50), (glow.get_width() // 2, glow.get_height() // 2), star_px_o * 2)
        self.screen.blit(glow, (oc[0] - glow.get_width() // 2, oc[1] - glow.get_height() // 2))
        pygame.draw.circle(self.screen, star_col, oc, star_px_o)
        for r in runs(pts_front):
            if len(r) > 1:
                pygame.draw.lines(self.screen, theme.chart_ms_band, False, r, 2)
        if pz_ > 0:
            pygame.draw.circle(self.screen, p_col, (int(planet_o[0]), int(planet_o[1])), 4)
            pygame.draw.circle(self.screen, theme.text_primary, (int(planet_o[0]), int(planet_o[1])), 4, 1)
        # captions
        cap = micro_font.render(f"a / R★ = {ar:,.0f} · star {'to scale' if scale_o >= 3 else 'at min 3 px'}", True, theme.text_tertiary)
        self.screen.blit(cap, cap.get_rect(midbottom=(orbit_box.centerx, orbit_box.bottom - 6)))
        eye = micro_font.render("you are the observer (line of sight into the screen)", True, theme.text_tertiary)
        while eye.get_width() > orbit_box.width - 16:
            eye = micro_font.render("observer looks into the screen", True, theme.text_tertiary)
            break
        self.screen.blit(eye, eye.get_rect(midtop=(orbit_box.centerx, orbit_box.top + 22)))

        # Close-up during transit ----------------------------------------------
        hdr = micro_font.render("CLOSE-UP OF THE STELLAR DISC", True, theme.text_tertiary)
        self.screen.blit(hdr, (close_box.left + 10, close_box.top + 8))
        cc = (close_box.centerx, close_box.centery + 8)
        star_px_c = int(min(close_box.width, close_box.height - 40) * 0.36)

        def to_screen_c(x: float, y: float) -> Tuple[float, float]:
            return cc[0] + x * star_px_c, cc[1] - y * star_px_c

        # chord of the orbit across the disc (projected path near conjunction)
        chord = []
        for i in range(-60, 61):
            m = 2 * math.pi * (i / 60.0) * (3.0 / max(ar, 3.0))
            x, y, z, _ = dm.sky_position(m, ar, s.inc_deg, s.eccentricity)
            if z > 0 and abs(x) < 3.2:
                chord.append(to_screen_c(x, y))
        pygame.draw.circle(self.screen, star_col, cc, star_px_c)
        # limb shading hint (uniform-disc model, purely cosmetic gradient ring)
        pygame.draw.circle(self.screen, _mix(star_col, theme.field_bg, 0.35), cc, star_px_c, 2)
        if len(chord) > 1:
            _dashed_polyline(self.screen, _mix(theme.text_primary, theme.field_bg, 0.4), chord, 4, 4, 1)
        if pz_ > 0 and abs(px_) < 3.2 and abs(py_) < 3.2:
            pp = to_screen_c(px_, py_)
            pr = max(3, int(round(k * star_px_c)))
            pygame.draw.circle(self.screen, (8, 10, 14), (int(pp[0]), int(pp[1])), pr)
            pygame.draw.circle(self.screen, p_col, (int(pp[0]), int(pp[1])), pr, 1)
        # impact-parameter guide
        by = cc[1] - s.impact_b * star_px_c
        if abs(s.impact_b) < 1.6:
            _dashed_line(self.screen, theme.chart_reference, (cc[0] - star_px_c * 1.5, by), (cc[0] + star_px_c * 1.5, by), 3, 5)
            bl = micro_font.render(f"b = {abs(s.impact_b):.2f}", True, theme.text_secondary)
            self.screen.blit(bl, bl.get_rect(midleft=(cc[0] + star_px_c * 1.5 + 4, by)))
        cap = micro_font.render(f"R_p / R★ = {k:.4f} · planet {'to scale' if k * star_px_c >= 3 else 'at min 3 px'}", True, theme.text_tertiary)
        self.screen.blit(cap, cap.get_rect(midbottom=(close_box.centerx, close_box.bottom - 6)))

        # Light curve -----------------------------------------------------------
        lc_top = orbit_box.bottom + 12
        lc_box = pygame.Rect(area.left, lc_top, chart_w, area.bottom - lc_top)
        self._chart_frame(lc_box)
        hdr = micro_font.render("LIGHT CURVE AROUND MID-TRANSIT  ·  uniform stellar disc, no limb darkening", True, theme.text_tertiary)
        self.screen.blit(hdr, (lc_box.left + 10, lc_box.top + 8))
        plot = pygame.Rect(lc_box.left + 64, lc_box.top + 28, lc_box.width - 80, lc_box.height - 60)
        pts, half_w = dm.transit_light_curve_hours(s.period_days, s.a_au, r_star, s.radius_earth, s.inc_deg, s.eccentricity, n=301)
        depth_axis = max(s.depth * 1.3, 1e-7)

        def lx(t: float) -> float:
            return plot.left + (t + half_w) / (2 * half_w) * plot.width

        def ly(flux: float) -> float:
            return plot.top + 8 + (1.0 - flux) / depth_axis * (plot.height - 16)

        for f in (0.0, 0.5, 1.0):
            gy = ly(1.0 - depth_axis * f)
            pygame.draw.line(self.screen, theme.chart_grid, (plot.left, gy), (plot.right, gy), 1)
            tl = micro_font.render("1.0" if f == 0 else f"−{dm.format_depth(depth_axis * f)}", True, theme.chart_axis_text)
            self.screen.blit(tl, tl.get_rect(midright=(plot.left - 6, gy)))
        for t in (-half_w, -half_w / 2, 0.0, half_w / 2, half_w):
            gx = lx(t)
            pygame.draw.line(self.screen, theme.chart_grid, (gx, plot.top), (gx, plot.bottom), 1)
            tl = micro_font.render(f"{t:+.1f} h" if t else "0", True, theme.chart_axis_text)
            self.screen.blit(tl, tl.get_rect(midtop=(gx, plot.bottom + 4)))
        pygame.draw.line(self.screen, theme.chart_axis, (plot.left, plot.top), (plot.left, plot.bottom), 1)
        pygame.draw.line(self.screen, theme.chart_axis, (plot.left, plot.bottom), (plot.right, plot.bottom), 1)
        ay = pygame.transform.rotate(micro_font.render("Relative flux", True, theme.text_tertiary), 90)
        self.screen.blit(ay, ay.get_rect(center=(lc_box.left + 14, plot.centery)))
        line = [(lx(t), ly(f)) for t, f in pts]
        if len(line) > 1:
            pygame.draw.lines(self.screen, theme.accent_soft, False, line, 2)
        if s.transits:
            _dashed_line(self.screen, theme.chart_reference, (lx(-s.duration_hours / 2), plot.top + 4), (lx(-s.duration_hours / 2), plot.bottom), 3, 4)
            _dashed_line(self.screen, theme.chart_reference, (lx(s.duration_hours / 2), plot.top + 4), (lx(s.duration_hours / 2), plot.bottom), 3, 4)
            dl = micro_font.render(f"duration {dm.format_duration_hours(s.duration_hours)} · depth {dm.format_depth(1.0 - min(f for _, f in pts))}", True, theme.text_secondary)
            self.screen.blit(dl, dl.get_rect(midtop=(plot.centerx, plot.top + 2)))
        else:
            nl = small_font.render(f"No transit at i = {s.inc_deg:.1f}° (needs i ≥ {s.min_inc_deg:.2f}°)", True, theme.warning_soft)
            self.screen.blit(nl, nl.get_rect(center=(plot.centerx, plot.centery)))
        # current-time marker if the animated planet is inside the window
        t_now_h = ((phase_now + 0.5) % 1.0 - 0.5) * s.period_days * 24.0
        if -half_w <= t_now_h <= half_w:
            mx = lx(t_now_h)
            pygame.draw.line(self.screen, theme.text_primary, (mx, plot.top), (mx, plot.bottom), 1)

        # Side readouts -------------------------------------------------------
        sx0 = area.left + chart_w + 12
        y = area.top
        self._readout(sx0, y, "Transit depth (R_p/R★)²", dm.format_depth(s.depth) + f"  ({s.depth_ppm:,.0f} ppm)", theme.accent_soft, side_w); y += 38
        self._readout(sx0, y, "Duration at this i", dm.format_duration_hours(s.duration_hours), None, side_w); y += 38
        self._readout(sx0, y, "Impact parameter b", f"{abs(s.impact_b):.2f}  (transit if b < {1 + k:.2f})",
                      theme.success_soft if s.transits else theme.warning_soft, side_w); y += 38
        self._readout(sx0, y, "Geometric transit probability", f"{s.transit_prob * 100:.2f} %  (random orientation)", None, side_w); y += 38
        self._readout(sx0, y, "Minimum inclination to transit", f"{s.min_inc_deg:.2f}°", None, side_w); y += 40
        label, tone = dm.transit_detectability(s.depth_ppm)
        if not s.transits:
            label = f"No transit from this viewpoint. If it did transit: {label[0].lower()}{label[1:]}"
            tone = "hard"
        y = self._wrapped(label, small_font, self._tone_color(tone), sx0, y, side_w) + 6
        ui_theme.divider(self.screen, sx0, y, side_w, theme); y += 8
        ui_theme.draw_section_label(self.screen, micro_font, "DEPTH VS. PHOTOMETRIC PRECISION (LOG SCALE)", (sx0, y), theme=theme); y += 6
        y = self._log_scale_bar(sx0 + 6, y, side_w - 12, 1.0, 1e5, s.depth_ppm, dm.format_depth(s.depth),
                                ((20.0, "Kepler"), (60.0, "TESS"), (1000.0, "ground")),
                                (("Earth", 84.0), ("TRAPPIST-1e", 7000.0), ("Jupiter", 10500.0)),
                                lambda v: f"{v:g} ppm" if v < 1e4 else f"{v / 1e4:g} %")
        self._wrapped("Depth ≈ (R_p/R★)²; duration T ≈ (P/π)·asin[(R★/a)·√((1+k)²−b²)/sin i] (Winn 2010). "
                      "Transit + RV together give the true mass and hence bulk density.",
                      micro_font, theme.text_tertiary, sx0, y, side_w)

    # -------------------------------------------------------------- Compare tab
    def _render_compare(self, area: pygame.Rect, planet: Dict[str, Any], star: Dict[str, Any], s: dm.DetectionSummary) -> None:
        theme = self.theme
        title_font, label_font, small_font, micro_font, _ = self._fonts()
        siblings = self._siblings(star) or [planet]
        rows = [dm.summarize_planet(p, star, self.inclination_deg) for p in siblings]
        selected_name = s.name

        hdr = micro_font.render(f"ALL PLANETS OF {str(star.get('name', 'THE STAR')).upper()}  ·  coplanar at i = {self.inclination_deg:.1f}°  ·  distance 10 pc assumed for angles",
                                True, theme.text_tertiary)
        self.screen.blit(hdr, (area.left, area.top))
        cols = [
            ("Planet", 104, "l", lambda r: r.name),
            ("a (AU)", 56, "r", lambda r: f"{r.a_au:.3g}"),
            ("Period", 60, "r", lambda r: f"{r.period_days:,.0f} d" if r.period_days >= 100 else f"{r.period_days:.1f} d"),
            ("M (Mearth)", 64, "r", lambda r: _fmt(r.mass_earth)),
            ("R (Rearth)", 60, "r", lambda r: _fmt(r.radius_earth)),
            ("K (RV)", 74, "r", lambda r: dm.format_velocity(r.k_m_s)),
            ("Depth", 66, "r", lambda r: dm.format_depth(r.depth)),
            ("P(tr.)", 52, "r", lambda r: f"{r.transit_prob * 100:.2f}%"),
            ("Dur.", 54, "r", lambda r: dm.format_duration_hours(r.duration_hours)),
            ("Astrom.", 74, "r", lambda r: f"{r.astrometry_uas_10pc:.3g} µas"),
            ("Contrast", 66, "r", lambda r: f"{r.reflected_contrast:.1e}"),
        ]
        total_w = sum(w for _, w, _, _ in cols)
        scale = min(1.0, area.width / total_w)
        x = area.left
        y = area.top + 18
        header_rects = []
        for name, w, align, _ in cols:
            w = int(w * scale)
            t = micro_font.render(name.upper(), True, theme.text_tertiary)
            self.screen.blit(t, t.get_rect(topleft=(x, y)) if align == "l" else t.get_rect(topright=(x + w - 4, y)))
            header_rects.append((x, w, align))
            x += w
        y += 16
        ui_theme.divider(self.screen, area.left, y, area.width, theme); y += 6
        row_h = 22
        max_rows = max(1, (area.height // 2 - 40) // row_h)
        for r in rows[:max_rows]:
            is_sel = r.name == selected_name
            if is_sel:
                hl = pygame.Rect(area.left - 4, y - 3, area.width + 8, row_h)
                pygame.draw.rect(self.screen, _mix(theme.field_bg, theme.accent, 0.25), hl, border_radius=4)
            for (name, w, align, fn), (cx, cw, _) in zip(cols, header_rects):
                txt = fn(r)
                col = theme.text_primary if is_sel or name == "Planet" else theme.text_secondary
                if name == "K (RV)":
                    col = self._tone_color(dm.rv_detectability(r.k_m_s)[1])
                elif name == "Depth":
                    col = self._tone_color(dm.transit_detectability(r.depth_ppm)[1])
                elif name == "Duration" and not r.transits:
                    col = theme.text_tertiary
                surf = small_font.render(txt, True, col)
                while surf.get_width() > cw - 6 and len(txt) > 3:
                    txt = txt[:-2].rstrip() + "…"
                    surf = small_font.render(txt, True, col)
                self.screen.blit(surf, surf.get_rect(topleft=(cx, y)) if align == "l" else surf.get_rect(topright=(cx + cw - 4, y)))
            y += row_h
        if len(rows) > max_rows:
            self.screen.blit(micro_font.render(f"+{len(rows) - max_rows} more planets", True, theme.text_tertiary), (area.left, y)); y += 16
        y += 4
        ui_theme.divider(self.screen, area.left, y, area.width, theme); y += 10

        # Method notes ---------------------------------------------------------
        col_w = (area.width - 24) // 2
        blocks = [
            ("RADIAL VELOCITY", dm.rv_detectability(s.k_m_s),
             f"{s.name}: K = {dm.format_velocity(s.k_m_s)}. Favors massive planets on short orbits around low-mass stars; yields M_p sin i. "
             f"Reference: Jupiter tugs the Sun at 12.5 m/s, Earth at 9 cm/s."),
            ("TRANSIT PHOTOMETRY", dm.transit_detectability(s.depth_ppm),
             f"{s.name}: depth {dm.format_depth(s.depth)}, geometric chance {s.transit_prob * 100:.2f} %. Favors large planets on close orbits "
             f"around small stars; yields R_p/R★ and inclination. Kepler reached ~20 ppm, TESS ~60 ppm."),
            ("ASTROMETRY", ("Gaia-class precision is ~25 µas per epoch" if s.astrometry_uas_10pc >= 25 else "Below Gaia's ~25 µas per-epoch precision",
                            "good" if s.astrometry_uas_10pc >= 25 else "hard"),
             f"{s.name}: the star's reflex wobble subtends {s.astrometry_uas_10pc:.3g} µas at 10 pc. Favors massive planets on wide orbits — the opposite of RV."),
            ("DIRECT IMAGING", ("Within JWST/ELT reach (contrast > 1e-9 at > 100 mas)" if s.reflected_contrast >= 1e-9 and s.separation_mas_10pc >= 100 else "Beyond current contrast / resolution",
                                "ok" if s.reflected_contrast >= 1e-9 and s.separation_mas_10pc >= 100 else "hard"),
             f"{s.name}: reflected-light contrast {s.reflected_contrast:.1e} (A_g = 0.3) at {s.separation_mas_10pc:.0f} mas separation (10 pc). "
             f"Favors young, hot giants far from their star."),
        ]
        col_tops = [y, y]
        for i, (head, (verdict, tone), body) in enumerate(blocks):
            cx = area.left + (i % 2) * (col_w + 24)
            cy = col_tops[i % 2]
            ui_theme.draw_section_label(self.screen, micro_font, head, (cx, cy), theme=theme); cy += 18
            cy = self._wrapped(verdict, small_font, self._tone_color(tone), cx, cy, col_w)
            cy = self._wrapped(body, micro_font, theme.text_secondary, cx, cy + 2, col_w) + 10
            col_tops[i % 2] = cy
        y = max(col_tops)
        self._wrapped("All values are computed from the sandbox parameters with textbook formulas (Winn 2010; Lovis & Fischer 2010). "
                      "Detection verdicts compare against order-of-magnitude instrument precisions and ignore stellar activity, "
                      "observing cadence and photon noise.", micro_font, theme.text_tertiary, area.left, y, area.width)
