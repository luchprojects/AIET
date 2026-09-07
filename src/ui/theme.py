"""
UI theme for AIET v2 (instrument-grade dark mode for Pygame).

Single source of truth for colors, radii, spacing and small drawing components
so the simulator chrome can be refreshed without scattering magic numbers
across main_window.py / diagnostics_panel.py.

Rules
-----
- Tokens are named by ROLE (what they mean), never by value.
- Helpers here are paint-only. They never change geometry that hit-testing
  relies on; callers pass in the rects they already own.
- Nothing in this module touches physics, ML, presets, or persistence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

import pygame

Color = Tuple[int, int, int]
ColorA = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UITheme:
    # Canvas / space
    canvas: Color = (12, 14, 20)
    canvas_grid: Color = (26, 30, 42)
    canvas_grid_fine: Color = (18, 21, 30)
    home_bg: Color = (14, 16, 22)

    # Chrome surfaces
    bar_bg: Color = (22, 24, 30)
    bar_border: Color = (48, 52, 64)
    panel_bg: Color = (28, 30, 38)
    panel_border: Color = (58, 62, 74)
    panel_elevated: Color = (36, 40, 52)
    menu_bg: Color = (30, 33, 43)
    field_bg: Color = (18, 20, 26)

    # Text
    text_primary: Color = (245, 247, 250)
    text_secondary: Color = (168, 174, 188)
    text_tertiary: Color = (118, 124, 138)
    text_on_accent: Color = (255, 255, 255)

    # Accents / semantics
    accent: Color = (10, 132, 255)        # selection + primary action
    accent_soft: Color = (86, 166, 255)
    accent_muted: Color = (30, 52, 86)
    success: Color = (48, 209, 88)        # habitability positive / confirm
    success_soft: Color = (120, 200, 150)
    warning: Color = (255, 159, 10)       # engulfment / instability
    warning_soft: Color = (255, 190, 100)
    warning_muted: Color = (70, 54, 30)
    danger: Color = (255, 69, 58)         # destructive confirm only
    danger_soft: Color = (255, 120, 110)
    danger_muted: Color = (72, 42, 44)

    # Interactive states
    hover: Color = (46, 52, 68)
    selected: Color = (38, 60, 98)
    pressed: Color = (34, 44, 62)
    disabled: Color = (60, 64, 76)

    # Sandbox overlays
    orbit_ring: Color = (78, 86, 104)
    orbit_ring_moon: Color = (58, 64, 80)
    orbit_trail: Color = (64, 70, 88)
    star_trail: Color = (120, 112, 96)
    selection_ring: Color = (86, 166, 255)
    hover_ring: Color = (168, 174, 188)
    hz_overlay: ColorA = (48, 209, 88, 34)
    label_pill: ColorA = (10, 12, 18, 170)
    overlay_scrim: ColorA = (6, 8, 12, 165)
    tooltip_bg: ColorA = (24, 27, 36, 236)
    tooltip_border: Color = (58, 62, 74)

    # Readouts
    hab_text: Color = (120, 200, 150)
    hab_ci_text: Color = (100, 160, 125)
    hud_speed_text: Color = (86, 166, 255)

    # Scientific charts (Star Data panel, diagnostics graphs)
    chart_grid: Color = (36, 40, 52)
    chart_axis: Color = (78, 86, 104)
    chart_axis_text: Color = (118, 124, 138)
    chart_reference: Color = (168, 174, 188)   # Sun / reference overlays
    chart_track: Color = (120, 128, 148)       # schematic evolutionary track
    chart_ms_band: Color = (86, 110, 150)      # main-sequence locus
    chart_line_marker: Color = (200, 206, 218) # spectral line ticks
    chart_hz_band: Color = (72, 168, 112)      # habitable-zone band (HZ over time, transit/RV overlays)
    chart_redshift: Color = (232, 120, 104)    # receding / redshifted half of an RV curve
    chart_blueshift: Color = (104, 164, 240)   # approaching / blueshifted half of an RV curve
    chart_planet: Color = (210, 216, 228)      # generic planet marker on instrument charts

    # Provenance tags (measured vs assumed vs modified)
    tag_measured_bg: Color = (30, 52, 86)
    tag_measured_text: Color = (140, 190, 255)
    tag_assumed_bg: Color = (70, 54, 30)
    tag_assumed_text: Color = (255, 190, 100)
    tag_modified_bg: Color = (44, 48, 60)
    tag_modified_text: Color = (200, 206, 218)

    # Legacy aliases used across main_window
    menu_bg_legacy: Color = (30, 33, 43)
    panel_muted_text: Color = (168, 174, 188)
    panel_border_legacy: Color = (58, 62, 74)
    active_tab: Color = (10, 132, 255)

    # Radii & spacing (4-pt grid)
    radius_sm: int = 6
    radius_md: int = 10
    radius_lg: int = 14
    radius_xl: int = 20
    space_xs: int = 4
    space_sm: int = 8
    space_md: int = 12
    space_lg: int = 16
    space_xl: int = 24


THEME = UITheme()

# Type scale (Inter). Sizes are unchanged from the legacy loader so existing
# layout rects stay valid; the point of this table is a single place to read.
TYPE_SCALE = {
    "display": 80,   # home title
    "h1": 54,        # title_font
    "h2": 28,        # font / button_font: modal titles, readouts
    "body": 18,      # subtitle_font: panel labels, modal body
    "label": 16,     # tab_font, home disclaimer
    "small": 14,     # tiny_font: menus, HUD, tooltips
    "caption": 11,   # micro_font: hints
}

# Real Inter files bundled in src/ui/fonts/. Do not fake weights with set_bold().
FONT_FILES = {
    "regular": "Inter-Regular.ttf",
    "medium": "Inter-Medium.ttf",
    "semibold": "Inter-SemiBold.ttf",
    "bold": "Inter-Bold.ttf",
}

# Apple-style hierarchy: Regular for reading, Medium for chrome/captions
# (Inter at 11–16 px reads thin compared with SF Text), SemiBold for titles,
# Bold only for the large home wordmark.
TYPE_WEIGHTS = {
    "display": "bold",
    "h1": "semibold",
    "h2": "semibold",
    "body": "regular",
    "label": "medium",
    "small": "medium",
    "caption": "medium",
}

_WEIGHT_FALLBACK = {
    "bold": ("bold", "semibold", "medium", "regular"),
    "semibold": ("semibold", "medium", "bold", "regular"),
    "medium": ("medium", "regular"),
    "regular": ("regular",),
}

DIAGNOSTICS_STYLE = {
    "width": 380,
    "min_height": 400,
    "padding": 18,
    "line_spacing": 22,
    "section_spacing": 28,
    "button_height": 32,
    "button_spacing": 10,
    "border_radius": THEME.radius_lg,
    "bg_color": THEME.panel_bg,
    "border_color": THEME.panel_border,
    "header_color": THEME.text_primary,
    "label_color": THEME.text_secondary,
    "value_color": THEME.text_primary,
    "section_bg": THEME.panel_elevated,
    "status_green": THEME.success_soft,
    "status_yellow": THEME.warning_soft,
    "status_red": THEME.danger_soft,
    "status_gray": THEME.text_tertiary,
    "button_normal": THEME.panel_elevated,
    "button_hover": THEME.hover,
    "button_border": THEME.panel_border,
    "button_text": THEME.text_primary,
}


def inter_font_filename(weight: str) -> str:
    """Bundled Inter filename for a named weight."""
    return FONT_FILES.get(weight, FONT_FILES["regular"])


def resolve_inter_path(weight: str, font_path_fn: Callable[[str], str]) -> Optional[str]:
    """
    First existing Inter file along the weight fallback chain.

    Returns None if even Regular is missing (caller then uses SysFont).
    """
    for candidate in _WEIGHT_FALLBACK.get(weight, ("regular",)):
        path = font_path_fn(FONT_FILES[candidate])
        if os.path.exists(path):
            return path
    return None


def make_inter_font(
    size: int,
    weight: str = "regular",
    *,
    font_path_fn: Optional[Callable[[str], str]] = None,
) -> pygame.font.Font:
    """Load one Inter face. Size stays exact; missing weights fall back, never fake-bold."""
    path = None
    if font_path_fn is not None:
        path = resolve_inter_path(weight, font_path_fn)
    if path:
        try:
            return pygame.font.Font(path, size)
        except Exception:
            pass
    try:
        return pygame.font.SysFont("Inter", size)
    except Exception:
        return pygame.font.Font(None, size)


def load_type_kit(font_path_fn: Callable[[str], str]) -> dict:
    """All TYPE_SCALE faces with TYPE_WEIGHTS applied. Keys match TYPE_SCALE."""
    return {
        role: make_inter_font(size, TYPE_WEIGHTS.get(role, "regular"), font_path_fn=font_path_fn)
        for role, size in TYPE_SCALE.items()
    }


def render_tracked(
    font: pygame.font.Font,
    text: str,
    color: Color,
    tracking: int = 1,
) -> pygame.Surface:
    """
    Render uppercase labels with 1 px tracking (SF Text caption convention).

    Tracking is paint-only: callers still own layout rects.
    """
    if not text:
        return pygame.Surface((1, 1), pygame.SRCALPHA)
    glyphs = [font.render(ch, True, color) for ch in text]
    width = sum(g.get_width() for g in glyphs) + tracking * max(0, len(glyphs) - 1)
    height = max(g.get_height() for g in glyphs)
    surf = pygame.Surface((max(1, width), height), pygame.SRCALPHA)
    x = 0
    for glyph in glyphs:
        surf.blit(glyph, (x, 0))
        x += glyph.get_width() + tracking
    return surf


def _pointer_pressed_on(rect: pygame.Rect) -> bool:
    try:
        return bool(pygame.mouse.get_pressed()[0] and rect.collidepoint(pygame.mouse.get_pos()))
    except Exception:
        return False


def apply_theme(viz: Any, theme: UITheme = THEME) -> None:
    """
    Attach theme tokens to the visualizer instance.

    Paint-only: this overrides the legacy color attributes main_window reads
    everywhere, but never touches geometry (button heights, margins, option
    heights) because hit-testing depends on those.
    """
    viz.theme = theme

    viz.DARK_BLUE = theme.canvas
    viz.GRID_COLOR = theme.canvas_grid
    viz.LIGHT_GRAY = theme.text_secondary
    viz.GRAY = theme.text_tertiary

    viz.UI_MENU_BG = theme.menu_bg_legacy
    viz.UI_PANEL_BG = theme.panel_bg
    viz.UI_PANEL_TEXT = theme.text_primary
    viz.UI_PANEL_MUTED_TEXT = theme.panel_muted_text
    viz.UI_PANEL_BORDER = theme.panel_border_legacy
    viz.ACTIVE_TAB_COLOR = theme.active_tab

    viz.dropdown_background_color = theme.panel_elevated
    viz.dropdown_text_color = theme.text_primary
    viz.dropdown_border_color = theme.panel_border


# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------


def fill_canvas(screen: pygame.Surface, theme: UITheme = THEME) -> None:
    """Deep space background."""
    screen.fill(theme.canvas)


def fill_home_canvas(screen: pygame.Surface, theme: UITheme = THEME) -> None:
    """Home screen background: flat with one very soft accent bloom behind the title."""
    w, h = screen.get_size()
    screen.fill(theme.home_bg)
    glow = pygame.Surface((w, h), pygame.SRCALPHA)
    r, g, b = theme.accent
    pygame.draw.circle(glow, (r, g, b, 14), (w // 2, int(h * 0.18)), min(w, h) // 3)
    screen.blit(glow, (0, 0))


def draw_scrim(screen: pygame.Surface, theme: UITheme = THEME) -> None:
    """Full-screen dim behind modals."""
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    overlay.fill(theme.overlay_scrim)
    screen.blit(overlay, (0, 0))


def draw_shadow(
    screen: pygame.Surface,
    rect: pygame.Rect,
    *,
    radius: int,
    offset: int = 4,
    alpha: int = 90,
) -> None:
    shadow = rect.move(0, offset)
    shadow_surf = pygame.Surface(shadow.size, pygame.SRCALPHA)
    pygame.draw.rect(shadow_surf, (0, 0, 0, alpha), shadow_surf.get_rect(), border_radius=radius)
    screen.blit(shadow_surf, shadow.topleft)


def draw_top_bar(screen: pygame.Surface, rect: pygame.Rect, theme: UITheme = THEME) -> None:
    pygame.draw.rect(screen, theme.bar_bg, rect)
    pygame.draw.line(screen, theme.bar_border, (rect.left, rect.bottom - 1), (rect.right, rect.bottom - 1), 1)


def draw_side_panel(screen: pygame.Surface, rect: pygame.Rect, theme: UITheme = THEME) -> None:
    """Right-hand customization panel: flat surface with a hairline on the inner edge."""
    if rect.width <= 0 or rect.height <= 0:
        return
    pygame.draw.rect(screen, theme.panel_bg, rect)
    pygame.draw.line(screen, theme.panel_border, (rect.left, rect.top), (rect.left, rect.bottom), 1)


def draw_menu_surface(
    screen: pygame.Surface,
    rect: pygame.Rect,
    theme: UITheme = THEME,
    radius: Optional[int] = None,
) -> None:
    r = radius if radius is not None else theme.radius_sm
    draw_shadow(screen, rect, radius=r, offset=3, alpha=110)
    pygame.draw.rect(screen, theme.menu_bg, rect, border_radius=r)
    pygame.draw.rect(screen, theme.panel_border, rect, 1, border_radius=r)


def draw_hud_panel(screen: pygame.Surface, rect: pygame.Rect, theme: UITheme = THEME) -> None:
    """Translucent HUD card (time controls, scale bar)."""
    surf = pygame.Surface(rect.size, pygame.SRCALPHA)
    r, g, b = theme.panel_bg
    pygame.draw.rect(surf, (r, g, b, 210), surf.get_rect(), border_radius=theme.radius_sm)
    screen.blit(surf, rect.topleft)
    pygame.draw.rect(screen, theme.panel_border, rect, 1, border_radius=theme.radius_sm)


def draw_modal_frame(
    screen: pygame.Surface,
    rect: pygame.Rect,
    *,
    tone: str = "neutral",
    scrim: bool = True,
    theme: UITheme = THEME,
) -> None:
    """
    One modal chrome for every dialog.

    tone: "neutral" (accent stripe), "warning" (orange), "danger" (red).
    """
    if scrim:
        draw_scrim(screen, theme)
    r = theme.radius_lg
    draw_shadow(screen, rect, radius=r, offset=6, alpha=140)
    pygame.draw.rect(screen, theme.panel_bg, rect, border_radius=r)
    pygame.draw.rect(screen, theme.panel_border, rect, 1, border_radius=r)
    stripe_color = {
        "warning": theme.warning,
        "danger": theme.danger,
    }.get(tone, theme.accent)
    stripe = pygame.Rect(rect.left + r, rect.top, rect.width - 2 * r, 2)
    pygame.draw.rect(screen, stripe_color, stripe)


def draw_toast(
    screen: pygame.Surface,
    rect: pygame.Rect,
    *,
    tone: str = "neutral",
    theme: UITheme = THEME,
) -> None:
    """Transient notice card. tone: neutral | warning | danger."""
    surf = pygame.Surface(rect.size, pygame.SRCALPHA)
    r, g, b = theme.panel_elevated
    pygame.draw.rect(surf, (r, g, b, 236), surf.get_rect(), border_radius=theme.radius_sm)
    screen.blit(surf, rect.topleft)
    pygame.draw.rect(screen, theme.panel_border, rect, 1, border_radius=theme.radius_sm)
    stripe_color = {
        "warning": theme.warning,
        "danger": theme.danger,
    }.get(tone, theme.accent)
    pygame.draw.rect(screen, stripe_color, pygame.Rect(rect.left, rect.top + 6, 2, rect.height - 12))


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def draw_toolbar_button(
    screen: pygame.Surface,
    rect: pygame.Rect,
    *,
    hover: bool = False,
    active: bool = False,
    pressed: Optional[bool] = None,
    theme: UITheme = THEME,
) -> None:
    if pressed is None:
        pressed = _pointer_pressed_on(rect)
    if pressed:
        fill = theme.pressed
        border = theme.accent_soft if active else theme.panel_border
    elif active:
        fill = theme.selected
        border = theme.accent_soft
    elif hover:
        fill = theme.hover
        border = theme.panel_border
    else:
        fill = theme.panel_elevated
        border = theme.panel_border
    pygame.draw.rect(screen, fill, rect, border_radius=theme.radius_sm)
    pygame.draw.rect(screen, border, rect, 1, border_radius=theme.radius_sm)


def draw_primary_button(
    screen: pygame.Surface,
    rect: pygame.Rect,
    *,
    hover: bool = False,
    theme: UITheme = THEME,
) -> None:
    fill = theme.accent_soft if hover else theme.accent
    pygame.draw.rect(screen, fill, rect, border_radius=theme.radius_md)


def draw_button(
    screen: pygame.Surface,
    rect: pygame.Rect,
    label: str,
    font: pygame.font.Font,
    *,
    kind: str = "secondary",
    hover: bool = False,
    active: bool = False,
    disabled: bool = False,
    theme: UITheme = THEME,
) -> None:
    """
    Button with label. kind: primary | secondary | danger | warning | ghost.

    Callers keep ownership of `rect` (hit-testing); this only paints.
    """
    radius = theme.radius_sm
    pressed = (not disabled) and _pointer_pressed_on(rect)
    if disabled:
        fill, border, text = theme.disabled, theme.disabled, theme.text_tertiary
    elif kind == "primary":
        fill = theme.accent if pressed else (theme.accent_soft if hover else theme.accent)
        border, text = fill, theme.text_on_accent
    elif kind == "danger":
        fill = theme.danger if pressed else (theme.danger_soft if hover else theme.danger)
        border, text = fill, theme.text_on_accent
    elif kind == "warning":
        fill = theme.warning if pressed else (theme.warning_soft if hover else theme.warning)
        border, text = fill, (20, 16, 8)
    elif kind == "ghost":
        fill = theme.pressed if pressed else (theme.hover if hover else None)
        border, text = theme.panel_border, theme.text_primary
    else:  # secondary
        fill = theme.pressed if pressed else (theme.hover if hover else theme.panel_elevated)
        border = theme.accent_soft if active else theme.panel_border
        text = theme.text_primary
    if fill is not None:
        pygame.draw.rect(screen, fill, rect, border_radius=radius)
    pygame.draw.rect(screen, border, rect, 1, border_radius=radius)
    surf = font.render(label, True, text)
    # Optical nudge: Latin caps sit high in the em box; −1 px matches SF label centering.
    label_rect = surf.get_rect(center=rect.center)
    label_rect.y -= 1
    screen.blit(surf, label_rect)


def draw_input_field(
    screen: pygame.Surface,
    rect: pygame.Rect,
    *,
    focused: bool = False,
    theme: UITheme = THEME,
) -> None:
    pygame.draw.rect(screen, theme.field_bg, rect, border_radius=theme.radius_sm)
    border = theme.accent if focused else theme.panel_border
    pygame.draw.rect(screen, border, rect, 1, border_radius=theme.radius_sm)
    if focused:
        # Focus ring sits outside the hit rect (Apple HIG); geometry is unchanged.
        ring = rect.inflate(4, 4)
        pygame.draw.rect(screen, theme.accent_soft, ring, 1, border_radius=theme.radius_sm + 2)


def item_background(
    theme: UITheme,
    hover: bool,
    selected: bool = False,
) -> Color:
    if hover:
        return theme.hover
    if selected:
        return theme.selected
    return theme.menu_bg


def draw_menu_item(
    screen: pygame.Surface,
    rect: pygame.Rect,
    *,
    hover: bool = False,
    selected: bool = False,
    theme: UITheme = THEME,
) -> Color:
    """Paint a menu row; returns the text color to use for its label."""
    pygame.draw.rect(screen, item_background(theme, hover, selected), rect)
    if selected and not hover:
        pygame.draw.rect(screen, theme.accent, pygame.Rect(rect.left, rect.top + 6, 2, rect.height - 12))
    return theme.text_primary if (hover or selected) else theme.text_secondary


def draw_chevron(
    screen: pygame.Surface,
    center: Tuple[int, int],
    *,
    open_: bool,
    color: Color,
) -> None:
    """Small disclosure chevron: right when closed, down when open."""
    cx, cy = center
    if open_:
        pts = [(cx - 4, cy - 2), (cx + 4, cy - 2), (cx, cy + 3)]
    else:
        pts = [(cx - 2, cy - 4), (cx - 2, cy + 4), (cx + 3, cy)]
    pygame.draw.polygon(screen, color, pts)


def draw_close_x(
    screen: pygame.Surface,
    rect: pygame.Rect,
    *,
    hover: bool = False,
    theme: UITheme = THEME,
) -> None:
    if hover:
        pygame.draw.rect(screen, theme.hover, rect, border_radius=theme.radius_sm)
    color = theme.text_primary if hover else theme.text_secondary
    inset = max(4, rect.width // 4)
    pygame.draw.line(screen, color, (rect.left + inset, rect.top + inset), (rect.right - inset, rect.bottom - inset), 2)
    pygame.draw.line(screen, color, (rect.left + inset, rect.bottom - inset), (rect.right - inset, rect.top + inset), 2)


def draw_info_glyph(
    screen: pygame.Surface,
    center: Tuple[int, int],
    font: pygame.font.Font,
    *,
    radius: int = 7,
    hover: bool = False,
    theme: UITheme = THEME,
) -> None:
    """Circled 'i' used for parameter tooltips."""
    color = theme.accent_soft if hover else theme.text_tertiary
    pygame.draw.circle(screen, color, center, radius, 1)
    glyph = font.render("i", True, color)
    screen.blit(glyph, glyph.get_rect(center=(center[0], center[1] + 0)))


def draw_tag(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    topleft: Tuple[int, int],
    *,
    kind: str = "measured",
    theme: UITheme = THEME,
) -> pygame.Rect:
    """
    Provenance pill. kind: measured | assumed | modified.
    Returns the painted rect so callers can lay out after it.
    """
    bg, fg = {
        "assumed": (theme.tag_assumed_bg, theme.tag_assumed_text),
        "modified": (theme.tag_modified_bg, theme.tag_modified_text),
    }.get(kind, (theme.tag_measured_bg, theme.tag_measured_text))
    surf = font.render(text.upper(), True, fg)
    rect = surf.get_rect()
    rect.inflate_ip(10, 4)
    rect.topleft = topleft
    pygame.draw.rect(screen, bg, rect, border_radius=rect.height // 2)
    screen.blit(surf, surf.get_rect(center=rect.center))
    return rect


def draw_text_pill(
    screen: pygame.Surface,
    surf: pygame.Surface,
    center: Tuple[int, int],
    *,
    theme: UITheme = THEME,
) -> pygame.Rect:
    """Backing pill for floating labels over the sandbox. Returns the label rect."""
    rect = surf.get_rect(center=center)
    pill = rect.inflate(10, 4)
    pill_surf = pygame.Surface(pill.size, pygame.SRCALPHA)
    pygame.draw.rect(pill_surf, theme.label_pill, pill_surf.get_rect(), border_radius=pill.height // 2)
    screen.blit(pill_surf, pill.topleft)
    screen.blit(surf, rect)
    return rect


def divider(screen: pygame.Surface, x: int, y: int, width: int, theme: UITheme = THEME) -> None:
    pygame.draw.line(screen, theme.panel_border, (x, y), (x + width, y), 1)


def draw_section_label(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    topleft: Tuple[int, int],
    *,
    theme: UITheme = THEME,
) -> pygame.Rect:
    """Uppercase tracked section label (panel sub-headers)."""
    surf = render_tracked(font, text.upper(), theme.text_tertiary, tracking=1)
    rect = surf.get_rect(topleft=topleft)
    screen.blit(surf, rect)
    return rect


def wrap_text(font: pygame.font.Font, text: str, max_width: int) -> Sequence[str]:
    """Greedy word wrap to max_width pixels."""
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        test = " ".join(cur + [w])
        if font.size(test)[0] <= max_width or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines
