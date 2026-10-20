"""Shared drawing primitives for the README and docs figures.

Style comes from the blog toolchain's ``figure-style/style.json`` so every
figure stays in the same visual family: white canvas, thin rounded panels,
slightly sketchy connectors, Kalam plus LXGW WenKai.
"""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BLOG = ROOT.parent / "blog"
STYLE_PATH = BLOG / "figure-style" / "style.json"
FONT_DIR = BLOG / "figure-style" / "fonts"
OUT_DIRS = [ROOT / "assets", ROOT / "docs" / "_static" / "img"]

STYLE = json.loads(STYLE_PATH.read_text(encoding="utf-8"))
COLORS = STYLE["colors"]
BG, INK, MUTED = COLORS["background"], COLORS["ink"], COLORS["muted"]
TEAL, GOLD, RED = COLORS["teal"], COLORS["gold"], COLORS["red"]
BLUE, WHITE, LINE = COLORS["blue"], COLORS["white"], COLORS["line"]
PALE_TEAL, PALE_GOLD = COLORS["pale_teal"], COLORS["pale_gold"]
PALE_BLUE, PALE_GRAY, PALE_RED = COLORS["pale_blue"], COLORS["pale_gray"], COLORS["pale_red"]

FONT_LATIN = FONT_DIR / "Kalam-Regular.ttf"
BOLD_LATIN = FONT_DIR / "Kalam-Bold.ttf"
FONT_CJK = FONT_DIR / "LXGWWenKai-Regular.ttf"
BOLD_CJK = FONT_DIR / "LXGWWenKai-Regular.ttf"

# Fonts are referenced relative to the SVG, which is published in two
# directories at different depths.
FONT_RELATIVE = {
    ROOT / "assets": "../../blog/figure-style/fonts",
    ROOT / "docs" / "_static" / "img": "../../../../blog/figure-style/fonts",
}


@lru_cache(maxsize=200)
def load_font(path, size):
    return ImageFont.truetype(str(path), size)


def font_for(char, size, bold):
    if ord(char) < 256 and unicodedata.east_asian_width(char) not in ("W", "F"):
        return load_font(BOLD_LATIN if bold else FONT_LATIN, size)
    return load_font(BOLD_CJK if bold else FONT_CJK, size)


def svg_family(char):
    return STYLE["fonts"]["latin"]["family"] if ord(char) < 256 else STYLE["fonts"]["cjk"]["family"]


class Canvas:
    def __init__(self, name, w, h, title, out_dir):
        self.name, self.w, self.h, self.title, self.out_dir = name, w, h, title, out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.im = Image.new("RGB", (w, h), BG)
        self.d = ImageDraw.Draw(self.im)
        base = FONT_RELATIVE[out_dir]
        css = []
        for family, regular, bold in (("Kalam", FONT_LATIN, BOLD_LATIN),
                                      ("LXGW WenKai", FONT_CJK, BOLD_CJK)):
            for weight, path in ((400, regular), (700, bold)):
                css.append(f'@font-face{{font-family:"{family}";'
                           f'src:url("{base}/{path.name}");font-weight:{weight}}}')
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img"><title>{title}</title>',
            f'<style>{"".join(css)}</style>',
            f'<rect width="{w}" height="{h}" fill="{BG}"/>',
        ]

    def rect(self, x, y, w, h, fill=WHITE, radius=16, stroke=None, outline=True):
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > self.w or y + h > self.h:
            raise ValueError(f"{self.name}: panel outside canvas: {(x, y, w, h)}")
        radius *= STYLE["shapes"]["corner_radius_scale"]
        if outline and fill == WHITE and STYLE["shapes"]["outline_white_panels"] and stroke is None:
            stroke = LINE
        border = max(1, round(STYLE["lines"]["border_width"]))
        self.d.rounded_rectangle((x, y, x + w, y + h), radius, fill, outline=stroke, width=border)
        self.svg.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}"'
            + (f' stroke="{stroke}" stroke-width="{STYLE["lines"]["border_width"]}"' if stroke else "")
            + "/>"
        )

    def stroke_rect(self, x, y, w, h, color, width=3, radius=10):
        radius *= STYLE["shapes"]["corner_radius_scale"]
        self.d.rounded_rectangle((x, y, x + w, y + h), radius, fill=None, outline=color, width=width)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
                        f'fill="none" stroke="{color}" stroke-width="{width}"/>')

    def line(self, x1, y1, x2, y2, color=LINE, width=2):
        width = max(1, round(width * STYLE["lines"]["width_scale"]))
        distance = math.hypot(x2 - x1, y2 - y1)
        seed = hashlib.sha256(f"{x1},{y1},{x2},{y2}".encode()).digest()[0] / 255 - 0.5
        offset = STYLE["lines"]["sketch_offset"] * seed * 2 if distance > 40 else 0
        cx = (x1 + x2) / 2 - (y2 - y1) / max(1, distance) * offset
        cy = (y1 + y2) / 2 + (x2 - x1) / max(1, distance) * offset
        points = []
        for step in range(21):
            t = step / 20
            points.append(((1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t * t * x2,
                           (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t * t * y2))
        self.d.line(points, fill=color, width=width)
        self.svg.append(f'<path d="M {x1} {y1} Q {cx:.2f} {cy:.2f} {x2} {y2}" '
                        f'fill="none" stroke="{color}" stroke-width="{width}"/>')

    def _head(self, x, y, angle, color, width):
        for delta in (-0.52, 0.52):
            self.line(x, y,
                      x - 15 * math.cos(angle + delta), y - 15 * math.sin(angle + delta),
                      color, width)

    def arrow(self, x1, y1, x2, y2, color=MUTED, width=3):
        self.line(x1, y1, x2, y2, color, width)
        self._head(x2, y2, math.atan2(y2 - y1, x2 - x1), color, width)

    def arrow_both(self, x1, y1, x2, y2, color=MUTED, width=2):
        self.line(x1, y1, x2, y2, color, width)
        angle = math.atan2(y2 - y1, x2 - x1)
        self._head(x2, y2, angle, color, width)
        self._head(x1, y1, angle + math.pi, color, width)

    def curve_arrow(self, x1, y1, x2, y2, ctrl_x, ctrl_y, color=MUTED, width=2):
        points = []
        for step in range(25):
            t = step / 24
            points.append(((1 - t) ** 2 * x1 + 2 * (1 - t) * t * ctrl_x + t * t * x2,
                           (1 - t) ** 2 * y1 + 2 * (1 - t) * t * ctrl_y + t * t * y2))
        self.d.line(points, fill=color, width=max(1, round(width)))
        self.svg.append(f'<path d="M {x1:.1f} {y1:.1f} Q {ctrl_x:.1f} {ctrl_y:.1f} '
                        f'{x2:.1f} {y2:.1f}" fill="none" stroke="{color}" stroke-width="{width}"/>')
        angle = math.atan2(y2 - ctrl_y, x2 - ctrl_x)
        self._head(x2, y2, angle, color, width)

    def dashes(self, x1, y1, x2, y2, color=MUTED, width=2, dash=12, gap=10):
        length = math.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        cursor = 0.0
        while cursor < length:
            end = min(cursor + dash, length)
            self.line(x1 + ux * cursor, y1 + uy * cursor, x1 + ux * end, y1 + uy * end, color, width)
            cursor = end + gap

    def measure(self, s, size=28, bold=False):
        runs = []
        for ch in s:
            family = svg_family(ch)
            if runs and runs[-1][0] == family:
                runs[-1][1] += ch
            else:
                runs.append([family, ch])
        width = sum(self.d.textlength(run, font=font_for(run[0], size, bold)) for _, run in runs)
        top = min(font_for(run[0], size, bold).getbbox(run, anchor="ls")[1] for _, run in runs)
        bottom = max(font_for(run[0], size, bold).getbbox(run, anchor="ls")[3] for _, run in runs)
        return width, bottom - top

    def text(self, x, y, s, size=28, color=INK, bold=False, maxw=None):
        runs = []
        for ch in s:
            family = svg_family(ch)
            if runs and runs[-1][0] == family:
                runs[-1][1] += ch
            else:
                runs.append([family, ch])
        length = sum(self.d.textlength(run, font=font_for(run[0], size, bold)) for _, run in runs)
        if maxw is not None and length > maxw:
            raise ValueError(f"{self.name}: text too wide ({length:.1f} > {maxw}): {s}")
        if x + length > self.w - 20:
            raise ValueError(f"{self.name}: text outside canvas: {s}")
        baseline = y + max(-font_for(run[0], size, bold).getbbox(run, anchor="ls")[1] for _, run in runs)
        bottom = baseline + max(font_for(run[0], size, bold).getbbox(run, anchor="ls")[3] for _, run in runs)
        if x < 0 or y < 0 or bottom > self.h - 16:
            raise ValueError(f"{self.name}: text outside canvas vertically: {s}")
        svg_runs = []
        cursor = x
        for family, run in runs:
            f = font_for(run[0], size, bold)
            self.d.text((cursor, baseline), run, font=f, fill=color, anchor="ls")
            svg_runs.append(f'<tspan x="{cursor:.2f}" font-family="{family}">{run}</tspan>')
            cursor += self.d.textlength(run, font=f)
        self.svg.append(f'<text x="{x}" y="{baseline}" font-size="{size}" '
                        f'font-weight="{700 if bold else 400}" fill="{color}">'
                        f'{"".join(svg_runs)}</text>')

    def box_text(self, x, y, w, h, s, size=28, color=INK, bold=False):
        width, height = self.measure(s, size, bold)
        self.text(x + (w - width) / 2, y + (h - height) / 2, s, size, color, bold, maxw=w)

    def save(self):
        self.im.save(self.out_dir / f"{self.name}.png")
        (self.out_dir / f"{self.name}.svg").write_text("\n".join(self.svg + ["</svg>"]), encoding="utf-8")

    def save_plain(self, crop_top, suffix="-plain"):
        """Save a second variant with the in-figure title band cropped off.

        Pans the SVG viewBox and crops the PNG instead of redrawing: every panel
        keeps the exact coordinates it was authored with, so the titled and plain
        variants can never drift apart.
        """
        name = f"{self.name}{suffix}"
        new_h = self.h - crop_top
        self.im.crop((0, crop_top, self.w, self.h)).save(self.out_dir / f"{name}.png")
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{new_h}" '
            f'viewBox="0 {crop_top} {self.w} {new_h}" role="img"><title>{self.title}</title>'
        )
        svg = [head, *self.svg[1:], "</svg>"]
        (self.out_dir / f"{name}.svg").write_text("\n".join(svg), encoding="utf-8")
