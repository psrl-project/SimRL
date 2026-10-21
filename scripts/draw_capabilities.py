#!/usr/bin/env python
"""Render the four-panel 'what SimRL can do' figure to SVG and PNG."""
from __future__ import annotations

import math

from figure_canvas import (
    BLUE,
    GOLD,
    INK,
    MUTED,
    OUT_DIRS,
    PALE_BLUE,
    PALE_GOLD,
    PALE_RED,
    PALE_TEAL,
    RED,
    TEAL,
    Canvas,
)

PANEL_W, PANEL_H = 720, 400
PANEL_X = (60, 820)
PANEL_Y = (150, 596)


def panel(c, col, row, letter, title, color, fill):
    x, y = PANEL_X[col], PANEL_Y[row]
    c.rect(x, y, PANEL_W, PANEL_H, fill=fill, radius=10, outline=False)
    c.text(x + 28, y + 28, letter, 36, color, True)
    c.text(x + 66, y + 28, title, 36, INK, True)
    return x, y


def draw_a(c):
    x, y = panel(c, 0, 0, "A", "Simulate frameworks", BLUE, PALE_BLUE)
    c.text(x + 28, y + 82, "10+ RL systems, one control plane", 28, MUTED)

    bw, bh, gap = 144, 70, 30
    top = y + 136
    names = ["Dataset", "Rollout", "Queue", "Train"]
    boxes = []
    for i, name in enumerate(names):
        bx = x + 28 + i * (bw + gap)
        c.rect(bx, top, bw, bh)
        c.box_text(bx, top, bw, bh, name, 27)
        boxes.append((bx, bx + bw))
        if i:
            c.arrow(boxes[i - 1][1] + 5, top + bh / 2, bx - 5, top + bh / 2, MUTED, 2)

    # Arrowheads stop between two stages, the three places a system can
    # intervene, coloured as in the staleness-blog pipeline figure.
    targets = [boxes[1][0] - 9, boxes[2][0] - 9, boxes[3][0] - 9]
    tags = [("L1  Admission", BLUE), ("L2  Runtime", GOLD), ("L3  Filtering", TEAL)]
    starts = [x + 26, x + 254, x + 480]
    for (label, color), start, tx in zip(tags, starts, targets):
        c.arrow(start, top + bh + 80, tx, top + bh + 10, color, 3)
        c.text(start, top + bh + 100, label, 24, color, True)

    c.text(x + 28, top + bh + 148, "verl · AReaL · SkyRL · ROLL · slime · OpenRLHF",
           21, MUTED)
    c.text(x + 28, top + bh + 178, "TRL · NeMo-RL · Dressage · prime-rl · PSRL",
           21, MUTED)


def draw_b(c):
    x, y = panel(c, 1, 0, "B", "Design a deployment", GOLD, PALE_GOLD)
    c.text(x + 28, y + 82, "Pick the GPU split and the parallelism", 28, MUTED)

    cell, gap = 48, 11
    row = y + 192
    roll, train = 5, 3
    gx = x + 36
    for _ in range(roll + train):
        c.rect(gx, row, cell, cell, radius=10)
        gx += cell + gap

    roll_w = roll * cell + (roll - 1) * gap
    train_x = x + 36 + (roll + train) * (cell + gap) - gap - train * cell - (train - 1) * gap
    train_w = train * cell + (train - 1) * gap
    c.stroke_rect(x + 30, row - 10, roll_w + 8, cell + 20, GOLD, 3)
    c.stroke_rect(train_x - 2, row - 10, train_w + 12, cell + 20, TEAL, 3)
    c.text(x + 34, row - 56, "rollout", 26, GOLD, True)
    c.text(train_x, row - 56, "training", 26, TEAL, True)

    # The split itself is the knob, so show it can slide either way.
    c.arrow_both(x + 30, row + 86, x + 30 + roll_w + train_w + 18, row + 86, MUTED, 2)
    c.text(x + 34, row + 100, "more GPUs here means fewer there", 22, MUTED)

    c.text(x + 34, row + 154, "rollout: TP", 26, MUTED, True)
    c.text(x + 34, row + 190, "training: TP · PP · DP · EP · CP", 26, MUTED, True)


def draw_c(c):
    x, y = panel(c, 0, 1, "C", "Predict at runtime", TEAL, PALE_TEAL)
    c.text(x + 28, y + 82, "Load shifts, so scale the fleet", 28, MUTED)

    bw, bh = 160, 92
    sx, sy = x + 44, y + 166
    c.rect(sx, sy, bw, bh)
    c.box_text(sx, sy, bw, bh, "live system", 27)

    fx = sx + bw + 76
    c.rect(fx, sy, bw + 90, bh)
    c.text(fx + 20, sy + 24, "forecast", 27, INK, True)
    c.text(fx + 20, sy + 64, "ETA · KV · step", 23, MUTED)
    c.arrow(sx + bw + 6, sy + bh / 2, fx - 6, sy + bh / 2, MUTED, 2)

    back = fx + bw + 90 + 38
    c.dashes(fx + bw + 90, sy + bh / 2, back, sy + bh / 2, GOLD, 2)
    c.dashes(back, sy + bh / 2, back, sy + bh + 78, GOLD, 2)
    c.dashes(back, sy + bh + 78, sx + bw / 2, sy + bh + 78, GOLD, 2)
    c.arrow(sx + bw / 2, sy + bh + 78, sx + bw / 2, sy + bh + 6, GOLD, 2)
    c.text(sx + bw / 2 + 26, sy + bh + 100, "move engines · change TP", 25, GOLD, True)


def draw_d(c):
    x, y = panel(c, 1, 1, "D", "Auto research", RED, PALE_RED)
    c.text(x + 28, y + 82, "Is kv-cache aware routing better here?", 28, MUTED)

    bw, bh = 196, 68
    cx = x + PANEL_W / 2
    boxes = {
        "proposal agent": (cx - bw / 2, y + 150),
        "validation agent": (cx - bw - 66, y + 302),
        "analysis agent": (cx + 66, y + 302),
    }
    for label, (nx, ny) in boxes.items():
        c.rect(nx, ny, bw, bh)
        c.box_text(nx, ny, bw, bh, label, 25)

    gx, gy = cx, y + 258

    def rim(target, source):
        tx, ty = boxes[target][0] + bw / 2, boxes[target][1] + bh / 2
        sx, sy = boxes[source][0] + bw / 2, boxes[source][1] + bh / 2
        dx, dy = tx - sx, ty - sy
        dist = math.hypot(dx, dy)
        ux, uy = dx / dist, dy / dist
        reach = min((bw / 2) / abs(ux) if abs(ux) > 1e-6 else float("inf"),
                    (bh / 2) / abs(uy) if abs(uy) > 1e-6 else float("inf"))
        trim = reach + 12
        return (sx + ux * trim, sy + uy * trim, tx - ux * trim, ty - uy * trim)

    # Bow every edge away from the centre so the three read as a ring rather
    # than a triangle with straight lines.
    cycle = [("proposal agent", "validation agent"),
             ("validation agent", "analysis agent"),
             ("analysis agent", "proposal agent")]
    for source, target in cycle:
        x1, y1, x2, y2 = rim(target, source)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = mx - gx, my - gy
        norm = math.hypot(dx, dy) or 1
        chord = math.hypot(x2 - x1, y2 - y1)
        ctrl_x = mx + dx / norm * chord * 0.42
        ctrl_y = my + dy / norm * chord * 0.42
        c.curve_arrow(x1, y1, x2, y2, ctrl_x, ctrl_y, MUTED, 2)


def build(out_dirs=None):
    for out_dir in out_dirs or OUT_DIRS:
        c = Canvas("simrl-capabilities", 1600, 1046, "What you can do with SimRL", out_dir)
        c.text(60, 46, "What you can do with SimRL", 54, INK, True)
        c.text(60, 106,
               "Simulate a framework, design a deployment, predict at runtime, research policies.",
               30, MUTED)
        draw_a(c)
        draw_b(c)
        draw_c(c)
        draw_d(c)
        c.save()
        # Every embed (README, docs/index.md) sits right under its own
        # heading saying "What you can do with SimRL", so the in-figure
        # title would be a duplicate -- crop it for both.
        c.save_plain(146)


if __name__ == "__main__":
    build()
