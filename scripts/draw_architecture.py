#!/usr/bin/env python
"""Render the SimRL architecture figure to SVG and PNG.

Three stacked bands, one per stage of the pipeline: measure with profiler
adapters, cost modeling fitted from their records, simulate with the engine
loop that asks those models for every duration.
"""
from __future__ import annotations

from figure_canvas import (
    BLUE,
    GOLD,
    INK,
    MUTED,
    OUT_DIRS,
    PALE_BLUE,
    PALE_GOLD,
    PALE_TEAL,
    TEAL,
    Canvas,
)

BAND_X, BAND_W = 60, 1480
BAND_H = 256
BAND_GAP = 84
BAND_Y = (150, 150 + BAND_H + BAND_GAP, 150 + 2 * (BAND_H + BAND_GAP))
CARD_H = 180
CARD_TOP = 22
CARD_GAP = 28
LINE_GAP = 33
LABEL_W = 300
COL_X = BAND_X + LABEL_W
ROW_RIGHT = BAND_X + BAND_W - 20

SUB_TONES = {
    "profiler": (PALE_BLUE, BLUE),
    "cost": (PALE_GOLD, GOLD),
    "simulate": (PALE_TEAL, TEAL),
}


def band(c, y, letter, title, module_path, gpu_note, tone):
    fill, color = SUB_TONES[tone]
    c.rect(BAND_X, y, BAND_W, BAND_H, fill=fill, radius=10, outline=False)
    c.text(BAND_X + 26, y + 30, letter, 36, color, True)
    c.text(BAND_X + 64, y + 30, title, 36, INK, True)
    c.text(BAND_X + 26, y + 96, module_path, 24, MUTED)
    c.text(BAND_X + 26, y + 130, gpu_note, 24, MUTED, True)


def card(c, x, y, w, title, lines, tone):
    _, color = SUB_TONES[tone]
    c.rect(x, y, w, CARD_H)
    pad = 18
    c.text(x + pad, y + 16, title, 28, color, True, maxw=w - 2 * pad)
    # Centre the body lines in the space under the title so a 2-line card
    # and a 3-line card both look intentional, not half-empty.
    title_bottom = y + 16 + 38
    body_h = (len(lines) - 1) * LINE_GAP + 30
    start = title_bottom + (CARD_H - (title_bottom - y) - body_h) / 2
    for i, line in enumerate(lines):
        c.text(x + pad, start + i * LINE_GAP, line, 23, MUTED, maxw=w - 2 * pad)


def row(c, y, weights, specs, tone):
    """Lay out len(weights) cards left to right, sized proportionally to
    weights, spanning exactly from COL_X to ROW_RIGHT with CARD_GAP between."""
    total_gap = CARD_GAP * (len(weights) - 1)
    available = ROW_RIGHT - COL_X - total_gap
    unit = available / sum(weights)
    x = COL_X
    for w, (title, lines) in zip(weights, specs):
        width = w * unit
        card(c, x, y, width, title, lines, tone)
        x += width + CARD_GAP


def build(out_dirs=None):
    for out_dir in out_dirs or OUT_DIRS:
        c = Canvas("simrl-architecture", 1600, 1130, "How SimRL is built", out_dir)
        c.text(60, 44, "How SimRL is built", 54, INK, True)
        c.text(60, 104,
               "Profile a real system once, fit cost models to the records, then simulate on CPU.",
               31, MUTED)

        # -- Measure ---------------------------------------------------------
        y = BAND_Y[0]
        band(c, y, "1", "Measure", "srl/profiler/", "needs GPUs", "profiler")
        row(c, y + CARD_TOP, [3, 3, 5], [
            ("training", ["Megatron-Bridge", "FSDP2"]),
            ("rollout", ["vLLM server", "KV probe", "prefix probe"]),
            ("kernel", ["attention · GEMM · collective",
                        "elementwise · GDN · optimizer",
                        "shape grids and holdout split"]),
        ], "profiler")
        c.text(COL_X, y + CARD_TOP + CARD_H + 24,
               "each adapter records, none of them models", 22, MUTED)

        # -- Cost modeling -----------------------------------------------------
        y = BAND_Y[1]
        band(c, y, "2", "Cost modeling", "srl/cost/", "no GPU", "cost")
        row(c, y + CARD_TOP, [1, 1, 1], [
            ("rollout cost", ["roofline", "composed kernel sum", "DistServe · PSRL fitted"]),
            ("training cost", ["fixed · analytical · linear", "Hydraulis",
                                "FSDP latency and memory"]),
            ("kernel database", ["measured cubes", "speed-of-light", "utility surfaces"]),
        ], "cost")
        c.text(COL_X, y + CARD_TOP + CARD_H + 24,
               "each model interpolates inside its fitted domain", 22, MUTED)

        # -- Simulate ----------------------------------------------------------
        y = BAND_Y[2]
        band(c, y, "3", "Simulate", "srl/simulator/", "no GPU", "simulate")
        row(c, y + CARD_TOP, [3, 3, 3, 3], [
            ("simulator", ["discrete-event loop", "two backends", "C++ and Python"]),
            ("policy", ["10+ system presets", "L1 · L2 · L3 controls", "staleness rules"]),
            ("component", ["rollout engine", "router", "recompute · sync"]),
            ("workload", ["grouped traces", "length models"]),
        ], "simulate")
        c.text(COL_X, y + CARD_TOP + CARD_H + 24,
               "every duration comes from a cost model", 22, MUTED)

        # -- Seams -------------------------------------------------------------
        for band_top in (BAND_Y[0], BAND_Y[1]):
            top = band_top + BAND_H
            bottom = top + BAND_GAP
            label = "profile records, JSONL" if band_top == BAND_Y[0] else "cost artifacts, JSON"
            c.arrow(COL_X + 60, top + 10, COL_X + 60, bottom - 10, MUTED, 3)
            c.text(COL_X + 90, (top + bottom) / 2 - 2, label, 23, MUTED)

        c.save()


if __name__ == "__main__":
    build()
