"""Vision: decodes real PNG pixels and derives a structured visual percept.

Works on 8-bit PNGs (gray / RGB / palette / gray+alpha / RGBA) using the
pure-stdlib decoder in ``imaging.py``.  Produces object blobs, palette,
brightness, edge density, and a natural-language description.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ..core.stimulus import Stimulus
from ..util import clamp
from .base import Capability, VisionPercept
from .imaging import Rows, decode_png, is_png

NAMED_COLORS: List[Tuple[str, int, int, int]] = [
    ("black", 0, 0, 0), ("white", 255, 255, 255), ("red", 220, 40, 40),
    ("green", 60, 180, 75), ("blue", 50, 90, 200), ("yellow", 240, 220, 60),
    ("orange", 240, 140, 40), ("purple", 140, 60, 180), ("pink", 240, 140, 180),
    ("brown", 130, 90, 50), ("gray", 128, 128, 128), ("cyan", 70, 200, 200),
]

GRID_W, GRID_H = 48, 36


def color_name(r: int, g: int, b: int) -> str:
    if max(r, g, b) - min(r, g, b) < 24:
        lum = (r + g + b) / 3
        if lum < 60:
            return "black"
        if lum > 200:
            return "white"
        return "gray"
    best, best_d = "gray", 1 << 30
    for name, cr, cg, cb in NAMED_COLORS:
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_d:
            best, best_d = name, d
    return best


def _position(cx: float, cy: float) -> str:
    vert = "top" if cy < 0.37 else "bottom" if cy > 0.63 else "middle"
    horiz = "left" if cx < 0.37 else "right" if cx > 0.63 else "center"
    if vert == "middle" and horiz == "center":
        return "center"
    return f"{vert}-{horiz}" if horiz != "center" else vert


class VisionCapability(Capability):
    name = "vision"
    description = "Sees PNG images: colors, brightness, regions, object blobs."
    phase = "perceive"

    def can_handle(self, stimulus: Stimulus, ctx=None) -> bool:
        return stimulus.data is not None and stimulus.modality.value == "image"

    def handle(self, stimulus: Stimulus, ctx=None) -> VisionPercept | None:
        data = stimulus.data or b""
        if not is_png(data):
            raise ValueError("vision currently supports 8-bit PNG images only")
        width, height, rows = decode_png(data)
        buckets = self._downsample(rows, width, height)

        brightness, palette = self._stats(buckets)
        edges = self._edge_density(buckets)
        objects = self._find_blobs(buckets)

        obj_txt = ", ".join(
            f"a {o['color']} shape at the {o['position']} ({int(o['size']*100)}% of frame)"
            for o in objects[:3]) or "no distinct shapes"
        bright_txt = ("bright" if brightness > 0.66 else "dim" if brightness < 0.33
                      else "moderately lit")
        description = (f"I see a {width}x{height} image, {bright_txt}, "
                       f"dominated by {' and '.join(palette[:2])}; "
                       f"I notice {obj_txt}.")

        return VisionPercept(
            description=description,
            confidence=0.8,
            width=width, height=height,
            objects=objects, palette=palette,
            brightness=round(brightness, 3),
            edge_density=round(edges, 3),
            attributes={"objects": objects, "palette": palette},
        )

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _downsample(rows: Rows, width: int, height: int) -> List[List[Tuple[int, int, int]]]:
        grid: List[List[Tuple[int, int, int]]] = []
        for gy in range(GRID_H):
            row = []
            for gx in range(GRID_W):
                x = min(width - 1, int(gx * width / GRID_W))
                y = min(height - 1, int(gy * height / GRID_H))
                row.append(rows[y][x])
            grid.append(row)
        return grid

    @staticmethod
    def _stats(grid) -> Tuple[float, List[str]]:
        total = 0.0
        color_counts: Dict[str, int] = {}
        for row in grid:
            for (r, g, b) in row:
                total += (r + g + b) / 765.0
                name = color_name(r, g, b)
                color_counts[name] = color_counts.get(name, 0) + 1
        n = len(grid) * len(grid[0])
        brightness = total / n
        palette = [c for c, _ in sorted(color_counts.items(), key=lambda kv: -kv[1])[:3]]
        return brightness, palette

    @staticmethod
    def _edge_density(grid) -> float:
        edges = 0
        h, w = len(grid), len(grid[0])
        for y in range(h):
            for x in range(1, w):
                if color_name(*grid[y][x - 1]) != color_name(*grid[y][x]):
                    edges += 1
        for y in range(1, h):
            for x in range(w):
                if color_name(*grid[y - 1][x]) != color_name(*grid[y][x]):
                    edges += 1
        return clamp(edges / (2 * h * w), 0.0, 1.0)

    def _find_blobs(self, grid) -> List[Dict]:
        h, w = len(grid), len(grid[0])
        labels = [[color_name(*grid[y][x]) for x in range(w)] for y in range(h)]
        visited = [[False] * w for _ in range(h)]
        blobs: List[Dict] = []
        for sy in range(h):
            for sx in range(w):
                if visited[sy][sx]:
                    continue
                target = labels[sy][sx]
                stack = [(sy, sx)]
                visited[sy][sx] = True
                cells: List[Tuple[int, int]] = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for ny, nx in ((y+1, x), (y-1, x), (y, x+1), (y, x-1)):
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny][nx] \
                                and labels[ny][nx] == target:
                            visited[ny][nx] = True
                            stack.append((ny, nx))
                size = len(cells) / (h * w)
                if target in ("white", "black") or size < 0.015 or size > 0.5:
                    continue          # tiny specks and the background aren't objects
                ys = [c[0] for c in cells]
                xs = [c[1] for c in cells]
                cy = (min(ys) + max(ys)) / 2 / h
                cx = (min(xs) + max(xs)) / 2 / w
                bbox_area = ((max(ys) - min(ys) + 1) * (max(xs) - min(xs) + 1)) / (h * w)
                blobs.append({
                    "color": target, "position": _position(cx, cy),
                    "size": round(size, 3),
                    "compactness": round(size / bbox_area, 2) if bbox_area else 0.0,
                })
        blobs.sort(key=lambda b: -b["size"])
        return blobs[:6]
