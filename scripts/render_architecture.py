"""Render the policy-value network as a stacked-volume / ResNet-skip figure.

Produces the "feature-map volumes + residual skip arcs" look (the standard
publication style) as a crisp, font-independent SVG for the Pages app, plus a
PNG for visual review. Labels are English + numbers so the output does not
depend on a CJK font being installed; the Chinese explanation lives in the page
caption, not in the figure.

Usage:
  python scripts/render_architecture.py --channels 128 --blocks 8 \
    --policy-channels 12 --value-hidden 384 --out-stem docs/assets/diagrams/arch-v3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Embed text as vector paths so the SVG renders identically everywhere with no
# font dependency (the build host has no CJK fonts; labels are ASCII anyway).
matplotlib.rcParams["svg.fonttype"] = "path"
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


def _shade(hex_color: str, factor: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = (max(0, min(255, int(c * factor))) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def volume_hw(sq, n, dx):
    """Half-width (and half-height) of a volume's bounding box."""
    return (sq + (n - 1) * dx) / 2


def draw_volume(ax, cx, cy, color, *, sq=8.0, n=7, dx=0.9, dy=0.9, z=2):
    """A stack of offset squares → a 3D feature-map volume (back-to-front)."""
    edge = _shade(color, 0.62)
    hw = volume_hw(sq, n, dx)
    for i in range(n - 1, -1, -1):
        x = cx - hw + i * dx
        y = cy - sq / 2 + i * dy
        ax.add_patch(
            Rectangle(
                (x, y), sq, sq,
                facecolor=color, edgecolor=edge, linewidth=1.1,
                zorder=z + (n - i) * 0.01,
            )
        )


def label(ax, cx, y, title, shape, color):
    ax.text(cx, y, title, ha="center", va="top", fontsize=12.5,
            fontweight="bold", color="#1b2330")
    if shape:
        ax.text(cx, y - 3.0, shape, ha="center", va="top", fontsize=10.5,
                color=color, fontweight="bold", family="monospace")


def flow_arrow(ax, x1, y1, x2, y2, color="#8a96a3", lw=2.2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
        lw=lw, color=color, shrinkA=0, shrinkB=0, zorder=5,
    ))


def skip_arc(ax, x1, x2, y, color="#8a96a3", lw=2.2, rad=-0.55):
    ax.add_patch(FancyArrowPatch(
        (x1, y), (x2, y), connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=14, lw=lw, color=color,
        shrinkA=2, shrinkB=2, zorder=4,
    ))


def plus_node(ax, cx, cy, r=1.7):
    ax.add_patch(Circle((cx, cy), r, facecolor="#2bb7c4",
                        edgecolor="#1b7e88", linewidth=1.3, zorder=6))
    ax.text(cx, cy - 0.05, "+", ha="center", va="center", color="white",
            fontsize=13, fontweight="bold", zorder=7)


def _lerp(c1, c2, t):
    a, b = c1.lstrip("#"), c2.lstrip("#")
    ch = [int(a[i : i + 2], 16) + (int(b[i : i + 2], 16) - int(a[i : i + 2], 16)) * t
          for i in (0, 2, 4)]
    return "#" + "".join(f"{max(0, min(255, round(v))):02x}" for v in ch)


def tower_color(t):
    """Gradient coral -> teal -> slate across the residual tower."""
    return _lerp("#ec6248", "#45a6a0", t * 2) if t < 0.5 \
        else _lerp("#45a6a0", "#586a82", (t - 0.5) * 2)


def render(cfg: dict, out_svg: Path, png: Path | None = None) -> None:
    channels = cfg["channels"]
    blocks = cfg["blocks"]

    # Make the GEOMETRY carry the architecture, so models look different at a
    # glance: channels -> feature-map slices (volume thickness), blocks -> the
    # actual number of residual blocks drawn. v1 (192x12) is thus a visibly
    # longer, chunkier tower than v3/v4 (128x8).
    depth = max(4, round(channels / 20))   # slices in the main volumes
    bdepth = max(3, round(channels / 30))  # slices per residual block
    main_y = 22.0
    amber, pol_c, val_c = "#f2b84b", "#1f8a8c", "#c0492f"

    sq, dx = 8.5, 0.85
    hw = volume_hw(sq, depth, dx)
    bsq, bdx = 6.4, 0.7
    bhw = volume_hw(bsq, bdepth, bdx)
    pitch = bhw * 2 + 3.6                  # residual-block centre spacing

    # x layout — computed so the canvas grows with width (channels) and depth
    # (blocks); each model renders to its own size.
    in_cx = 9.0
    conv_cx = in_cx + 11 + hw
    block_cx = [conv_cx + hw + 7 + bhw + i * pitch for i in range(blocks)]
    last_plus = block_cx[-1] + bhw + 2.4
    hh = volume_hw(6.5, 5, 0.8)            # head half-width
    head_cx = last_plus + 13 + hh
    W = head_cx + hh + 22
    H = 44.0

    fig, ax = plt.subplots(figsize=(W * 0.118, H * 0.118), dpi=150)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # Input board (two planes) ------------------------------------------------
    bx, by = in_cx - 3.1, main_y - 5.9  # front (board) plane origin
    ax.add_patch(Rectangle((bx - 1.4, by + 1.4), 9, 9, facecolor="#1b1f24",
                           edgecolor="#0d0f12", linewidth=1.2, zorder=2))
    ax.add_patch(Rectangle((bx, by), 9, 9, facecolor="#c98f4a",
                           edgecolor="#0d0f12", linewidth=1.2, zorder=3))
    for g in range(1, 5):  # faint grid hint on the front plane
        ax.plot([bx + g * 1.8, bx + g * 1.8], [by, by + 9], color="#8a5a25",
                lw=0.5, zorder=4)
        ax.plot([bx, bx + 9], [by + g * 1.8, by + g * 1.8], color="#8a5a25",
                lw=0.5, zorder=4)
    label(ax, in_cx, main_y - 9.5, "Board", "2x10x10", "#677184")

    # Stem conv ---------------------------------------------------------------
    draw_volume(ax, conv_cx, main_y, amber, sq=sq, n=depth, dx=dx, dy=dx)
    label(ax, conv_cx, main_y - 9.5, "Conv 3x3", f"{channels}x10x10", "#9a7a2a")

    # Residual tower: one stacked-volume block per real residual block --------
    node_x = conv_cx + hw
    flow_arrow(ax, node_x, main_y, block_cx[0] - bhw - 0.4, main_y)
    for i, cx in enumerate(block_cx):
        left, right = cx - bhw, cx + bhw
        plus_cx = right + 2.4
        col = tower_color(i / max(1, blocks - 1))
        skip_arc(ax, left - 1.0, plus_cx, main_y + 2.4, rad=-0.6, lw=1.8)
        draw_volume(ax, cx, main_y, col, sq=bsq, n=bdepth, dx=bdx, dy=bdx)
        flow_arrow(ax, right, main_y, plus_cx - 1.6, main_y, lw=1.7)
        plus_node(ax, plus_cx, main_y, r=1.5)
        if i < blocks - 1:
            flow_arrow(ax, plus_cx + 1.6, main_y, block_cx[i + 1] - bhw - 0.4, main_y, lw=1.7)
        node_x = plus_cx + 1.6
    mid = (block_cx[0] + block_cx[-1]) / 2
    ax.text(mid, main_y + bhw + 7.0, "skip connection every block", ha="center",
            va="center", fontsize=10.5, color="#8a96a3", fontstyle="italic")
    label(ax, mid, main_y - 9.5, f"Residual  x{blocks}", f"{channels} ch", "#5a6573")

    # Split into the two heads ------------------------------------------------
    pol_y, val_y = main_y + 8.0, main_y - 8.0
    flow_arrow(ax, node_x, main_y, head_cx - hh - 0.4, pol_y, color=pol_c)
    flow_arrow(ax, node_x, main_y, head_cx - hh - 0.4, val_y, color=val_c)
    draw_volume(ax, head_cx, pol_y, pol_c, sq=6.5, n=5, dx=0.8, dy=0.8)
    draw_volume(ax, head_cx, val_y, val_c, sq=6.5, n=5, dx=0.8, dy=0.8)
    tx = head_cx + hh + 2.5
    ax.text(tx, pol_y + 1.0, "Policy", ha="left", va="center", fontsize=12.5,
            fontweight="bold", color=pol_c)
    ax.text(tx, pol_y - 2.6, "-> 100 moves", ha="left", va="center",
            fontsize=10, color=pol_c, family="monospace")
    ax.text(tx, val_y + 1.0, "Value", ha="left", va="center", fontsize=12.5,
            fontweight="bold", color=val_c)
    ax.text(tx, val_y - 2.6, "-> [-1, 1]", ha="left", va="center",
            fontsize=10, color=val_c, family="monospace")

    # Legend ------------------------------------------------------------------
    plus_node(ax, in_cx, 3.4, r=1.5)
    ax.text(in_cx + 3.0, 3.4, "element-wise addition (residual skip)", ha="left",
            va="center", fontsize=10, color="#677184")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg)
    if png is not None:
        fig.savefig(png)
    plt.close(fig)
    print(f"wrote {out_svg}" + (f" and {png}" if png else ""))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--policy-channels", type=int, default=12)
    p.add_argument("--value-hidden", type=int, default=384)
    p.add_argument("--out", type=Path, required=True, help="output .svg path")
    p.add_argument("--png", type=Path, default=None, help="optional PNG for review")
    a = p.parse_args()
    render(
        {
            "channels": a.channels,
            "blocks": a.blocks,
            "policy_channels": a.policy_channels,
            "value_hidden": a.value_hidden,
        },
        a.out,
        a.png,
    )


if __name__ == "__main__":
    main()
