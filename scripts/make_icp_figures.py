#!/usr/bin/env python3
"""Generate publication-style schematic figures for the ICP paper."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"


def add_box(ax, xy, wh, title, subtitle="", fc="#f7f9fb", ec="#233142"):
    rect = Rectangle(xy, wh[0], wh[1], linewidth=1.4, edgecolor=ec, facecolor=fc)
    ax.add_patch(rect)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] * 0.62, title,
            ha="center", va="center", fontsize=10, fontweight="bold", color="#17212b")
    if subtitle:
        ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] * 0.34, subtitle,
                ha="center", va="center", fontsize=8.2, color="#465362")


def add_arrow(ax, start, end, label=""):
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                            linewidth=1.2, color="#233142")
    ax.add_patch(arrow)
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.12,
                label, ha="center", va="center", fontsize=7.8, color="#465362")


def framework():
    fig, ax = plt.subplots(figsize=(10.2, 3.7), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        ((0.25, 1.05), (1.35, 0.85), "Input window", "multivariate signals"),
        ((2.05, 1.05), (1.35, 0.85), "Encoder", "latent state z"),
        ((3.85, 1.05), (1.55, 0.85), "Concept layer", "grounded states c"),
        ((6.0, 1.05), (1.65, 0.85), "FAN block", "fuzzy weights alpha"),
        ((8.25, 1.05), (1.45, 0.85), "Decision", "risk / anomaly"),
    ]
    colors = ["#eef4f8", "#f7f9fb", "#eef8f3", "#fff5e8", "#f4eef8"]
    for (xy, wh, title, subtitle), color in zip(boxes, colors):
        add_box(ax, xy, wh, title, subtitle, fc=color)

    add_arrow(ax, (1.6, 1.48), (2.05, 1.48))
    add_arrow(ax, (3.4, 1.48), (3.85, 1.48))
    add_arrow(ax, (5.4, 1.48), (6.0, 1.48))
    add_arrow(ax, (7.65, 1.48), (8.25, 1.48))

    add_box(ax, (3.75, 0.18), (1.75, 0.48), "concept targets", "heuristic / expert rules",
            fc="#ffffff", ec="#73808c")
    add_arrow(ax, (4.62, 0.66), (4.62, 1.05), "supervision")

    ax.text(0.25, 2.58, "Concept-mediated decision pathway",
            fontsize=12, fontweight="bold", color="#17212b")
    ax.text(0.25, 2.34, "Prediction is forced through interpretable concept evidence before aggregation.",
            fontsize=8.5, color="#465362")

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "framework_architecture.png", bbox_inches="tight")
    plt.close(fig)


def interpretability():
    fig, ax = plt.subplots(figsize=(8.6, 4.3), dpi=300)
    ax.set_xlim(0, 8.6)
    ax.set_ylim(0, 4.1)
    ax.axis("off")

    add_box(ax, (0.35, 2.55), (1.65, 0.75), "Concepts", "c1 ... cK", fc="#eef8f3")
    add_box(ax, (0.35, 1.45), (1.65, 0.75), "Membership", "mu(c)", fc="#fff5e8")
    add_box(ax, (2.75, 2.0), (1.65, 0.75), "Attention", "alpha", fc="#f4eef8")
    add_box(ax, (5.15, 2.0), (1.85, 0.75), "Contribution", "alpha * evidence", fc="#eef4f8")

    add_arrow(ax, (2.0, 2.92), (2.75, 2.48))
    add_arrow(ax, (2.0, 1.82), (2.75, 2.22))
    add_arrow(ax, (4.4, 2.38), (5.15, 2.38))

    names = ["tank level", "flow", "pressure", "actuator-sensor"]
    vals = [0.92, 0.24, 0.08, 0.36]
    x0, y0 = 0.55, 0.55
    for i, (name, val) in enumerate(zip(names, vals)):
        y = y0 + i * 0.28
        ax.text(x0, y, name, fontsize=7.8, ha="left", va="center", color="#17212b")
        ax.add_patch(Rectangle((2.0, y - 0.055), 2.1, 0.11, facecolor="#dfe8ef", edgecolor="none"))
        ax.add_patch(Rectangle((2.0, y - 0.055), 2.1 * val, 0.11, facecolor="#3a6ea5", edgecolor="none"))
        ax.text(4.25, y, f"{val:.2f}", fontsize=7.5, ha="left", va="center", color="#465362")

    ax.text(0.35, 3.72, "Internal explanation object",
            fontsize=12, fontweight="bold", color="#17212b")
    ax.text(0.35, 3.48, "The explanation is the ranked set of concept contributions, not a post-hoc surrogate.",
            fontsize=8.5, color="#465362")
    ax.text(5.15, 1.22, "Top evidence is removed or inserted\nat inference time for faithfulness tests.",
            fontsize=8.3, color="#465362", ha="left")

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "interpretability_scheme.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    framework()
    interpretability()


if __name__ == "__main__":
    main()
