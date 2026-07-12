#!/usr/bin/env python3
"""Plot BS=1 decode-only GPU bubble ratio for eager vs CUDA Graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = [
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-30B-A3B",
    "Qwen3.5-27B",
]

# Decode-only BS=1, prompt=16, decode=512 values from blog table.
EAGER_BUBBLE = np.array([84.7, 76.3, 45.3, 28.9, 88.1, 54.8])
CUDAGRAPH_BUBBLE = np.array([14.0, 8.6, 2.4, 2.0, 7.8, 1.6])


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root() / "assets/figs/fig1_decode_bs1_bubble.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        default="BS=1 Decode: GPU Bubble Ratio",
        help="Figure title.",
    )
    parser.add_argument(
        "--xlabel",
        default="Model (BS=1, prompt=16, decode=512)",
        help="X-axis label.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI.")
    parser.add_argument("--width", type=float, default=18.34, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=10.66, help="Figure height in inches.")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively.")
    return parser.parse_args()


def apply_style(dpi: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 24,
            "axes.titlesize": 30,
            "axes.titleweight": "bold",
            "axes.labelsize": 24,
            "legend.fontsize": 21,
            "figure.dpi": 100,
            "savefig.dpi": dpi,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "-",
        }
    )


def add_value_labels(ax: plt.Axes, bars) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.3,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=19,
            color="#333333",
        )


def plot(args: argparse.Namespace) -> None:
    apply_style(args.dpi)

    x = np.arange(len(MODELS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(args.width, args.height))

    eager_bars = ax.bar(x - width / 2, EAGER_BUBBLE, width, label="eager", color="#D55E00")
    cg_bars = ax.bar(
        x + width / 2,
        CUDAGRAPH_BUBBLE,
        width,
        label="cudagraph",
        color="#0072B2",
    )

    ax.set_title(args.title, pad=18)
    ax.set_ylabel("GPU bubble ratio (%)")
    ax.set_xlabel(args.xlabel, labelpad=18)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=22, ha="right")
    ax.set_ylim(0, 104)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y")
    ax.grid(axis="x", alpha=0.12)

    add_value_labels(ax, eager_bars)
    add_value_labels(ax, cg_bars)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    if args.show:
        plt.show()
    print(args.output)


def main() -> None:
    plot(parse_args())


if __name__ == "__main__":
    main()
