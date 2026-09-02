"""Generate ACL-style PNG figures for the paper.

Placeholder numbers live in DATA below; replace them with experimental
results, then re-run. Output goes to tmp/figures/. Reliability diagrams
are written in two styles: a line plot (original notebook style) and a
5-bin bar chart.

Usage (from the repository root):

    python tmp/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------------------------------------------------------
# DATA from results/plots (and labeled bar values). Reliability / HDP
# diagrams are not in results/; those two stay as structural placeholders.
# ---------------------------------------------------------------------------
# CoT bars: results/plots/{llama3,qwen2.5}/{mmlu-pro,race}/reasoning_compare_bars.png
#   (qwen2.5/mmlu-pro uses reasoning_compare_per_heads.png, which carries
#   the same aggregated CoT vs cropped ECE labels).
# Method trends: results/plots/{llama3,qwen2.5}/{mmlu-pro,race}/cal_heads_trends.png
#   K = 1, 3, 5, 7, 10, 15, 20, 30; series digitized from the plots and
#   pinned to the annotated *best values.
# ---------------------------------------------------------------------------

_BIN_EDGES = np.linspace(0.0, 1.0, 11)
_BIN_CENTERS = 0.5 * (_BIN_EDGES[:-1] + _BIN_EDGES[1:])

DATA = {
    "reliability_proposed": {
        # No reliability PNG in results/; placeholder for Qwen2.5 / MMLU-Pro.
        "confidence": _BIN_CENTERS.copy(),
        "accuracy": np.array(
            [0.07, 0.14, 0.23, 0.36, 0.47, 0.54, 0.67, 0.76, 0.88, 0.93]
        ),
        "counts": np.array([95, 110, 125, 140, 155, 170, 185, 175, 160, 185], dtype=float),
    },
    "reliability_beta": {
        "confidence": _BIN_CENTERS.copy(),
        "accuracy": np.array(
            [0.08, 0.18, 0.28, 0.41, 0.52, 0.58, 0.63, 0.72, 0.81, 0.90]
        ),
        "counts": np.array([8, 12, 18, 35, 90, 820, 380, 55, 22, 10], dtype=float),
    },
    "cot_compare": {
        "labels": [
            "Llama-3-8B\nMMLU-Pro",
            "Llama-3-8B\nRACE",
            "Qwen2.5-7B\nMMLU-Pro",
            "Qwen2.5-7B\nRACE",
        ],
        "cot": np.array([0.0403, 0.0404, 0.0467, 0.0427]),
        "cropped": np.array([0.0448, 0.0399, 0.0506, 0.0522]),
    },
    "method_trends": {
        "k": np.array([1, 3, 5, 7, 10, 15, 20, 30]),
        "panels": [
            {
                "row_label": "Llama-3-8B",
                "col_label": "MMLU-Pro",
                "mlp": np.array([0.0436, 0.0443, 0.0445, 0.0464, 0.0466, 0.0478, 0.0462, 0.0460]),
                "mlp_beta": np.array([0.0432, 0.0441, 0.0435, 0.0455, 0.0461, 0.0485, 0.0479, 0.0487]),
                "w_beta": np.array([0.0400, 0.0405, 0.0399, 0.0395, 0.0381, 0.0388, 0.0402, 0.0413]),
            },
            {
                "row_label": "Llama-3-8B",
                "col_label": "RACE",
                "mlp": np.array([0.0356, 0.0366, 0.0383, 0.0373, 0.0362, 0.0354, 0.0381, 0.0394]),
                "mlp_beta": np.array([0.0355, 0.0367, 0.0391, 0.0391, 0.0384, 0.0371, 0.0391, 0.0396]),
                "w_beta": np.array([0.0404, 0.0415, 0.0428, 0.0449, 0.0439, 0.0435, 0.0447, 0.0462]),
            },
            {
                "row_label": "Qwen2.5-7B",
                "col_label": "MMLU-Pro",
                "mlp": np.array([0.0512, 0.0527, 0.0563, 0.0540, 0.0520, 0.0532, 0.0504, 0.0493]),
                "mlp_beta": np.array([0.0532, 0.0554, 0.0577, 0.0553, 0.0529, 0.0527, 0.0478, 0.0462]),
                "w_beta": np.array([0.0506, 0.0511, 0.0514, 0.0501, 0.0471, 0.0443, 0.0420, 0.0414]),
            },
            {
                "row_label": "Qwen2.5-7B",
                "col_label": "RACE",
                "mlp": np.array([0.0474, 0.0474, 0.0473, 0.0470, 0.0472, 0.0477, 0.0482, 0.0482]),
                "mlp_beta": np.array([0.0436, 0.0439, 0.0447, 0.0445, 0.0447, 0.0474, 0.0481, 0.0493]),
                "w_beta": np.array([0.0477, 0.0471, 0.0465, 0.0467, 0.0469, 0.0481, 0.0491, 0.0504]),
            },
        ],
    },
    "hdp_heatmap": {
        # Shape (n_layers, n_heads). Layer 0 at the bottom (origin="lower").
        "values": None,  # filled below with a reproducible placeholder
        "n_layers": 28,
        "n_heads": 28,
    },
}


def _placeholder_hdp(n_layers: int, n_heads: int, seed: int = 42) -> np.ndarray:
    """Signed HDP matrix: weak noise plus a few concentrated heads."""
    rng = np.random.default_rng(seed)
    hdp = rng.normal(0.0, 0.012, size=(n_layers, n_heads))
    hotspots = [
        (22, 5, 0.18),
        (18, 14, 0.15),
        (25, 3, 0.13),
        (21, 6, 0.11),
        (15, 7, 0.10),
        (26, 11, 0.09),
        (8, 20, -0.09),
        (4, 12, -0.07),
        (12, 2, 0.08),
        (23, 19, 0.07),
    ]
    for layer, head, value in hotspots:
        hdp[layer, head] = value
        for dl, dh in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            ll, hh = layer + dl, head + dh
            if 0 <= ll < n_layers and 0 <= hh < n_heads:
                hdp[ll, hh] += 0.35 * value
    return hdp


DATA["hdp_heatmap"]["values"] = _placeholder_hdp(
    DATA["hdp_heatmap"]["n_layers"],
    DATA["hdp_heatmap"]["n_heads"],
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# Okabe–Ito (colorblind-safe).
OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

DPI = 300
OUT_DIR = Path(__file__).resolve().parent / "figures"

HEAD_STYLE = {
    "mlp": {"color": OI["blue"], "marker": "o", "label": "MLP"},
    "mlp_beta": {"color": OI["orange"], "marker": "s", "label": "MLP+Beta"},
    "w_beta": {"color": OI["purple"], "marker": "D", "label": "Weighted-Beta"},
}


def apply_acl_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "TeX Gyre Termes",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.labelpad": 6,
            "xtick.major.pad": 4,
            "ytick.major.pad": 4,
            "lines.linewidth": 1.25,
            "lines.markersize": 5.0,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": DPI,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def restyle_axes(ax: plt.Axes, *, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(width=0.6, length=3)
    if grid == "y":
        ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="#bbbbbb", zorder=0)
        ax.set_axisbelow(True)
    elif grid == "both":
        ax.grid(True, linestyle=":", linewidth=0.4, color="#bbbbbb", zorder=0)
        ax.set_axisbelow(True)


def save_fig(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.4, facecolor="white")
    plt.close(fig)
    return path


def ece_from_bins(confidence: np.ndarray, accuracy: np.ndarray, counts: np.ndarray) -> float:
    mask = counts > 0
    total = counts[mask].sum()
    if total <= 0:
        return float("nan")
    return float(np.sum(np.abs(accuracy[mask] - confidence[mask]) * (counts[mask] / total)))


def bin_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 1:
        half = min(centers[0], 1.0 - centers[0], 0.05)
        return np.array([max(0.0, centers[0] - half), min(1.0, centers[0] + half)])
    half = 0.5 * (centers[1] - centers[0])
    edges = np.empty(len(centers) + 1)
    edges[0] = max(0.0, centers[0] - half)
    edges[-1] = min(1.0, centers[-1] + half)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    return edges


def coarsen_bins(entry: dict, n_bins: int = 5) -> dict:
    """Merge equal-width bins into ``n_bins`` groups (weighted by counts)."""
    conf = np.asarray(entry["confidence"], dtype=float)
    acc = np.asarray(entry["accuracy"], dtype=float)
    counts = np.asarray(entry["counts"], dtype=float)
    n = len(conf)
    group = max(1, n // n_bins)
    new_conf, new_acc, new_counts = [], [], []
    for i in range(n_bins):
        start = i * group
        stop = n if i == n_bins - 1 else (i + 1) * group
        sl = slice(start, stop)
        w = counts[sl]
        wsum = w.sum()
        new_counts.append(wsum)
        new_conf.append(np.average(conf[sl], weights=w) if wsum > 0 else conf[sl].mean())
        new_acc.append(np.average(acc[sl], weights=w) if wsum > 0 else acc[sl].mean())
    return {
        "confidence": np.asarray(new_conf),
        "accuracy": np.asarray(new_acc),
        "counts": np.asarray(new_counts),
    }


def _ece_box(ax: plt.Axes, ece: float) -> None:
    ax.text(
        0.97,
        0.06,
        f"ECE = {ece:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "linewidth": 0.5,
        },
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_reliability_line(entry: dict, out_name: str, color: str) -> Path:
    """Line reliability diagram (adaptive-bin coverage via fill), original notebook style."""
    confidence = np.asarray(entry["confidence"], dtype=float)
    accuracy = np.asarray(entry["accuracy"], dtype=float)
    counts = np.asarray(entry["counts"], dtype=float)
    edges = bin_edges_from_centers(confidence)
    mask = counts > 0
    ece = ece_from_bins(confidence, accuracy, counts)

    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#888888",
        linewidth=0.9,
        zorder=1,
        label="Perfect calibration",
    )
    for i in np.flatnonzero(mask):
        ax.fill_between(
            [edges[i], edges[i + 1]],
            0.0,
            accuracy[i],
            color=color,
            alpha=0.22,
            linewidth=0,
            zorder=2,
        )
    ax.plot(
        confidence[mask],
        accuracy[mask],
        color=color,
        marker="o",
        linewidth=1.6,
        markersize=5.5,
        zorder=3,
        clip_on=False,
        label="Model",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    restyle_axes(ax, grid="both")
    ax.legend(loc="upper left", handlelength=1.4, borderaxespad=0.3)
    _ece_box(ax, ece)
    return save_fig(fig, out_name)


def plot_reliability_bars(entry: dict, out_name: str, bar_color: str) -> Path:
    """Guo-style bars with a confidence histogram; intended for few (e.g. 5) bins."""
    confidence = np.asarray(entry["confidence"], dtype=float)
    accuracy = np.asarray(entry["accuracy"], dtype=float)
    counts = np.asarray(entry["counts"], dtype=float)
    ece = ece_from_bins(confidence, accuracy, counts)
    n = max(len(confidence), 1)
    width = 0.72 / n

    fig = plt.figure(figsize=(3.3, 3.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1], sharex=ax)

    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=0.9, zorder=1)
    mask = counts > 0
    ax.bar(
        confidence[mask],
        accuracy[mask],
        width=width,
        color=bar_color,
        edgecolor=OI["black"],
        linewidth=0.4,
        zorder=3,
        label="Accuracy",
    )
    gap_bottom = np.minimum(accuracy[mask], confidence[mask])
    gap_height = np.abs(accuracy[mask] - confidence[mask])
    ax.bar(
        confidence[mask],
        gap_height,
        bottom=gap_bottom,
        width=width,
        color=OI["vermillion"],
        alpha=0.40,
        edgecolor=OI["vermillion"],
        linewidth=0.3,
        zorder=2,
        label="Gap",
    )

    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.02, 1.04)
    ax.set_ylabel("Accuracy")
    ax.tick_params(labelbottom=False)
    restyle_axes(ax, grid="both")
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.legend(loc="upper left", handlelength=1.2, borderaxespad=0.3)
    _ece_box(ax, ece)

    ax_hist.bar(
        confidence[mask],
        counts[mask],
        width=width,
        color=bar_color,
        edgecolor=OI["black"],
        linewidth=0.4,
        zorder=3,
    )
    ax_hist.set_xlim(-0.04, 1.04)
    ax_hist.set_xlabel("Confidence")
    ax_hist.set_ylabel("Count")
    ax_hist.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    restyle_axes(ax_hist, grid="y")
    return save_fig(fig, out_name)


def plot_cot_compare(entry: dict) -> Path:
    labels = entry["labels"]
    cot = np.asarray(entry["cot"], dtype=float)
    cropped = np.asarray(entry["cropped"], dtype=float)
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    ax.bar(
        x - width / 2,
        cot,
        width,
        color=OI["green"],
        edgecolor=OI["black"],
        linewidth=0.4,
        label="CoT",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        cropped,
        width,
        color=OI["sky"],
        edgecolor=OI["black"],
        linewidth=0.4,
        label="Cropped",
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ECE")
    ax.set_xlim(-0.7, len(labels) - 0.3)
    ymax = max(cot.max(), cropped.max())
    ax.set_ylim(0.0, ymax * 1.22)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    restyle_axes(ax, grid="y")
    ax.legend(loc="upper right", ncol=2, handlelength=1.3)
    return save_fig(fig, "cot-compare.png")


def plot_method_trends(entry: dict) -> Path:
    k = np.asarray(entry["k"], dtype=float)
    panels = entry["panels"]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.0), sharex=True)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.84, bottom=0.12, wspace=0.30, hspace=0.32)

    for idx, panel in enumerate(panels):
        ax = axes[idx // 2, idx % 2]
        series = []
        for key in ("mlp", "mlp_beta", "w_beta"):
            style = HEAD_STYLE[key]
            y = np.asarray(panel[key], dtype=float)
            ax.plot(
                k,
                y,
                color=style["color"],
                marker=style["marker"],
                label=style["label"],
                zorder=3,
            )
            series.append(y)

        stacked = np.vstack(series)
        min_val = stacked.min()
        min_flat = int(np.argmin(stacked))
        _, min_k_idx = divmod(min_flat, stacked.shape[1])
        ax.scatter(
            [k[min_k_idx]],
            [min_val],
            marker="*",
            s=90,
            c=OI["vermillion"],
            zorder=5,
            linewidths=0.3,
            edgecolors=OI["black"],
            clip_on=False,
            label="Minimum" if idx == 0 else None,
        )

        restyle_axes(ax, grid="y")
        ax.set_xlim(-1.5, max(k) + 3)
        pad = 0.10 * (stacked.max() - stacked.min() + 1e-6)
        ax.set_ylim(stacked.min() - pad, stacked.max() + pad)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))

        if idx // 2 == 0:
            ax.set_title(panel["col_label"], loc="center", pad=4)
        if idx % 2 == 0:
            ax.set_ylabel("ECE")
        if idx // 2 == 1:
            ax.set_xlabel(r"Number of heads $K$")

    fig.text(0.03, 0.66, "Llama-3-8B", rotation=90, va="center", ha="center", fontsize=9)
    fig.text(0.03, 0.30, "Qwen2.5-7B", rotation=90, va="center", ha="center", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.55, 0.99),
        handlelength=1.6,
        columnspacing=1.4,
    )
    return save_fig(fig, "method-trends.png")


def plot_hdp_heatmap(entry: dict) -> Path:
    values = np.asarray(entry["values"], dtype=float)
    n_layers, n_heads = values.shape

    fig, ax = plt.subplots(figsize=(3.3, 2.8))
    vmax = float(np.max(np.abs(values))) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
    )
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xlim(-0.5, n_heads - 0.5)
    ax.set_ylim(-0.5, n_layers - 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.tick_params(width=0.6, length=3)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("HDP", labelpad=8)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(width=0.6, length=2.5, labelsize=7, pad=3)
    return save_fig(fig, "hdp-heatmap.png")


def main() -> None:
    apply_acl_style()
    proposed = DATA["reliability_proposed"]
    beta = DATA["reliability_beta"]
    written = [
        plot_reliability_line(proposed, "reliability-proposed-mmlu.png", OI["blue"]),
        plot_reliability_line(beta, "reliability-beta-mmlu.png", OI["orange"]),
        plot_reliability_bars(
            coarsen_bins(proposed, n_bins=5),
            "reliability-proposed-mmlu-bars.png",
            OI["blue"],
        ),
        plot_reliability_bars(
            coarsen_bins(beta, n_bins=5),
            "reliability-beta-mmlu-bars.png",
            OI["orange"],
        ),
        plot_cot_compare(DATA["cot_compare"]),
        plot_method_trends(DATA["method_trends"]),
        plot_hdp_heatmap(DATA["hdp_heatmap"]),
    ]
    print("Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
