#!/usr/bin/env python3
"""Plot showing embed_dim=128 dominates across all MAX values."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# Load data
with open("/Users/iskander/code/presto/experiments/2026-03-13_1218_claude_mhcflurry-max-dim-sweep/all_results.json") as f:
    results = json.load(f)

# Constants
MAX_VALUES = [25000, 50000, 75000, 100000, 125000, 150000]
MAX_LABELS = ["25k", "50k", "75k", "100k", "125k", "150k"]
EMBED_DIMS = [128, 256, 384, 512]

# Colors: a qualitative palette that's colorblind-friendly and distinct
COLORS = {
    128:  "#2166ac",  # strong blue
    256:  "#66bd63",  # green
    384:  "#f4a582",  # salmon/orange
    512:  "#b2182b",  # dark red
}
MARKERS = {128: "o", 256: "s", 384: "D", 512: "^"}

# Organize data: embed_dim -> {max_nM: test_spearman}
data = defaultdict(dict)
for r in results:
    data[r["embed_dim"]][r["max_nM"]] = r["test_spearman"]

# Compute group stats
group_means = {}
group_stds = {}
group_mins = {}
group_maxs = {}
group_vals = {}
for dim in EMBED_DIMS:
    vals = [data[dim][m] for m in MAX_VALUES]
    group_vals[dim] = vals
    group_means[dim] = np.mean(vals)
    group_stds[dim] = np.std(vals)
    group_mins[dim] = np.min(vals)
    group_maxs[dim] = np.max(vals)

# --- Figure setup ---
fig = plt.figure(figsize=(14, 6.5))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1.2], wspace=0.28)

ax_main = fig.add_subplot(gs[0])
ax_box = fig.add_subplot(gs[1])

# ==============================
# LEFT PANEL: Lines + shaded bands
# ==============================
x_idx = np.arange(len(MAX_VALUES))

for dim in EMBED_DIMS:
    y = [data[dim][m] for m in MAX_VALUES]
    color = COLORS[dim]
    marker = MARKERS[dim]

    # Shaded band: min-max across all MAX for this dim (horizontal band)
    ax_main.axhspan(group_mins[dim], group_maxs[dim], alpha=0.08, color=color, zorder=0)

    # Line + markers
    ax_main.plot(
        x_idx, y,
        color=color, marker=marker, markersize=9, linewidth=2.2,
        markeredgecolor="white", markeredgewidth=1.2,
        label=f"d={dim}  (mean={group_means[dim]:.4f})",
        zorder=3,
    )

    # Mean dashed line
    ax_main.axhline(
        group_means[dim], color=color, linestyle="--", linewidth=1.0, alpha=0.55, zorder=1
    )

    # Annotate mean on the right edge
    ax_main.annotate(
        f"{group_means[dim]:.4f}",
        xy=(len(MAX_VALUES) - 1 + 0.15, group_means[dim]),
        fontsize=9, fontweight="bold", color=color,
        va="center", ha="left",
    )

ax_main.set_xticks(x_idx)
ax_main.set_xticklabels(MAX_LABELS, fontsize=11)
ax_main.set_xlabel("MAX (nM ceiling)", fontsize=12, fontweight="bold")
ax_main.set_ylabel("Test Spearman", fontsize=12, fontweight="bold")
ax_main.set_title("Test Spearman vs MAX Ceiling by Encoder Dimension", fontsize=13, fontweight="bold")
ax_main.legend(
    loc="lower left", fontsize=10, framealpha=0.92, edgecolor="gray",
    title="embed_dim", title_fontsize=10,
)
ax_main.grid(axis="y", alpha=0.3, linewidth=0.5)
ax_main.set_xlim(-0.3, len(MAX_VALUES) - 1 + 0.7)

# Set y-axis to give some breathing room
all_vals = [data[d][m] for d in EMBED_DIMS for m in MAX_VALUES]
y_lo, y_hi = min(all_vals), max(all_vals)
pad = (y_hi - y_lo) * 0.08
ax_main.set_ylim(y_lo - pad, y_hi + pad)

# Add rank annotations at each x position
for xi, m in enumerate(MAX_VALUES):
    vals_at_m = [(data[dim][m], dim) for dim in EMBED_DIMS]
    vals_at_m.sort(reverse=True)
    # If ordering is strictly 128 > 256 > 384 > 512, mark with a checkmark
    order = [v[1] for v in vals_at_m]
    if order == [128, 256, 384, 512]:
        ax_main.annotate(
            "\u2713",  # checkmark
            xy=(xi, y_lo - pad * 0.05),
            fontsize=13, color="#2ca02c", ha="center", va="top",
            fontweight="bold",
        )

# ==============================
# RIGHT PANEL: Box/strip plot
# ==============================
positions = np.arange(len(EMBED_DIMS))
bp_data = [group_vals[dim] for dim in EMBED_DIMS]

bp = ax_box.boxplot(
    bp_data, positions=positions, widths=0.45,
    patch_artist=True, showmeans=True,
    meanprops=dict(marker="D", markerfacecolor="black", markeredgecolor="black", markersize=5),
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(markersize=4),
)

for patch, dim in zip(bp["boxes"], EMBED_DIMS):
    patch.set_facecolor(COLORS[dim])
    patch.set_alpha(0.55)
    patch.set_edgecolor(COLORS[dim])
    patch.set_linewidth(1.5)

# Overlay individual points (jittered)
rng = np.random.RandomState(42)
for i, dim in enumerate(EMBED_DIMS):
    y = group_vals[dim]
    jitter = rng.uniform(-0.1, 0.1, size=len(y))
    ax_box.scatter(
        positions[i] + jitter, y,
        color=COLORS[dim], edgecolor="white", s=50, zorder=5, linewidth=0.8
    )
    # Annotate mean above the box
    ax_box.annotate(
        f"{group_means[dim]:.4f}",
        xy=(positions[i], group_maxs[dim] + 0.002),
        fontsize=8.5, fontweight="bold", color=COLORS[dim],
        ha="center", va="bottom",
    )

ax_box.set_xticks(positions)
ax_box.set_xticklabels([str(d) for d in EMBED_DIMS], fontsize=11)
ax_box.set_xlabel("embed_dim", fontsize=12, fontweight="bold")
ax_box.set_ylabel("Test Spearman", fontsize=12, fontweight="bold")
ax_box.set_title("Aggregated Across\nAll MAX Values", fontsize=12, fontweight="bold")
ax_box.grid(axis="y", alpha=0.3, linewidth=0.5)
ax_box.set_ylim(ax_main.get_ylim())

# Add descending arrow annotation between boxes
for i in range(len(EMBED_DIMS) - 1):
    mid_x = (positions[i] + positions[i + 1]) / 2
    mid_y = (group_means[EMBED_DIMS[i]] + group_means[EMBED_DIMS[i + 1]]) / 2
    delta = group_means[EMBED_DIMS[i]] - group_means[EMBED_DIMS[i + 1]]
    ax_box.annotate(
        "",
        xy=(positions[i + 1] - 0.22, group_means[EMBED_DIMS[i + 1]] + 0.001),
        xytext=(positions[i] + 0.22, group_means[EMBED_DIMS[i]] - 0.001),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5),
    )

# Supertitle
fig.suptitle(
    "Encoder Dimension Dominance: Smaller is Better",
    fontsize=16, fontweight="bold", y=0.99,
)

fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.12, wspace=0.28)
out_path = "/Users/iskander/code/presto/experiments/2026-03-13_1218_claude_mhcflurry-max-dim-sweep/figures/embed_dim_dominance.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved to {out_path}")
plt.close()
