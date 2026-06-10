import argparse
import json
import os
import math
import random
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import seaborn as sns
import pandas as pd


FULL_CANVAS_CMAP = ListedColormap(
    [
        "#d8dbe2",  # outside this target canvas
        "#ffffff",  # unfilled
        "#4c78a8",  # generated ordinary token
        "#f28e2b",  # generated after EOS
        "#7b61ff",  # generated special token
        "#d62728",  # generated EOS
    ]
)
FULL_CANVAS_NORM = BoundaryNorm(
    [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
    FULL_CANVAS_CMAP.N,
)

SPECIAL_MARKERS = {
    "<EOS>": ("*", "#d62728"),
    "<PAD>": ("s", "#6b7280"),
    "<UTR3>": ("D", "#2ca02c"),
    "<UTR5>": ("^", "#17becf"),
    "<CDS>": ("P", "#8c564b"),
    "<BOS>": ("o", "#9467bd"),
    "<MASK>": ("X", "#e377c2"),
}

POST_EOS_COLUMNS = [
    "id",
    "eos_position",
    "eos_generation_step",
    "position",
    "position_after_eos",
    "generation_step",
    "generated_after_eos_token",
    "token_id",
    "token_label",
    "is_special",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--resolution", type=int, default=100)
    return parser.parse_args()


def infer_order_format(item):
    order_format = item.get("order_format")
    if order_format:
        return order_format

    model_type = str(item.get("model_type", "")).lower().replace("-", "")
    if model_type == "loarm":
        return "step_to_position"
    return "position_to_step"


def normalize_generation_order(item):
    raw_order = [int(position) for position in item["generation_order"]]
    seq_len = int(item.get("sequence_length", len(raw_order)))
    seq_len = max(0, min(seq_len, len(raw_order)))
    if seq_len == 0:
        return [], 0

    order_format = infer_order_format(item)
    if order_format == "position_to_step":
        step_to_position = [None] * seq_len
        for position, step in enumerate(raw_order[:seq_len]):
            if 0 <= step < seq_len and step_to_position[step] is None:
                step_to_position[step] = position
        if any(position is None for position in step_to_position):
            return [], 0
        order = step_to_position
    elif order_format == "step_to_position":
        order = raw_order[:seq_len]
    else:
        raise ValueError(f"Unsupported order_format={order_format!r}")

    if any(position < 0 or position >= seq_len for position in order):
        return [], 0
    return order, seq_len


def get_utr5_token_length(item, utr5_lengths):
    value = item.get("utr5_token_length")
    if value is not None:
        return int(value)
    return utr5_lengths.get(str(item["id"]))


def normalize_full_generation_order(item):
    raw_order = item.get("full_generation_order")
    if raw_order is None:
        return [], 0

    order = [int(position) for position in raw_order]
    target_len = int(item.get("target_length", len(order)))
    target_len = max(0, target_len)
    if target_len == 0:
        return [], 0

    order = [position for position in order if 0 <= position < target_len]
    return order, target_len


def get_target_ids(item, target_len):
    raw_target_ids = item.get("target_ids")
    if raw_target_ids is None:
        return []
    return [int(token_id) for token_id in raw_target_ids[:target_len]]


def infer_special_tokens(item, target_ids, target_len):
    raw_special_tokens = item.get("special_tokens") or {}
    special_tokens = {
        name: int(token_id)
        for name, token_id in raw_special_tokens.items()
        if token_id is not None
    }
    if special_tokens:
        return special_tokens

    if target_ids:
        max_token_id = max(target_ids)
        base = 1
        while base * 4 <= max_token_id:
            base *= 4
        if base >= 16 and max_token_id >= base:
            names = ["<PAD>", "<BOS>", "<EOS>", "<CDS>", "<UTR5>", "<UTR3>", "<MASK>"]
            special_tokens.update(
                {name: base + offset for offset, name in enumerate(names)}
            )

    seq_len = int(item.get("sequence_length", 0))
    if target_ids and 0 < seq_len < target_len:
        special_tokens.setdefault("<EOS>", target_ids[seq_len - 1])
    return special_tokens


def special_name_by_id(item, target_ids, target_len):
    special_tokens = infer_special_tokens(item, target_ids, target_len)
    return {int(token_id): name for name, token_id in special_tokens.items()}


def get_eos_position(item, target_ids, target_len, id_to_special):
    eos_ids = [token_id for token_id, name in id_to_special.items() if name == "<EOS>"]
    if eos_ids and target_ids:
        eos_id = eos_ids[0]
        for position, token_id in enumerate(target_ids[:target_len]):
            if token_id == eos_id:
                return position

    seq_len = int(item.get("sequence_length", target_len))
    if 0 < seq_len < target_len:
        return seq_len - 1
    return None


def get_special_positions(item, target_ids, target_len):
    id_to_special = special_name_by_id(item, target_ids, target_len)
    positions = []
    for position, token_id in enumerate(target_ids[:target_len]):
        token_name = id_to_special.get(token_id)
        if token_name:
            positions.append((position, token_id, token_name))

    eos_position = get_eos_position(item, target_ids, target_len, id_to_special)
    if eos_position is not None and not any(
        position == eos_position and token_name == "<EOS>"
        for position, _, token_name in positions
    ):
        token_id = target_ids[eos_position] if eos_position < len(target_ids) else None
        positions.append((eos_position, token_id, "<EOS>"))
    return positions


def full_canvas_items(data):
    items = []
    for item in data:
        order, target_len = normalize_full_generation_order(item)
        target_ids = get_target_ids(item, target_len)
        if order and target_len > 0:
            items.append((item, order, target_len, target_ids))
    return items


def select_evenly_spaced_items(items, count=12):
    if len(items) <= count:
        return items
    step = len(items) / count
    return [items[int(idx * step)] for idx in range(count)]


def short_label(value, max_len=22):
    value = str(value)
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def canvas_category(position, eos_position, token_name):
    if token_name == "<EOS>":
        return 4
    if token_name:
        return 3
    if eos_position is not None and position > eos_position:
        return 2
    return 1


def write_post_eos_generation_csv(data, output_dir):
    rows = []
    for item, order, target_len, target_ids in full_canvas_items(data):
        id_to_special = special_name_by_id(item, target_ids, target_len)
        eos_position = get_eos_position(item, target_ids, target_len, id_to_special)
        if eos_position is None:
            continue

        eos_generation_step = order.index(eos_position) if eos_position in order else None
        for generation_step, position in enumerate(order):
            if position <= eos_position:
                continue

            token_id = target_ids[position] if position < len(target_ids) else None
            token_label = id_to_special.get(token_id, "")
            rows.append(
                {
                    "id": item.get("id", ""),
                    "eos_position": eos_position,
                    "eos_generation_step": eos_generation_step,
                    "position": position,
                    "position_after_eos": position - eos_position,
                    "generation_step": generation_step,
                    "generated_after_eos_token": (
                        generation_step > eos_generation_step
                        if eos_generation_step is not None
                        else None
                    ),
                    "token_id": token_id,
                    "token_label": token_label,
                    "is_special": bool(token_label),
                }
            )

    output_path = os.path.join(output_dir, "post_eos_generation_positions.csv")
    pd.DataFrame(rows, columns=POST_EOS_COLUMNS).to_csv(output_path, index=False)
    print(f"Wrote after-EOS generation positions to {output_path}")


def plot_full_canvas_generation_order(data, output_dir):
    items = full_canvas_items(data)
    if not items:
        print("No full-canvas LO-ARM generation orders found. Skipping full-canvas plot.")
        return

    items.sort(key=lambda item_tuple: item_tuple[2], reverse=True)
    selected_items = select_evenly_spaced_items(items, count=24)
    max_len = max(target_len for _, _, target_len, _ in selected_items)

    matrix = np.full((len(selected_items), max_len), np.nan)
    special_labels_present = set()
    after_eos_patch = False

    for row_idx, (item, order, target_len, target_ids) in enumerate(selected_items):
        for generation_step, position in enumerate(order):
            matrix[row_idx, position] = (generation_step + 1) / max(1, len(order))

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#edf0f4")

    fig_height = max(4, 0.34 * len(selected_items) + 2.5)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Relative generation step")

    for row_idx, (item, _, target_len, target_ids) in enumerate(selected_items):
        id_to_special = special_name_by_id(item, target_ids, target_len)
        eos_position = get_eos_position(item, target_ids, target_len, id_to_special)
        if eos_position is not None and eos_position + 1 < target_len:
            ax.add_patch(
                Rectangle(
                    (eos_position + 0.5, row_idx - 0.5),
                    target_len - eos_position - 0.5,
                    1.0,
                    facecolor="#f28e2b",
                    alpha=0.16,
                    edgecolor="none",
                )
            )
            after_eos_patch = True

        for position, _, token_name in get_special_positions(item, target_ids, target_len):
            marker, color = SPECIAL_MARKERS.get(token_name, ("o", "#7b61ff"))
            ax.scatter(
                position,
                row_idx,
                marker=marker,
                s=80 if token_name == "<EOS>" else 36,
                facecolor=color,
                edgecolor="black",
                linewidth=0.45,
                zorder=3,
            )
            special_labels_present.add(token_name)

    ax.set_yticks(np.arange(len(selected_items)))
    ax.set_yticklabels([short_label(item.get("id", "")) for item, _, _, _ in selected_items], fontsize=8)
    ax.set_xlabel("LO-ARM target canvas position")
    ax.set_ylabel("Sequence ID")
    ax.set_title("Full LO-ARM Canvas Generation Order")

    handles = []
    if after_eos_patch:
        handles.append(Patch(facecolor="#f28e2b", alpha=0.25, label="Positions after EOS"))
    for token_name in sorted(special_labels_present):
        marker, color = SPECIAL_MARKERS.get(token_name, ("o", "#7b61ff"))
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=color,
                markeredgecolor="black",
                label=token_name,
                markersize=8,
            )
        )
    if handles:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(5, len(handles)))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "full_canvas_generation_order.png"), dpi=300)
    plt.close()


def create_full_canvas_generation_animation(data, output_dir):
    items = full_canvas_items(data)
    if not items:
        return

    items.sort(key=lambda item_tuple: item_tuple[2], reverse=True)
    selected_items = select_evenly_spaced_items(items, count=12)
    max_len = max(target_len for _, _, target_len, _ in selected_items)
    max_steps = max(len(order) for _, order, _, _ in selected_items)

    n_items = len(selected_items)
    n_cols = 2 if n_items > 1 else 1
    n_rows = math.ceil(n_items / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, max(3, 1.5 * n_rows)), sharex=True)
    axes_flat = np.atleast_1d(axes).flatten()
    display_grids = []

    for idx, ax in enumerate(axes_flat):
        if idx >= n_items:
            ax.axis("off")
            continue

        item, _, target_len, target_ids = selected_items[idx]
        id_to_special = special_name_by_id(item, target_ids, target_len)
        eos_position = get_eos_position(item, target_ids, target_len, id_to_special)

        grid = np.zeros((1, max_len))
        grid[0, target_len:] = -1
        display_grids.append(grid)
        ax.imshow(grid, cmap=FULL_CANVAS_CMAP, norm=FULL_CANVAS_NORM, aspect="auto")

        if eos_position is not None:
            ax.axvline(x=eos_position + 0.5, color="#d62728", linestyle="--", linewidth=1.2)
        ax.set_yticks([])
        ax.set_ylabel(short_label(item.get("id", ""), max_len=18), rotation=0, labelpad=42, va="center", fontsize=8)

    axes_flat[-1].set_xlabel("LO-ARM target canvas position")
    if n_cols == 2 and n_items > 1:
        axes_flat[-2].set_xlabel("LO-ARM target canvas position")

    legend_handles = [
        Patch(facecolor="#4c78a8", label="Generated"),
        Patch(facecolor="#f28e2b", label="After EOS"),
        Patch(facecolor="#7b61ff", label="Special token"),
        Patch(facecolor="#d62728", label="EOS"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4)
    suptitle = fig.suptitle("Full LO-ARM Canvas Generation Progress (Step: 0)", fontsize=14, weight="bold")
    plt.tight_layout(rect=(0, 0.08, 1, 0.96))

    def update(frame):
        artists = []
        for idx, ax in enumerate(axes_flat[:n_items]):
            item, order, target_len, target_ids = selected_items[idx]
            id_to_special = special_name_by_id(item, target_ids, target_len)
            eos_position = get_eos_position(item, target_ids, target_len, id_to_special)
            grid = display_grids[idx]

            if frame > 0 and frame - 1 < len(order):
                position = order[frame - 1]
                token_id = target_ids[position] if position < len(target_ids) else None
                token_name = id_to_special.get(token_id, "")
                grid[0, position] = canvas_category(position, eos_position, token_name)

            ax.images[0].set_array(grid)
            artists.append(ax.images[0])

        suptitle.set_text(f"Full LO-ARM Canvas Generation Progress (Step: {frame} / {max_steps})")
        artists.append(suptitle)
        return artists

    ani = animation.FuncAnimation(fig, update, frames=max_steps + 1, blit=False)
    ani.save(os.path.join(output_dir, "full_canvas_generation.gif"), writer="pillow", fps=12)
    plt.close()


def plot_2d_evolution(data, output_dir, res):
    all_matrices = []
    for item in data:
        order, seq_len = normalize_generation_order(item)
        if seq_len < 10:
            continue
        state_matrix = np.zeros((seq_len, seq_len))
        for step in range(seq_len):
            if step > 0:
                state_matrix[step] = state_matrix[step-1]
            if step < len(order):
                pos_generated = order[step]
                state_matrix[step, pos_generated] = 1.0
        y_indices = np.linspace(0, seq_len - 1, res).astype(int)
        x_indices = np.linspace(0, seq_len - 1, res).astype(int)
        resampled_matrix = state_matrix[np.ix_(y_indices, x_indices)]
        all_matrices.append(resampled_matrix)
    if not all_matrices:
        return
    final_heatmap = np.mean(all_matrices, axis=0)
    plt.figure(figsize=(8, 7))
    sns.heatmap(final_heatmap, cmap="Blues", vmin=0.0, vmax=1.0, cbar_kws={'label': 'Probability Token is Present'}, xticklabels=False, yticklabels=False)
    plt.title(f"2D Generation Evolution Matrix (N={len(all_matrices)})\nNormalized Time vs. Position", fontsize=14)
    plt.ylabel("Relative Generation Step (Time \u2193)", fontsize=12)
    plt.xlabel("Relative Sequence Position (5' \u2192 3')", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2d_evolution_matrix.png"), dpi=300)
    plt.close()

def plot_regional_trajectory(data, utr5_lengths, output_dir, res):
    l5_curves, m5_curves, r5_curves = [], [], []
    l3_curves, m3_curves, r3_curves = [], [], []

    for item in data:
        order, seq_len = normalize_generation_order(item)
        u5_len = get_utr5_token_length(item, utr5_lengths)
        if seq_len < 10 or u5_len is None:
            continue

        u3_len = seq_len - u5_len

        if u5_len < 5 or u3_len < 5:
            continue

        l5_bnd, r5_bnd = int(0.2 * u5_len), int(0.8 * u5_len)
        l3_bnd, r3_bnd = u5_len + int(0.2 * u3_len), u5_len + int(0.8 * u3_len)

        tot_l5 = max(1, l5_bnd)
        tot_m5 = max(1, r5_bnd - l5_bnd)
        tot_r5 = max(1, u5_len - r5_bnd)

        tot_l3 = max(1, l3_bnd - u5_len)
        tot_m3 = max(1, r3_bnd - l3_bnd)
        tot_r3 = max(1, seq_len - r3_bnd)

        c_l5, c_m5, c_r5 = 0, 0, 0
        c_l3, c_m3, c_r3 = 0, 0, 0

        t_l5, t_m5, t_r5 = [], [], []
        t_l3, t_m3, t_r3 = [], [], []

        for step in range(seq_len):
            if step < len(order):
                pos = order[step]
                if pos < l5_bnd:
                    c_l5 += 1
                elif pos < r5_bnd:
                    c_m5 += 1
                elif pos < u5_len:
                    c_r5 += 1
                elif pos < l3_bnd:
                    c_l3 += 1
                elif pos < r3_bnd:
                    c_m3 += 1
                elif pos < seq_len:
                    c_r3 += 1

            t_l5.append(c_l5 / tot_l5)
            t_m5.append(c_m5 / tot_m5)
            t_r5.append(c_r5 / tot_r5)
            t_l3.append(c_l3 / tot_l3)
            t_m3.append(c_m3 / tot_m3)
            t_r3.append(c_r3 / tot_r3)

        steps_norm = np.linspace(0, seq_len - 1, res)
        base_steps = np.arange(seq_len)

        l5_curves.append(np.interp(steps_norm, base_steps, t_l5))
        m5_curves.append(np.interp(steps_norm, base_steps, t_m5))
        r5_curves.append(np.interp(steps_norm, base_steps, t_r5))
        l3_curves.append(np.interp(steps_norm, base_steps, t_l3))
        m3_curves.append(np.interp(steps_norm, base_steps, t_m3))
        r3_curves.append(np.interp(steps_norm, base_steps, t_r3))

    if not l5_curves:
        print("No valid sequences found for trajectories.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    x_axis = np.linspace(0, 100, res)

    ax1.plot(x_axis, np.mean(l5_curves, axis=0) * 100, label="Left Edge (First 20%)", color="crimson", lw=2)
    ax1.plot(x_axis, np.mean(m5_curves, axis=0) * 100, label="Middle (Middle 60%)", color="orange", lw=2)
    ax1.plot(x_axis, np.mean(r5_curves, axis=0) * 100, label="Right Edge (Last 20%)", color="royalblue", lw=2)
    ax1.set_xlabel("Overall Generation Progress (%)", fontsize=12)
    ax1.set_ylabel("Region Completion (%)", fontsize=12)
    ax1.set_title(f"5'UTR Strategy (N={len(l5_curves)})", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(x_axis, np.mean(l3_curves, axis=0) * 100, label="Left Edge (First 20%)", color="crimson", lw=2)
    ax2.plot(x_axis, np.mean(m3_curves, axis=0) * 100, label="Middle (Middle 60%)", color="orange", lw=2)
    ax2.plot(x_axis, np.mean(r3_curves, axis=0) * 100, label="Right Edge (Last 20%)", color="royalblue", lw=2)
    ax2.set_xlabel("Overall Generation Progress (%)", fontsize=12)
    ax2.set_ylabel("Region Completion (%)", fontsize=12)
    ax2.set_title(f"3'UTR Strategy (N={len(l3_curves)})", fontsize=14)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regional_trajectories.png"), dpi=300)
    plt.close()

def create_multi_generation_animation(data, utr5_lengths, output_dir):
    valid_items = []
    for item in data:
        order, seq_len = normalize_generation_order(item)
        if seq_len >= 10 and get_utr5_token_length(item, utr5_lengths) is not None:
            valid_items.append((item, order, seq_len))

    if len(valid_items) < 12:
        print(f"Not enough valid sequences for animation (found {len(valid_items)}, need at least 12). Skipping animation.")
        return
        
    # Sort sequences by descending length
    valid_items.sort(key=lambda x: x[2], reverse=True)
    
    # Grab 12 evenly spaced percentiles to show a perfect distribution of lengths
    idx_step = len(valid_items) / 12
    selected_items = [valid_items[int(i * idx_step)] for i in range(12)]
    
    max_arena_len = max(seq_len for _, _, seq_len in selected_items)
    
    fig, axes = plt.subplots(6, 2, figsize=(16, 11), sharex='col')
    axes_flat = axes.flatten()
    display_grids = [np.zeros((1, max_arena_len)) for _ in range(12)]
    ims = []
    max_steps = max(seq_len for _, _, seq_len in selected_items)
    
    for idx, ax in enumerate(axes_flat):
        item, _, seq_len = selected_items[idx]
        u5_len = get_utr5_token_length(item, utr5_lengths)
        
        display_grids[idx][0, seq_len:] = -0.3
        im = ax.imshow(display_grids[idx], cmap="Blues", vmin=-0.5, vmax=1.0, aspect="auto")
        ims.append(im)
        
        if u5_len < seq_len:
            ax.axvline(x=u5_len - 0.5, color="crimson", linestyle="--", linewidth=1.5)
            if idx in [0, 1]:
                ax.text(u5_len - 2, -0.7, "5'UTR", color="crimson", ha="right", va="center", fontsize=9, weight="bold")
                ax.text(u5_len + 2, -0.7, "3'UTR", color="crimson", ha="left", va="center", fontsize=9, weight="bold")
                
        ax.set_yticks([])
        ax.set_ylabel(f"ID: {item['id']}", rotation=0, labelpad=40, va="center", fontsize=8)
        
    axes[-1, 0].set_xlabel("Target Sequence Token Index Position")
    axes[-1, 1].set_xlabel("Target Sequence Token Index Position")
    suptitle = fig.suptitle("Parallel Generation Progress (Step: 0)", fontsize=14, weight="bold")
    plt.tight_layout()

    def update(frame):
        for idx in range(12):
            item, order, seq_len = selected_items[idx]
            if frame > 0 and (frame - 1) < seq_len:
                pos = order[frame - 1]
                if pos < seq_len and pos < max_arena_len:
                    display_grids[idx][0, pos] = 1.0
            ims[idx].set_array(display_grids[idx])
        suptitle.set_text(f"Parallel Generation Progress (Step: {frame} / {max_steps})")
        return ims + [suptitle]

    ani = animation.FuncAnimation(fig, update, frames=max_steps + 1, blit=False)
    ani.save(os.path.join(output_dir, "multi_sequence_generation.gif"), writer="pillow", fps=12)
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.input_json, "r") as f:
        data = json.load(f)
    df = pd.read_csv(args.input_csv)
    
    utr5_lengths = {}
    for _, row in df.iterrows():
        utr5_str = str(row.get("utr5", ""))
        if utr5_str == "nan":
            utr5_str = ""
        utr5_lengths[str(row["id"])] = math.ceil(len(utr5_str) / 3)
        
    plot_2d_evolution(data, args.output_dir, args.resolution)
    plot_regional_trajectory(data, utr5_lengths, args.output_dir, args.resolution)
    create_multi_generation_animation(data, utr5_lengths, args.output_dir)
    write_post_eos_generation_csv(data, args.output_dir)
    plot_full_canvas_generation_order(data, args.output_dir)
    create_full_canvas_generation_animation(data, args.output_dir)

if __name__ == "__main__":
    main()
