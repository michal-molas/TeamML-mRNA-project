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
import seaborn as sns
import pandas as pd

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

if __name__ == "__main__":
    main()
