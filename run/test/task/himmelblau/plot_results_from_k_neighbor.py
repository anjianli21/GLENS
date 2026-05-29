import os
import math
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CDF and compute statistics from pre-computed k-neighbor results")
    parser.add_argument("--file_paths", 
                        type=lambda x: [s.strip() for s in x.split(',')], 
                        required=True,
                        help="comma-separated list of k-neighbor pickle file paths")
    parser.add_argument("--legend_names", 
                        type=lambda x: [s.strip() for s in x.split(',')], 
                        required=True,
                        help="comma-separated list of legend names (must match number of file paths)")
    parser.add_argument("--plot_name", 
                        type=str, 
                        required=True,
                        help="base name for output plot file")
    parser.add_argument("--output_dir", 
                        type=str, 
                        help="output directory for plot files")
    # Flexible x/y axis for CDF plot (None = auto from data)
    parser.add_argument("--x_max", type=float, default=None, help="x-axis max for k-neighbor CDF (default: from data)")
    parser.add_argument("--x_min", type=float, default=0.0, help="x-axis min for k-neighbor CDF")
    parser.add_argument("--x_bin_size", type=float, default=None, help="x bin step for CDF (default: auto linspace)")
    parser.add_argument("--y_max", type=float, default=None, help="y-axis max for CDF, e.g. 1.0 or 0.7 (default: 1.0)")
    parser.add_argument("--y_min", type=float, default=0.0, help="y-axis min for CDF")
    parser.add_argument("--y_tick_step", type=float, default=None, help="y-axis tick step, e.g. 0.1 (default: 0.1)")

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = "outputs"

    # Validate that file_paths and legend_names have the same length
    if len(args.file_paths) != len(args.legend_names):
        raise ValueError(f"Number of file paths ({len(args.file_paths)}) must equal number of legend names ({len(args.legend_names)}).")

    # Prepare plot style
    plt.rcParams.update({
        "font.size": 22,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 22,
    })

    k_neighbors_data = []
    method_labels = []

    # Process each file path
    for idx, file_path in enumerate(args.file_paths):
        label = args.legend_names[idx]
        method_labels.append(label)

        if not os.path.exists(file_path):
            print(f"Warning: Data file not found: {file_path}")
            k_neighbors_data.append([])  # Empty list for missing files
            continue

        # Load k-neighbor results
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            k_neighbors = data["k_neighbors"]
            k_neighbors_data.append(k_neighbors)
            print(f"Loaded {len(k_neighbors)} samples for {label}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            k_neighbors_data.append([])  # Empty list for error cases
            continue

    # Fixed palette: same line colors and styles as robot plot script
    num_methods = 5
    colors = [
        "#0173B2",  # blue
        "#DE8F05",  # orange
        "#029E73",  # green
        "#6A51A3",  # violet
        "#E74C3C",  # red
    ]
    linestyles = [
        "-",               # solid
        (0, (8, 4)),       # long dash
        (0, (5, 2)),       # medium dash
        (0, (4, 2)),       # short-medium dash
        (0, (2, 1)),       # shortest dash
    ]

    # Flexible x-axis: use args or auto from data
    max_k_neighbor = 0
    for k_neighbors in k_neighbors_data:
        if len(k_neighbors) > 0:
            max_k_neighbor = max(max_k_neighbor, np.max(k_neighbors))
    x_min_plot = args.x_min
    x_max_plot = args.x_max if args.x_max is not None else max(10, int(math.ceil(max_k_neighbor)))
    if args.x_bin_size is not None and args.x_bin_size > 0:
        x_range = np.arange(x_min_plot, x_max_plot + 1e-9, args.x_bin_size)
        tick_step = max(1, int(round((x_max_plot - x_min_plot) / 12))) if (x_max_plot - x_min_plot) > 12 else 1
        tick_positions = list(range(int(x_min_plot), int(x_max_plot) + 1, tick_step))
    else:
        x_range = np.linspace(x_min_plot, x_max_plot, max(50, int(x_max_plot) * 10 + 1))
        tick_step = max(1, int(math.ceil((x_max_plot - x_min_plot) / 12)))
        tick_positions = list(range(int(x_min_plot), int(x_max_plot) + 1, tick_step))

    # Flexible y-axis
    y_min_plot = args.y_min
    y_max_plot = args.y_max if args.y_max is not None else 1.0
    y_tick_step = args.y_tick_step if args.y_tick_step is not None else 0.1
    y_ticks = np.arange(0, y_max_plot + 1e-9, y_tick_step)
    
    # === PLOT: Cumulative Distribution ===
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for i, (k_neighbors, label) in enumerate(zip(k_neighbors_data, method_labels)):
        if len(k_neighbors) == 0:
            print(f"Warning: No data available for {label}")
            continue
        k_neighbors = np.asarray(k_neighbors)
        # CDF at x_range points
        cdf_values = np.array([np.sum(k_neighbors <= x) / len(k_neighbors) for x in x_range])
        ax.plot(x_range, cdf_values, label=label, color=colors[i % num_methods], linestyle=linestyles[i % num_methods], linewidth=2.5, marker='o', markersize=4)
    
    ax.set_xlim(x_min_plot, x_max_plot)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions)
    ax.set_ylim(y_min_plot, y_max_plot * 1.05 if y_max_plot < 1.0 else 1.05)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{int(x*100)}%" for x in y_ticks])
    if y_max_plot >= 1.0:
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel("$k$-neighborhood", fontsize=24)
    ax.set_ylabel("Cumulative Fraction", fontsize=24)
    ax.legend(fontsize=22, loc='best')
    ax.grid(True, alpha=0.3)
    plt.subplots_adjust(left=0.07, right=0.94, top=0.94, bottom=0.12)

    # Save figure
    os.makedirs(args.output_dir, exist_ok=True)
    
    plot_filename = f"{args.plot_name}_k_neighbor_cdf.png"
    plot_path = os.path.join(args.output_dir, plot_filename)
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved CDF plot to {plot_path}")
    
    # Close figure to free memory
    plt.close(fig)

    # === COMPUTE STATISTICS ===
    stats_lines = []
    stats_lines.append("=" * 80)
    stats_lines.append("K-NEIGHBOR STATISTICS")
    stats_lines.append("=" * 80)
    stats_lines.append("")
    
    # Compute statistics for each method
    for i, (k_neighbors, label) in enumerate(zip(k_neighbors_data, method_labels)):
        if len(k_neighbors) == 0:
            stats_lines.append(f"Method: {label}")
            stats_lines.append("  No data available")
            stats_lines.append("")
            continue
        
        k_neighbors_array = np.array(k_neighbors)
        
        stats_lines.append(f"Method: {label}")
        stats_lines.append(f"  Total samples: {len(k_neighbors)}")
        stats_lines.append("")
        
        # Cumulative coverage ratios for k-neighbors 1-10
        stats_lines.append("  Cumulative Coverage Ratios:")
        for k in range(1, 16):
            coverage = np.sum(k_neighbors_array <= k) / len(k_neighbors)
            stats_lines.append(f"    {k}-neighbor: {coverage:.4f} ({coverage*100:.2f}%)")
        stats_lines.append("")
        
        # Statistical measures
        mean_val = np.mean(k_neighbors_array)
        std_val = np.std(k_neighbors_array)
        q25 = np.percentile(k_neighbors_array, 25)
        q50 = np.percentile(k_neighbors_array, 50)  # median
        q75 = np.percentile(k_neighbors_array, 75)
        min_val = np.min(k_neighbors_array)
        max_val = np.max(k_neighbors_array)
        
        stats_lines.append("  Statistical Measures:")
        stats_lines.append(f"    Mean:        {mean_val:.4f}")
        stats_lines.append(f"    Std:         {std_val:.4f}")
        stats_lines.append(f"    25% Quantile: {q25:.4f}")
        stats_lines.append(f"    50% Quantile: {q50:.4f} (Median)")
        stats_lines.append(f"    75% Quantile: {q75:.4f}")
        stats_lines.append(f"    Min:         {min_val:.0f}")
        stats_lines.append(f"    Max:         {max_val:.0f}")
        stats_lines.append("")
        stats_lines.append("-" * 80)
        stats_lines.append("")
    
    # Save statistics to file
    stats_filename = f"{args.plot_name}_k_neighbor_stats.txt"
    stats_path = os.path.join(args.output_dir, stats_filename)
    with open(stats_path, 'w') as f:
        f.write('\n'.join(stats_lines))
    print(f"Saved statistics to {stats_path}")
    
    # Also print to console
    print("\n" + "\n".join(stats_lines))

