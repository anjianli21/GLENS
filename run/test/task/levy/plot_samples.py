"""
Scatter-grid plots: rows = conditions (condition seeds), columns = test methods.
Ground-truth samples (circles) overlaid with matched test samples (crosses) in 2D slices of x.

Expects ``--gt_data_root`` to contain ``condition_seed_<id>/`` folders with GT pickles, and
test pickles whose layout matches ``load_test_samples`` (per-lambda x samples).
Problem-agnostic: use ``--x_label`` / ``--y_label`` for axis names when coordinates are not x1/x2.
"""
import argparse
import glob
import os
import pickle
import shutil
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

# Raster export resolution: matplotlib ``savefig(..., dpi=...)`` is PPI for PNG (and rasterized PDF layers).
SAVEFIG_DPI = 300


def _load_pkl_cpu(path: str):
    """Load pickle file, mapping CUDA tensors to CPU."""
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except:
        original_restore = torch.serialization.default_restore_location
        torch.serialization.default_restore_location = lambda s, loc: original_restore(s, 'cpu' if isinstance(loc, str) and loc.startswith('cuda') else loc)
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        finally:
            torch.serialization.default_restore_location = original_restore


def find_condition_seeds(gt_root: str) -> List[int]:
    seeds = []
    for name in os.listdir(gt_root):
        if name.startswith("condition_seed_"):
            try:
                seeds.append(int(name.split("_")[-1]))
            except ValueError:
                continue
    if not seeds:
        raise ValueError(f"No condition_seed_* directories found under {gt_root}")
    return sorted(seeds)


def load_gt_for_condition(gt_root: str, condition_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    cond_dir = os.path.join(gt_root, f"condition_seed_{condition_seed}")
    pkl_files = sorted(glob.glob(os.path.join(cond_dir, "*.pkl")))
    if not pkl_files:
        raise ValueError(f"No .pkl files found in {cond_dir}")

    lambda_values, xs_list = [], []
    for pkl_path in pkl_files:
        data = _load_pkl_cpu(pkl_path)
        if not isinstance(data, dict) or "condition_lambda_value" not in data:
            raise KeyError(f"Invalid GT pkl: {pkl_path}")

        lambda_values.append(np.asarray(data["condition_lambda_value"], dtype=float).reshape(-1))
        
        if "solver_info" in data:
            x_final = data["solver_info"][sorted(data["solver_info"].keys())[-1]]["x"]
        elif "x" in data:
            x_final = data["x"]
        else:
            raise KeyError(f"Neither 'solver_info' nor 'x' found in {pkl_path}")
        xs_list.append(np.asarray(x_final).reshape(-1))

    lambda_arr = np.stack(lambda_values, axis=0)
    if not np.allclose(lambda_arr, lambda_arr[0]):
        raise ValueError(f"Inconsistent lambda values in condition_seed_{condition_seed}")
    
    return lambda_arr[0], np.stack(xs_list, axis=0)


def load_all_gt(gt_root: str, condition_seeds: List[int]) -> Dict[int, Dict[str, np.ndarray]]:
    gt_data = {}
    for seed in condition_seeds:
        lam, xs = load_gt_for_condition(gt_root, seed)
        gt_data[seed] = {"lambda": lam, "xs": xs}
    return gt_data


def _to_numpy_2d(obj) -> np.ndarray:
    if torch.is_tensor(obj):
        obj = obj.detach().cpu().numpy()
    arr = np.asarray(obj)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    elif arr.ndim == 1:
        return arr.reshape(-1, 1)
    elif arr.ndim >= 3:
        return arr.reshape(-1, arr.shape[-1])
    return arr


def _concat_lambda_array(raw_lambda) -> np.ndarray:
    items = raw_lambda if isinstance(raw_lambda, list) else [raw_lambda]
    parts = []
    for item in items:
        arr = item.detach().cpu().numpy().astype(float) if torch.is_tensor(item) else np.asarray(item, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim >= 3:
            arr = arr.reshape(-1, arr.shape[-1])
        parts.append(arr)
    return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


def _concat_x_array(raw_x) -> np.ndarray:
    items = raw_x if isinstance(raw_x, list) else [raw_x]
    return np.concatenate([_to_numpy_2d(item) for item in items], axis=0) if len(items) > 1 else _to_numpy_2d(items[0])


def _expand_lambda(lam: np.ndarray, target_size: int) -> np.ndarray:
    if lam.shape[0] < target_size:
        return np.repeat(lam, target_size // lam.shape[0], axis=0)
    return lam


def load_test_samples(test_pkl_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = _load_pkl_cpu(test_pkl_path)
    
    if isinstance(data, dict):
        lambda_key = next((k for k in data.keys() if "lambda" in k.lower()), None)
        x_key = next((k for k in data.keys() if k.lower().startswith("x") or "sample" in k.lower()), None)
        if not lambda_key or not x_key:
            raise KeyError(f"Missing keys in test pkl: {list(data.keys())}")
        
        lambda_values = _expand_lambda(_concat_lambda_array(data[lambda_key]), _concat_x_array(data[x_key]).shape[0])
        xs = _concat_x_array(data[x_key])
    elif isinstance(data, list):
        lambda_list, xs_list = [], []
        for item in data:
            if not isinstance(item, dict):
                raise TypeError(f"Expected dict in test data list, got {type(item)}")
            lambda_key = next((k for k in item.keys() if "lambda" in k.lower()), None)
            x_key = next((k for k in item.keys() if k.lower().startswith("x") or "sample" in k.lower()), None)
            if not lambda_key or not x_key:
                raise KeyError(f"Missing keys in test item: {list(item.keys())}")
            
            x_arr = _concat_x_array(item[x_key])
            lam = _expand_lambda(_concat_lambda_array(item[lambda_key]), x_arr.shape[0])
            lambda_list.append(lam)
            xs_list.append(x_arr)
        lambda_values = np.concatenate(lambda_list, axis=0)
        xs = np.concatenate(xs_list, axis=0)
    else:
        raise TypeError(f"Unsupported test pkl type: {type(data)}")

    if lambda_values.shape[0] != xs.shape[0]:
        raise ValueError(f"Length mismatch: lambdas={lambda_values.shape[0]}, xs={xs.shape[0]}")

    lambda_values = lambda_values.reshape(lambda_values.shape[0], -1)
    order = np.lexsort(lambda_values.T[::-1])
    lambda_values, xs = lambda_values[order], xs[order]
    
    unique_lambdas, counts = np.unique(lambda_values, axis=0, return_counts=True)
    print(f"Loaded {xs.shape[0]} test samples, {len(unique_lambdas)} unique lambdas")
    print("Per-lambda counts:", dict(zip([tuple(float(v) for v in lam) for lam in unique_lambdas], counts)))
    
    return lambda_values, xs


def build_test_groups(lambda_values: np.ndarray, xs: np.ndarray) -> Dict[Tuple[float, ...], np.ndarray]:
    lambda_values = lambda_values.reshape(lambda_values.shape[0], -1)
    unique_lambdas, inverse = np.unique(lambda_values, axis=0, return_inverse=True)
    return {tuple(float(v) for v in lam): xs[inverse == idx] for idx, lam in enumerate(unique_lambdas)}


def _lambda_to_str(lam) -> str:
    arr = np.atleast_1d(np.asarray(lam, dtype=float))
    if arr.size == 1:
        return f"{arr.item():.2f}"
    parts = ", ".join(f"{v:.2f}" for v in arr[:4])
    return f"[{parts}{', ...' if arr.size > 4 else ''}]"


def find_matching_lambda(lam_gt: np.ndarray, lambda_candidates: np.ndarray, tol: float = 1e-5) -> np.ndarray:
    lam_gt_arr = np.asarray(lam_gt, dtype=float).reshape(-1)
    cand = np.asarray(lambda_candidates, dtype=float).reshape(-1, lam_gt_arr.shape[0])
    diffs = np.linalg.norm(cand - lam_gt_arr.reshape(1, -1), axis=1)
    best_idx = int(np.argmin(diffs))
    if diffs[best_idx] > tol:
        raise ValueError(f"No matching lambda for {_lambda_to_str(lam_gt_arr)}; closest is {_lambda_to_str(cand[best_idx])} (diff={diffs[best_idx]:.2e})")
    return cand[best_idx]


def _configure_publication_fonts(use_usetex: bool = True) -> None:
    """
    Match Overleaf/LaTeX text as closely as possible: use real LaTeX (usetex)
    when `latex` is on PATH; otherwise serif + Computer Modern mathtext.
    """
    base = {
        "axes.unicode_minus": False,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 18,
        "font.size": 16,
    }
    if use_usetex and shutil.which("latex"):
        mpl.rcParams.update(
            {
                **base,
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
                "text.latex.preamble": (
                    r"\usepackage{amsmath}"
                    r"\usepackage{amssymb}"
                ),
            }
        )
    else:
        if use_usetex:
            print(
                "Warning: LaTeX not found on PATH; using mathtext (CM-style math). "
                "Install TeX or run with --no_usetex to silence this."
            )
        mpl.rcParams.update(
            {
                **base,
                "text.usetex": False,
                "font.family": "serif",
                "mathtext.fontset": "cm",
            }
        )


def split_long_title(title: str, max_length: int = 40) -> str:
    """Split a long title into two lines if it exceeds max_length."""
    if len(title) <= max_length:
        return title
    # Try to split at a space near the middle
    mid = len(title) // 2
    # Look for a space before the middle
    split_idx = title.rfind(' ', 0, mid)
    if split_idx == -1:
        # If no space before middle, look after middle
        split_idx = title.find(' ', mid)
    if split_idx == -1:
        # If no space found, just split at middle
        split_idx = mid
    return title[:split_idx] + '\n' + title[split_idx + 1:]


def compute_gt_axis_limits(gt_data: Dict[int, Dict[str, np.ndarray]], x_dim1: int, x_dim2: int) -> Tuple[float, float, float, float]:
    """Compute axis limits from GT data only."""
    all_pts_x, all_pts_y = [], []
    for seed in gt_data:
        gt_xs = gt_data[seed]["xs"]
        if gt_xs.ndim != 2 or gt_xs.shape[1] <= max(x_dim1, x_dim2):
            continue
        gt_pts = gt_xs[:, [x_dim1, x_dim2]]
        all_pts_x.extend(gt_pts[:, 0].tolist())
        all_pts_y.extend(gt_pts[:, 1].tolist())
    
    if not all_pts_x or not all_pts_y:
        return 0.0, 1.0, 0.0, 1.0
    
    x_min, x_max = min(all_pts_x), max(all_pts_x)
    y_min, y_max = min(all_pts_y), max(all_pts_y)
    margin = 0.05 * max(x_max - x_min, y_max - y_min, 1e-8)
    return x_min - margin, x_max + margin, y_min - margin, y_max + margin


def _square_axis_limits(
    x_min: float, x_max: float, y_min: float, y_max: float
) -> Tuple[float, float, float, float]:
    """
    Match x and y axis spans (centered on each midpoint) so ``set_aspect('equal')``
    yields square axes boxes for the 2D slice.
    """
    xspan = x_max - x_min
    yspan = y_max - y_min
    span = max(xspan, yspan, 1e-12)
    xmid = 0.5 * (x_min + x_max)
    ymid = 0.5 * (y_min + y_max)
    half = 0.5 * span
    return xmid - half, xmid + half, ymid - half, ymid + half


# Inches per subplot row/column (same value => square grid cells in figure layout).
SUBPLOT_CELL_INCHES = 3.42

# GT circles: visible under crosses but still slightly softer than test markers.
_GT_CIRCLE_FACE = (0.58, 0.62, 0.70)
_GT_CIRCLE_EDGE = (0.30, 0.34, 0.40)
_GT_CIRCLE_ALPHA = 0.78
_GT_CIRCLE_SIZE = 105

# Default test-method colors (first four columns); cycles if there are more methods.
_METHOD_COLORS_HEX = ("#DE8F05", "#029E73", "#6A51A3", "#E74C3C")


def plot_overlay_cell(
    ax,
    seed: int,
    gt_data: Dict[int, Dict[str, np.ndarray]],
    test_groups: Dict[Tuple[float, ...], np.ndarray],
    method_name: str,
    method_color,
    x_dim1: int,
    x_dim2: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    lambda_match_tol: float,
    show_column_title: bool,
):
    """
    One condition (row) × one test method (column): GT as circles behind,
    matched test samples as crosses on top (method_color).
    """
    gt_xs = gt_data[seed]["xs"]
    if gt_xs.ndim == 2 and gt_xs.shape[1] > max(x_dim1, x_dim2):
        gt_pts = gt_xs[:, [x_dim1, x_dim2]]
        ax.scatter(
            gt_pts[:, 0],
            gt_pts[:, 1],
            facecolors=_GT_CIRCLE_FACE,
            edgecolors=_GT_CIRCLE_EDGE,
            linewidths=1.0,
            marker="o",
            s=_GT_CIRCLE_SIZE,
            alpha=_GT_CIRCLE_ALPHA,
            zorder=1,
            label="GT",
        )

    lam_gt_vec = np.asarray(gt_data[seed]["lambda"], dtype=float).reshape(-1)
    if test_groups:
        test_lambda_keys = list(test_groups.keys())
        lambda_candidates = np.array(test_lambda_keys, dtype=float).reshape(len(test_lambda_keys), -1)
        try:
            lam_test_vec = find_matching_lambda(lam_gt_vec, lambda_candidates, tol=lambda_match_tol)
            test_xs = test_groups[tuple(float(v) for v in lam_test_vec)]
            test_pts = test_xs[:, [x_dim1, x_dim2]]
            ax.scatter(
                test_pts[:, 0],
                test_pts[:, 1],
                color=method_color,
                marker="x",
                s=130,
                linewidths=2.0,
                alpha=0.92,
                zorder=3,
                label=method_name,
            )
        except ValueError as e:
            print(f"Warning: Could not match lambda for seed {seed} in method {method_name}: {e}")

    if show_column_title:
        title_text = split_long_title(method_name, max_length=40)
        ax.set_title(title_text)

    ax.tick_params(labelsize=14)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)


def _apply_grid_xy_labels(
    axes,
    n_rows: int,
    n_cols: int,
    xlabel: str = r"$x_1$",
    ylabel: str = r"$x_2$",
):
    """$x_1$ on the bottom row; $x_2$ on each left-column panel (horizontal, same size as $x_1$)."""
    axes_label_fs = mpl.rcParams["axes.labelsize"]
    for r in range(n_rows):
        for c in range(n_cols):
            if r == n_rows - 1:
                axes[r, c].set_xlabel(xlabel, fontsize=axes_label_fs)
    for r in range(n_rows):
        ax_l = axes[r, 0]
        ax_l.set_ylabel(ylabel, rotation=0, fontsize=axes_label_fs)
        # Force vertical-centering independent of font metrics/labelpad.
        # Make room between label and the leftmost subplot.
        ax_l.yaxis.set_label_coords(-0.24, 0.5)
        ax_l.yaxis.label.set_verticalalignment("center")
        ax_l.yaxis.label.set_horizontalalignment("center")


def plot_gt_and_test_methods(
    condition_seeds: List[int],
    gt_data: Dict[int, Dict[str, np.ndarray]],
    test_methods_data: List[Dict[Tuple[float, ...], np.ndarray]],
    test_method_names: List[str],
    x_dim1: int,
    x_dim2: int,
    output_path: str,
    lambda_match_tol: float,
    figure_title: Optional[str] = None,
    use_usetex: bool = True,
    xlabel: str = r"$x_1$",
    ylabel: str = r"$x_2$",
):
    """
    Rows = condition seeds, columns = test methods. Each panel overlays GT (circles)
    and that method's test samples (crosses, one color per column).

    Figure uses square grid cells; axis limits are squared so each panel is a square
    at equal data aspect (``set_aspect('equal', adjustable='box')``).
    """
    _configure_publication_fonts(use_usetex=use_usetex)

    # IMPORTANT: preserve the user-specified order (derived from --condition_seed_indices).
    # Do NOT sort, otherwise row order won't match the indices order the user chose.
    seeds_in_order = list(condition_seeds)
    n_rows = len(seeds_in_order)
    n_methods = len(test_method_names)
    if n_methods == 0:
        raise ValueError("At least one test method is required.")
    if n_rows == 0:
        raise ValueError("At least one condition seed is required.")

    method_colors = [_METHOD_COLORS_HEX[i % len(_METHOD_COLORS_HEX)] for i in range(n_methods)]

    # Square grid cells; tight GridSpec gaps keep subplots close.
    fig_w = SUBPLOT_CELL_INCHES * n_methods
    fig_h = SUBPLOT_CELL_INCHES * n_rows

    x_min, x_max, y_min, y_max = compute_gt_axis_limits(gt_data, x_dim1, x_dim2)
    x_min, x_max, y_min, y_max = _square_axis_limits(x_min, x_max, y_min, y_max)

    fig, axes = plt.subplots(
        n_rows,
        n_methods,
        figsize=(fig_w, fig_h),
        sharex="col",
        sharey="row",
        squeeze=False,
        gridspec_kw={"wspace": 0.03, "hspace": 0.072},
    )
    for r, seed in enumerate(seeds_in_order):
        for c in range(n_methods):
            ax = axes[r, c]
            plot_overlay_cell(
                ax,
                seed,
                gt_data,
                test_methods_data[c],
                test_method_names[c],
                method_colors[c],
                x_dim1,
                x_dim2,
                x_min,
                x_max,
                y_min,
                y_max,
                lambda_match_tol,
                show_column_title=(r == 0),
            )

    _apply_grid_xy_labels(axes, n_rows, n_methods, xlabel=xlabel, ylabel=ylabel)

    if figure_title:
        fig.suptitle(figure_title, fontsize=20, y=1.02)

    legend_handles = [
        Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markerfacecolor=_GT_CIRCLE_FACE,
            markeredgecolor=_GT_CIRCLE_EDGE,
            markersize=10,
            alpha=_GT_CIRCLE_ALPHA,
            markeredgewidth=1.0,
            label="Ground truth",
        ),
    ]
    for c, name in enumerate(test_method_names):
        clr = method_colors[c]
        legend_handles.append(
            Line2D(
                [],
                [],
                linestyle="None",
                marker="x",
                color=clr,
                markersize=11,
                markeredgewidth=2.0,
                label=name,
            )
        )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    # Fit axes up to ~rect top; place legend after tight_layout with its bottom just above that edge.
    fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=0.17, h_pad=0.22, pad=0.65)

    # Add α^(i) labels on the right side of the rightmost panel in each row (horizontal, no tilt).
    axes_label_fs = mpl.rcParams["axes.labelsize"]
    for r in range(n_rows):
        ax_r = axes[r, n_methods - 1]
        ax_r.yaxis.set_label_position("right")
        ax_r.set_ylabel(rf"$\alpha^{{({r + 1})}}$", rotation=0, fontsize=axes_label_fs)
        # Match $x_2$ centering: anchor label to the axes mid-height.
        ax_r.yaxis.set_label_coords(1.18, 0.5)
        ax_r.yaxis.label.set_verticalalignment("center")
        ax_r.yaxis.label.set_horizontalalignment("center")
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=min(1 + n_methods, 6),
        frameon=True,
        fontsize=14,
        borderaxespad=0.0,
    )
    fig.savefig(output_path, dpi=SAVEFIG_DPI, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        prog="plot_samples.py",
        description="Plot GT vs test samples: one row per condition, one column per method; GT circles + test crosses.",
    )
    parser.add_argument(
        "--gt_data_root",
        type=str,
        required=True,
        help="Root with condition_seed_<int>/ subdirs, each holding GT .pkl files (see load_gt_for_condition).",
    )
    parser.add_argument("--test_sample_pkls", 
                        type=lambda x: [s.strip() for s in x.split(',')], 
                        required=True,
                        help="comma-separated list of test sample pkl paths")
    parser.add_argument("--test_method_names", 
                        type=lambda x: [s.strip() for s in x.split(',')], 
                        required=True,
                        help="comma-separated list of test method names (must match number of pkl paths)")
    parser.add_argument("--condition_seed_indices", 
                        type=lambda x: [int(s.strip()) for s in x.split(',')], 
                        default=None,
                        help="comma-separated list of condition seed indices (0-based) to visualize, e.g., '0,1,3' for first, second, and fourth seeds. If not provided, uses first 3 seeds.")
    parser.add_argument("--x_dim1", type=int, default=0, help="First coordinate index into x for the horizontal axis")
    parser.add_argument("--x_dim2", type=int, default=1, help="Second coordinate index into x for the vertical axis")
    parser.add_argument(
        "--x_label",
        type=str,
        default=r"$x_1$",
        help=r"Horizontal axis label (default: $x_1$). Use mathtext/LaTeX as needed for your problem.",
    )
    parser.add_argument(
        "--y_label",
        type=str,
        default=r"$x_2$",
        help=r"Vertical axis label (default: $x_2$).",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--figure_name", type=str, required=True, help="Figure file name")
    parser.add_argument("--lambda_match_tol", type=float, default=1e-5, help="Lambda matching tolerance")
    parser.add_argument(
        "--no_usetex",
        action="store_true",
        help="Disable LaTeX text rendering; use matplotlib mathtext (CM-style) only",
    )
    args = parser.parse_args()

    # Validate that test_sample_pkls and test_method_names have the same length
    if len(args.test_sample_pkls) != len(args.test_method_names):
        raise ValueError(f"Number of test sample pkls ({len(args.test_sample_pkls)}) must equal number of test method names ({len(args.test_method_names)}).")

    output_path = os.path.join(args.output_dir, args.figure_name + ("" if args.figure_name.endswith(".png") else ".png"))
    os.makedirs(args.output_dir, exist_ok=True)

    all_seeds = find_condition_seeds(args.gt_data_root)
    
    # Select seeds based on indices if provided, otherwise use first 3
    if args.condition_seed_indices is not None:
        selected_seeds = []
        for idx in args.condition_seed_indices:
            if idx < 0 or idx >= len(all_seeds):
                raise ValueError(f"Condition seed index {idx} is out of range. Available indices: 0-{len(all_seeds)-1}")
            selected_seeds.append(all_seeds[idx])
    else:
        # Default: use first 3 seeds
        selected_seeds = all_seeds[:3] if len(all_seeds) >= 3 else all_seeds
    
    print(
        "Using GT condition_seeds (row order preserved): "
        f"{selected_seeds} (indices: {[all_seeds.index(s) for s in selected_seeds]})"
    )

    gt_data = load_all_gt(args.gt_data_root, selected_seeds)
    
    # Load all test methods
    test_methods_data = []
    for pkl_path in args.test_sample_pkls:
        if not os.path.exists(pkl_path):
            print(f"Warning: Test pkl not found: {pkl_path}")
            test_methods_data.append({})
            continue
        
        lambda_values, xs = load_test_samples(pkl_path)
        test_groups = build_test_groups(lambda_values, xs)
        test_methods_data.append(test_groups)

    print("\n=== Lambda Verification ===")
    for method_idx, (test_groups, method_name) in enumerate(zip(test_methods_data, args.test_method_names)):
        if not test_groups:
            print(f"⚠ Method {method_name}: No data loaded")
            continue
        
        test_lambda_keys = list(test_groups.keys())
        lambda_candidates = np.array(test_lambda_keys, dtype=float).reshape(len(test_lambda_keys), -1)
        
        print(f"\nMethod: {method_name}")
        # Preserve row order here too (same as --condition_seed_indices selection order).
        for seed in selected_seeds:
            lam_gt = np.asarray(gt_data[seed]["lambda"], dtype=float).reshape(-1)
            try:
                lam_test = find_matching_lambda(lam_gt, lambda_candidates, tol=args.lambda_match_tol)
                diff = np.linalg.norm(lam_gt - lam_test)
                print(f"  ✓ Seed {seed}: GT λ={_lambda_to_str(lam_gt)} = Test λ={_lambda_to_str(lam_test)} (diff={diff:.2e})")
            except ValueError as e:
                print(f"  ✗ Seed {seed}: {e}")
    print("\n=== Lambda verification complete ===\n")

    plot_gt_and_test_methods(
        selected_seeds,
        gt_data,
        test_methods_data,
        args.test_method_names,
        args.x_dim1,
        args.x_dim2,
        output_path,
        args.lambda_match_tol,
        None,
        use_usetex=not args.no_usetex,
        xlabel=args.x_label,
        ylabel=args.y_label,
    )


if __name__ == "__main__":
    main()

