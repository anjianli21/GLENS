"""
Paper figure: two side-by-side all-trajectories panels (different condition seeds),
one shared legend on the right. Same data rules as plot_test_warmstart_data.py
(optimality==1, first N trajectories by initial_guess_seed).
"""
import argparse
import os
import pickle
import shutil
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from matplotlib.ticker import ScalarFormatter
import numpy as np

from dataset.robot.legacy_car_keys import get_solver_x_value


def _configure_publication_fonts(use_usetex: bool = True) -> None:
    base = {
        "axes.unicode_minus": False,
        "axes.labelsize": 30,
        "axes.titlesize": 30,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 26,
        "font.size": 26,
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
                "axes.formatter.use_mathtext": True,
            }
        )


def setup_parameters(args):
    parameters = {}
    parameters["t_final_bound"] = [0.0, 20.0]
    parameters["timestep"] = args.timestep
    parameters["robot_num"] = args.robot_num
    parameters["u_num_per_robot"] = 2
    parameters["robot_u_bound"] = {"u0": [-1.0, 1.0], "u1": [-1.0, 1.0]}
    parameters["robot_v_bound"] = [-2.0, 2.0]
    parameters["robot_radius"] = 0.2
    parameters["robot_goal_radius"] = parameters["robot_radius"]
    parameters["obs_num"] = 2
    parameters["robot_start_pos"] = np.array([[0.0, 10.0], [10.0, 10.0], [5.0, 0.0]])[: args.robot_num]
    parameters["robot_start_v"] = [0.0] * parameters["robot_num"]
    parameters["robot_start_theta"] = [0.0, 0.0][: args.robot_num]
    parameters["robot_goal_pos"] = np.array([[10.0, 0.0], [0.0, 0.0], [5.0, 10.0]])[: args.robot_num]
    return parameters


def parameters_from_pickle(data, timestep, robot_num=2):
    args = SimpleNamespace(timestep=timestep, robot_num=robot_num, to_print_plot=False)
    parameters = setup_parameters(args)
    if "condition_data" in data:
        cd = data["condition_data"]
        parameters["obs_pos"] = np.asarray(cd["obs_pos"])
        parameters["obs_radius"] = np.asarray(cd["obs_radius"])
    return parameters


def integrate_dynamics(x_sol, parameters):
    robot_num = parameters["robot_num"]
    u_num_per_robot = parameters["u_num_per_robot"]
    robot_start_pos = parameters["robot_start_pos"]
    robot_start_v = parameters["robot_start_v"]
    robot_start_theta = parameters["robot_start_theta"]
    timestep = parameters["timestep"]

    t_final = x_sol["t_final"]
    if np.ndim(t_final) > 0:
        t_final = float(np.asarray(t_final).flatten()[0])
    else:
        t_final = float(t_final)

    robot_control = np.zeros((robot_num, timestep, 2))
    for i in range(robot_num):
        for t in range(timestep):
            for k in range(u_num_per_robot):
                key = f"robot_{i}_u{k}"
                robot_control[i, t, k] = x_sol[key][t]

    dt = t_final / timestep

    state_x = np.zeros((robot_num, timestep + 1))
    state_y = np.zeros((robot_num, timestep + 1))
    state_v = np.zeros((robot_num, timestep + 1))
    state_theta = np.zeros((robot_num, timestep + 1))

    for i in range(robot_num):
        state_x[i, 0] = robot_start_pos[i][0]
        state_y[i, 0] = robot_start_pos[i][1]
        state_v[i, 0] = robot_start_v[i]
        state_theta[i, 0] = robot_start_theta[i]

    dx = lambda v, theta: v * np.cos(theta)
    dy = lambda v, theta: v * np.sin(theta)
    dv = lambda a: a
    dtheta = lambda omega: omega

    for t in range(timestep):
        a = robot_control[:, t, 0]
        omega = robot_control[:, t, 1]

        k1_x = dx(state_v[:, t], state_theta[:, t])
        k1_y = dy(state_v[:, t], state_theta[:, t])
        k1_v = dv(a)
        k1_theta = dtheta(omega)

        k2_x = dx(state_v[:, t] + k1_v * dt / 2, state_theta[:, t] + k1_theta * dt / 2)
        k2_y = dy(state_v[:, t] + k1_v * dt / 2, state_theta[:, t] + k1_theta * dt / 2)
        k2_v = dv(a + k1_v * dt / 2)
        k2_theta = dtheta(omega + k1_theta * dt / 2)

        k3_x = dx(state_v[:, t] + k2_v * dt / 2, state_theta[:, t] + k2_theta * dt / 2)
        k3_y = dy(state_v[:, t] + k2_v * dt / 2, state_theta[:, t] + k2_theta * dt / 2)
        k3_v = dv(a + k2_v * dt / 2)
        k3_theta = dtheta(omega + k2_theta * dt / 2)

        k4_x = dx(state_v[:, t] + k3_v * dt, state_theta[:, t] + k3_theta * dt)
        k4_y = dy(state_v[:, t] + k3_v * dt, state_theta[:, t] + k3_theta * dt)
        k4_v = dv(a + k3_v * dt)
        k4_theta = dtheta(omega + k3_theta * dt)

        state_x[:, t + 1] = state_x[:, t] + (dt / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        state_y[:, t + 1] = state_y[:, t] + (dt / 6) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        state_v[:, t + 1] = state_v[:, t] + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        state_theta[:, t + 1] = state_theta[:, t] + (dt / 6) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)

    return state_x, state_y, state_v, state_theta


def pickle_to_x_sol(data):
    max_major_iter = data["max_major_iter"]
    x = data["solver_info"][max_major_iter]["x"]
    return {
        "t_final": np.asarray(get_solver_x_value(x, "t_final")).flatten(),
        "robot_0_u0": np.asarray(get_solver_x_value(x, "robot_0_u0")).flatten(),
        "robot_0_u1": np.asarray(get_solver_x_value(x, "robot_0_u1")).flatten(),
        "robot_1_u0": np.asarray(get_solver_x_value(x, "robot_1_u0")).flatten(),
        "robot_1_u1": np.asarray(get_solver_x_value(x, "robot_1_u1")).flatten(),
    }


def initial_guess_seed_from_path(p: Path) -> float:
    stem = p.stem
    if "initial_guess_seed_" in stem:
        try:
            return int(stem.split("initial_guess_seed_")[-1])
        except (ValueError, IndexError):
            pass
    return float("inf")


def load_state_histories(data_dir: Path, num_trajectories: int):
    """Load first `num_trajectories` optimal pickles (by initial_guess_seed) and integrate."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_pkl = list(data_dir.glob("sol_history_condition_seed_*_initial_guess_seed_*.pkl"))
    all_pkl.sort(key=initial_guess_seed_from_path)

    pkl_files = []
    for p in all_pkl:
        with open(p, "rb") as f:
            data = pickle.load(f)
        if data.get("optimality") == 1:
            pkl_files.append(p)
        if len(pkl_files) >= num_trajectories:
            break

    if not pkl_files:
        raise FileNotFoundError(
            f"No .pkl files with optimality==1 found in {data_dir} (checked {len(all_pkl)} files)"
        )

    with open(pkl_files[0], "rb") as f:
        first_data = pickle.load(f)
    x0 = first_data["solver_info"][first_data["max_major_iter"]]["x"]
    timestep = len(np.asarray(get_solver_x_value(x0, "robot_0_u0")).flatten())
    parameters = parameters_from_pickle(first_data, timestep=timestep, robot_num=2)

    state_histories = []
    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        x_sol = pickle_to_x_sol(data)
        state_histories.append(integrate_dynamics(x_sol=x_sol, parameters=parameters))

    return parameters, state_histories


def draw_all_trajectories_on_ax(ax, parameters, state_histories, *, show_ylabel: bool = True) -> None:
    robot_num = parameters["robot_num"]
    robot_start_pos = parameters["robot_start_pos"]
    robot_goal_pos = parameters["robot_goal_pos"]
    obs_num = parameters["obs_num"]
    obs_pos = parameters["obs_pos"]
    obs_radius = parameters["obs_radius"]

    colors = ["blue", "green", "orange", "purple"]
    label_font = mpl.rcParams.get("font.size", 26) * 0.85

    for i in range(robot_num):
        ax.plot(
            robot_start_pos[i][0],
            robot_start_pos[i][1],
            color=colors[i],
            marker="s",
            markersize=12,
        )
        ax.text(
            robot_start_pos[i][0],
            robot_start_pos[i][1],
            " Start",
            color=colors[i],
            verticalalignment="bottom",
            horizontalalignment="right",
            fontsize=label_font,
        )
        ax.plot(
            robot_goal_pos[i][0],
            robot_goal_pos[i][1],
            color=colors[i],
            marker="o",
            markersize=12,
            alpha=0.5,
        )
        ax.text(
            robot_goal_pos[i][0],
            robot_goal_pos[i][1],
            " Goal",
            color=colors[i],
            verticalalignment="bottom",
            horizontalalignment="right",
            fontsize=label_font,
        )

    for i in range(obs_num):
        ax.add_patch(
            Circle(
                (obs_pos[i][0], obs_pos[i][1]),
                obs_radius[i],
                color="red",
                alpha=0.5,
                label="obs" if i == 0 else None,
            )
        )

    for idx, (state_x, state_y, _, _) in enumerate(state_histories):
        for i in range(robot_num):
            ax.plot(
                state_x[i],
                state_y[i],
                color=colors[i],
                alpha=0.35,
                linewidth=1.2,
                label=f"Robot {i + 1}" if idx == 0 else None,
            )

    ax.set_xlabel(r"$x$")
    if show_ylabel:
        # Horizontal math $y$ (not rotated along the axis)
        ax.set_ylabel(r"$y$", rotation=0, loc="center", labelpad=8)
    if not mpl.rcParams.get("text.usetex", False):
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.tick_params(axis="both", which="major")
    ax.set_xticks([0.0, 2.5, 5.0, 7.5, 10.0])
    ax.set_xlim(-2, 11)
    ax.set_ylim(-2, 11)
    ax.set_aspect("equal")


def shared_legend_elements():
    """Proxy artists for one legend (matches trajectory colors)."""
    return [
        Line2D([0], [0], color="blue", lw=2.5, alpha=0.9, label=r"$\text{Robot 1}$"),
        Line2D([0], [0], color="green", lw=2.5, alpha=0.9, label=r"$\text{Robot 2}$"),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            color="red",
            markerfacecolor="red",
            markeredgecolor="darkred",
            markersize=22,
            markeredgewidth=1.2,
            alpha=0.5,
            label=r"$\text{obs}$",
        ),
    ]


def plot_two_panel_paper(
    data_base: Path,
    model_name: str,
    condition_seeds: list,
    num_trajectories: int,
    output_dir: Path,
    outfile_name: str,
    use_usetex: bool,
) -> None:
    if len(condition_seeds) != 2:
        raise ValueError(f"Expected exactly 2 condition seeds, got {condition_seeds}")

    _configure_publication_fonts(use_usetex=use_usetex)
    if mpl.rcParams.get("text.usetex", False):
        mpl.rcParams["axes.formatter.use_mathtext"] = False

    fig = plt.figure(figsize=(18, 7))
    gs = GridSpec(1, 3, width_ratios=[1.0, 1.0, 0.36], wspace=0.10)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    ax_leg = fig.add_subplot(gs[0, 2])
    ax_leg.axis("off")

    model_rel = Path(model_name) / "solver_info_log"

    for ax, seed, show_y in zip((ax0, ax1), condition_seeds, (True, False)):
        data_dir = data_base / model_rel / f"condition_seed_{seed}"
        parameters, state_histories = load_state_histories(data_dir, num_trajectories)
        draw_all_trajectories_on_ax(ax, parameters, state_histories, show_ylabel=show_y)
    ax1.tick_params(axis="y", labelleft=False)

    elements = shared_legend_elements()
    ax_leg.legend(
        handles=elements,
        loc="center",
        frameon=True,
        fancybox=False,
        edgecolor="0.3",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / outfile_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def parse_seeds(s: str) -> list:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    default_model = (
        "robot_k_10_solver_behavior_k_to_use_10_parameter_original_radius_original_raw_concat_64_cond_scale_1.5_"
        "on_diffusion_steps_4_guidance_step_1_step_size_100.0_filter_0p2_obj_no_filter_sample_num_100"
    )

    parser = argparse.ArgumentParser(
        description="Paper figure: two all-trajectories panels and a shared legend on the right."
    )
    parser.add_argument(
        "--data_base",
        type=Path,
        required=True,
        help=(
            "Path to warmstart test sample data root (directory containing the model subfolder). "
            "Must be provided."
        ),
    )
    parser.add_argument("--model_name", type=str, default=default_model, help="Subfolder under data_base")
    parser.add_argument(
        "--condition_seeds",
        type=str,
        default="183,190",
        help="Exactly two comma-separated condition seeds (left, right).",
    )
    parser.add_argument("--num_trajectories", type=int, default=20)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--outfile_name",
        type=str,
        default="paper_diverse_trajectory_condition_183_190.png",
    )
    parser.add_argument("--no_usetex", action="store_true")
    args = parser.parse_args()

    seeds = parse_seeds(args.condition_seeds)
    if len(seeds) != 2:
        raise SystemExit("--condition_seeds must contain exactly two integers, e.g. 183,190")

    plot_two_panel_paper(
        data_base=args.data_base,
        model_name=args.model_name,
        condition_seeds=seeds,
        num_trajectories=args.num_trajectories,
        output_dir=args.output_dir,
        outfile_name=args.outfile_name,
        use_usetex=not args.no_usetex,
    )


if __name__ == "__main__":
    main()
