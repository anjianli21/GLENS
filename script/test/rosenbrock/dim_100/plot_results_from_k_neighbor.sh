#!/usr/bin/env bash

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate glens

# NOTE: The 4 entries in `file_paths` and `legend_names` below use this order:
#   (1) Diffusolve  k=1
#   (2) Diffusolve  k=10
#   (3) NS          k=10
#   (4) GLENS (NS + SB) k=10
# Fill paths and legend labels in this same order (counts must match).

# Placeholders (edit these before running)
PROJECT_DIR="/directory/to/project/folder"

# Pickles produced by `compute_k_neighbor.sh` (same order as the NOTE above); filenames typically end with `_k_neighbor.pkl`.
file_paths=(
  "CHOOSE_YOUR_K_NEIGHBOR_RESULT_PATH_1"
  "CHOOSE_YOUR_K_NEIGHBOR_RESULT_PATH_2"
  "CHOOSE_YOUR_K_NEIGHBOR_RESULT_PATH_3"
  "CHOOSE_YOUR_K_NEIGHBOR_RESULT_PATH_4"
)

plot_name="CHOOSE_YOUR_PLOT_NAME"

# Directory where figure and stats files will be written. It can be placed under something like `cdf_plots/` or `k_neighbor_cumulative_cdf/`.
output_dir="/directory/to/cdf_plots_dir"

legend_names=(
  "DiffuSolve"
  "DiffuSolve-k"
  "NS-only"
  "GLENS"
)

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

file_paths_str=$(IFS=','; echo "${file_paths[*]}")
legend_names_str=$(IFS=','; echo "${legend_names[*]}")

python run/test/task/rosenbrock/plot_results_from_k_neighbor.py \
  --file_paths "${file_paths_str}" \
  --legend_names "${legend_names_str}" \
  --plot_name "${plot_name}" \
  --output_dir "${output_dir}" \
  --stats_subset "all" \
  --x_bin_size 1 \
  --x_max 25
