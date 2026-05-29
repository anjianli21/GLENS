#!/usr/bin/env bash

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate glens

# Placeholders (edit these before running)
PROJECT_DIR="/directory/to/project/folder"

ERROR_TOLERANCE="1e-3"

# NOTE: The 4 entries below (input_file_paths and output_file_paths) are ordered as:
#   (1) Diffusolve  k=1
#   (2) Diffusolve  k=10
#   (3) NS          k=10
#   (4) GLENS (NS + SB) k=10
# Fill paths in this same order so each input/output pair matches the right experiment.

# Paths to the sample pickles written by `generate_test_sample.sh`. Keep the same order as the NOTE above;
# filenames typically end with `_sample_num_100.pkl` (often with extra tokens from the wandb run name).
input_file_paths=(
  "CHOOSE_YOUR_INPUT_SAMPLE_PKL_1"
  "CHOOSE_YOUR_INPUT_SAMPLE_PKL_2"
  "CHOOSE_YOUR_INPUT_SAMPLE_PKL_3"
  "CHOOSE_YOUR_INPUT_SAMPLE_PKL_4"
)

# Directory where k-neighbor result pickles will be written. It can be placed under a `k_neighbor_results/` folder.
output_dir="/directory/to/k_neighbor_results_dir"

# Output pickle paths from `compute_k_neighbor_lbfgsb.py` (same order as `input_file_paths`). Filenames typically end with `_k_neighbor.pkl`.
output_file_paths=(
  "${output_dir}/CHOOSE_YOUR_K_NEIGHBOR_OUTPUT_1.pkl"
  "${output_dir}/CHOOSE_YOUR_K_NEIGHBOR_OUTPUT_2.pkl"
  "${output_dir}/CHOOSE_YOUR_K_NEIGHBOR_OUTPUT_3.pkl"
  "${output_dir}/CHOOSE_YOUR_K_NEIGHBOR_OUTPUT_4.pkl"
)

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# Convert arrays to comma-separated strings for Python script
input_file_paths_str=$(IFS=','; echo "${input_file_paths[*]}")
output_file_paths_str=$(IFS=','; echo "${output_file_paths[*]}")

python run/test/task/rosenbrock/compute_k_neighbor_lbfgsb.py \
  --file_paths "${input_file_paths_str}" \
  --output_paths "${output_file_paths_str}" \
  --error_type "tolerance" \
  --error_tolerance "${ERROR_TOLERANCE}"
