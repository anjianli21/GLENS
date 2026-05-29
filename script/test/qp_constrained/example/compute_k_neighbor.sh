#!/usr/bin/env bash

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate glens

# Placeholders (edit these before running)
PROJECT_DIR="/home/anjian/Desktop/project/solver_info_diffusion"

# NOTE: The 4 entries below (input_file_paths and output_file_paths) are ordered as:
#   (1) Diffusolve  k=1
#   (2) Diffusolve  k=10
#   (3) NS          k=10
#   (4) GLENS (NS + SB) k=10
# Fill paths in this same order so each input/output pair matches the right experiment.

# Paths to the sample pickles written by `generate_test_sample.sh`. Keep the same order as the NOTE above;
# filenames typically end with `_sample_num_100.pkl`.
input_file_paths=(
  "/media/anjian/T9/project/solver_info_submission/test_sample_data/qp_constrained/qp_constrained_generate_test_sample/qp_constrained_diffusolve_k_1_sample_num_100.pkl"
  "/media/anjian/T9/project/solver_info_submission/test_sample_data/qp_constrained/qp_constrained_generate_test_sample/qp_constrained_diffusolve_k_10_sample_num_100.pkl"
  "/media/anjian/T9/project/solver_info_submission/test_sample_data/qp_constrained/qp_constrained_generate_test_sample/qp_constrained_ns_k_10_sample_num_100.pkl"
  "/media/anjian/T9/project/solver_info_submission/test_sample_data/qp_constrained/qp_constrained_generate_test_sample/qp_constrained_glense_k_10_sample_num_100.pkl"
)

# Directory where k-neighbor result pickles will be written. It can be placed under a `k_neighbor_results/` folder.
output_dir="/media/anjian/T9/project/solver_info_submission/test_results/qp_constrained/qp_constrained_generate_test_sample/k_neighbor_results"

# Output pickle paths from `compute_k_neighbor.py` (same order as `input_file_paths`). Filenames typically end with `_k_neighbor.pkl`.
output_file_paths=(
  "${output_dir}/qp_constrained_diffusolve_k_1_sample_num_100_k_neighbor.pkl"
  "${output_dir}/qp_constrained_diffusolve_k_10_sample_num_100_k_neighbor.pkl"
  "${output_dir}/qp_constrained_ns_k_10_sample_num_100_k_neighbor.pkl"
  "${output_dir}/qp_constrained_glense_k_10_sample_num_100_k_neighbor.pkl"
)

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# Convert arrays to comma-separated strings for Python script
input_file_paths_str=$(IFS=','; echo "${input_file_paths[*]}")
output_file_paths_str=$(IFS=','; echo "${output_file_paths[*]}")

python run/test/task/qp_constrained/compute_k_neighbor.py \
  --file_paths "${input_file_paths_str}" \
  --output_paths "${output_file_paths_str}" \
  --step_size 0.1 \
  --max_iter 100
