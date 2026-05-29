#!/usr/bin/env bash

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate glens

# Placeholders (edit these before running)
PROJECT_DIR="/directory/to/project/folder"

# NOTE: The 4 entries below (input_file_paths and output_file_paths) are ordered as:
#   (1) Diffusolve  k=1
#   (2) Diffusolve  k=10
#   (3) NS          k=10
#   (4) GLENS (NS + SB) k=10
# Fill paths in this same order so each input/output pair matches the right experiment.

# Paths to the sample pickles written by `generate_test_sample.sh`. Keep the same order as the NOTE above;
# filenames typically end with `_sample_num_100.pkl`.
input_file_paths=(
  "CHOOSE_YOUR_INPUT_FILE_PATH_1"
  "CHOOSE_YOUR_INPUT_FILE_PATH_2"
  "CHOOSE_YOUR_INPUT_FILE_PATH_3"
  "CHOOSE_YOUR_INPUT_FILE_PATH_4"
)

# Directory where k-neighbor result pickles will be written. It can be placed under a k_neighbor_results folder.
output_dir="/directory/to/output_dir"

# Here we fill in the name of the k-neighborhood result file. We can name the file end with `_k_neighbor.pkl`.
output_file_paths=(
  "${output_dir}/CHOOSE_YOUR_OUTPUT_FILENAME_1.pkl"
  "${output_dir}/CHOOSE_YOUR_OUTPUT_FILENAME_2.pkl"
  "${output_dir}/CHOOSE_YOUR_OUTPUT_FILENAME_3.pkl"
  "${output_dir}/CHOOSE_YOUR_OUTPUT_FILENAME_4.pkl"
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
