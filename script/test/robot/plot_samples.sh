#! /bin/bash
# One paper figure: two all-trajectories panels (condition seeds 183 and 190), shared legend on the right.

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate glens

# Placeholders (edit these before running)
PROJECT_DIR="/directory/to/project/folder"
DATA_BASE="/directory/to/test_sample_data"
MODEL_NAME="specify_name_of_the_model_used_to_sample_the_data"
OUTPUT_DIR="/directory/to/output_dir"
NUM_TRAJECTORIES=20


cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

python run/test/task/robot/plot_samples.py \
  --data_base "${DATA_BASE}" \
  --model_name "${MODEL_NAME}" \
  --condition_seeds "183,190" \
  --num_trajectories "${NUM_TRAJECTORIES}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@" || exit 1

echo "Done. Figure saved under ${OUTPUT_DIR}"
