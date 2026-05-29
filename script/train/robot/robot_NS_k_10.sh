#!/usr/bin/env bash

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate glens

# Placeholders (edit these before running)
PROJECT_DIR="/directory/to/project/folder"
# W&B usage:
# - Always set WANDB_PROJECT_NAME and WANDB_RUN_NAME when using wandb.
# - Enable wandb by setting WANDB_API_KEY.
# - Disable wandb by keeping WANDB_API_KEY="".
WANDB_API_KEY=""
WANDB_PROJECT_NAME="CHOOSE_YOUR_WANDB_PROJECT_NAME"
WANDB_RUN_NAME="CHOOSE_YOUR_WANDB_RUN_NAME"
DATA_ROOT_DIR="/directory/to/training_data_dir"
RESULT_FOLDER="/directory/to/results_dir"
DATA_STAT_FILE_PATH="/directory/to/stat.pkl"

WORKERS=4
WANDB_MODE="offline"

export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB__INIT_TIMEOUT="${WANDB__INIT_TIMEOUT:-300}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

python run/train/diffusion/train_diffusion.py \
    --workers=$WORKERS \
    --task_name=robot \
    --project_name="$WANDB_PROJECT_NAME" \
    --wandb_mode="$WANDB_MODE" \
    --wandb_api_key="$WANDB_API_KEY" \
    --wandb_run_name="$WANDB_RUN_NAME" \
    --data_root_dir="$DATA_ROOT_DIR" \
    --result_folder="$RESULT_FOLDER" \
    --condition_y_parameter_dim=6 \
    --condition_y_radius_dim=1 \
    --embed_y_all_dim=128 \
    --train_batch_size=512 \
    --unet_model_dim=64 \
    --unet_dim_mults="(1,2,4)" \
    --training_steps_limit=100000 \
    --data_repeat_num=1 \
    --training_timesteps=50 \
    --sampling_timesteps=50 \
    --main_model_type=diffusion \
    --data_x_channel=5 \
    --data_x_size_h=20 \
    --data_x_process_type=add_t_into_channel \
    --data_stat_type=min_max_stat \
    --data_stat_file_path="$DATA_STAT_FILE_PATH" \
    --training_data_dimension=20 \
    --k_neighbor_to_use=0,1,2,3,4,5,6,7,8,9 \
    --max_k_neighbor_to_encode=10 \
    --unet_type=1D \
    --train_lr=1e-5 \
    --condition_encoder_type=MLP \
    --dataset_downsample_condition_ratio=0.5 \
    --dataset_downsample_initial_guess_ratio=1.0 \
    --condition_encoder_data_type=parameter,radius \
    --use_lr_scheduler=False \
    --auto_normalize=False \
    --cond_drop_prob=0.1 \
    --filter_neighborhood_by_threshold=True \
    --filter_neighborhood_by_threshold_threshold=0.2 \
    --filter_data_by_objective=False \
    --filter_data_by_objective_threshold=12.0 \
    --condition_fusion_type=raw_concat \
    --parameter_data_process_type=original \
    --radius_data_process_type=original
