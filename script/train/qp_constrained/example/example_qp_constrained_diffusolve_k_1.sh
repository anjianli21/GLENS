#!/usr/bin/env bash

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate glens

PROJECT_DIR="/home/anjian/Desktop/project/solver_info_diffusion"
# W&B usage:
# - Always set WANDB_PROJECT_NAME and WANDB_RUN_NAME when using wandb.
# - Enable wandb by setting WANDB_API_KEY.
# - Disable wandb by keeping WANDB_API_KEY="".
WANDB_API_KEY=""
WANDB_PROJECT_NAME="qp_constrained_diffusion_experiment"
WANDB_RUN_NAME="qp_constrained_diffusolve_k_1"
DATA_ROOT_DIR="/media/anjian/T9/project/solver_info_submission/data/qp_constrained/training_data_qp_constrained_N_1000_M_100_dim_100"
RESULT_FOLDER="/media/anjian/T9/project/solver_info_submission/training_results/qp_constrained"
DATA_STAT_FILE_PATH="/media/anjian/T9/project/solver_info_submission/data/qp_constrained/stat_qp_constrained_dim_100_k_1.pkl"

WORKERS=4
WANDB_MODE="offline"

export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB__INIT_TIMEOUT="${WANDB__INIT_TIMEOUT:-300}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

python run/train/diffusion/train_diffusion.py \
    --workers=$WORKERS \
    --task_name=qp_constrained \
    --project_name="$WANDB_PROJECT_NAME" \
    --wandb_mode="$WANDB_MODE" \
    --wandb_api_key="$WANDB_API_KEY" \
    --wandb_run_name="$WANDB_RUN_NAME" \
    --data_root_dir="$DATA_ROOT_DIR" \
    --result_folder="$RESULT_FOLDER" \
    --condition_y_parameter_dim=1 \
    --condition_y_radius_dim=1 \
    --embed_y_all_dim=32 \
    --train_batch_size=256 \
    --unet_model_dim=64 \
    --unet_dim_mults="(1,2,4)" \
    --training_steps_limit=200000 \
    --data_repeat_num=1 \
    --training_timesteps=50 \
    --sampling_timesteps=50 \
    --main_model_type=diffusion \
    --data_x_channel=1 \
    --data_x_size_h=100 \
    --data_stat_type=all_neighbor_per_dim_stat \
    --data_stat_file_path="$DATA_STAT_FILE_PATH" \
    --training_data_dimension=100 \
    --k_neighbor_to_use=0 \
    --max_k_neighbor_to_encode=1 \
    --unet_type=1D \
    --train_lr=1e-5 \
    --condition_encoder_type=MLP \
    --dataset_downsample_condition_ratio=0.1 \
    --dataset_downsample_initial_guess_ratio=1.0 \
    --condition_encoder_data_type=parameter \
    --use_lr_scheduler=False \
    --auto_normalize=False \
    --condition_y_residual_encode_type=MLP \
    --cond_drop_prob=0.1 \
    --condition_fusion_type=raw_concat \
    --parameter_data_process_type=min_max \
    --parameter_data_min_list="0.0" \
    --parameter_data_max_list="30.0" \
    --radius_data_process_type=scale \
    --radius_data_scale=10.0 \
    --mlp_type=silu_layer_no_output_norm
