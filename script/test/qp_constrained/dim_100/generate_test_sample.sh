#!/usr/bin/env bash

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate glens

# Hard-coded W&B run names (problem_type = qp_constrained; same order as the 4 runs below).
wandb_run_name_list=(
  "qp_constrained_diffusolve_k_1"
  "qp_constrained_diffusolve_k_10"
  "qp_constrained_ns_k_10"
  "qp_constrained_glense_k_10"
)

# Placeholders (edit these before running)
# NOTE: The 4 test-sample runs in this script are ordered as:
#   (1) Diffusolve  k=1
#   (2) Diffusolve  k=10
#   (3) NS          k=10
#   (4) GLENS (NS + solver-behavior guidance) k=10
# Fill `checkpoint_parent_dir_list`, `solver_behavior_checkpoint_parent_dir_list`, and
# `data_stat_file_path_list` in this same order.
PROJECT_DIR="/directory/to/project/folder"
# W&B usage:
# - Always set WANDB_PROJECT_NAME and wandb_run_name_list when using wandb.
# - Enable wandb by setting WANDB_API_KEY.
# - Disable wandb by keeping WANDB_API_KEY="".
WANDB_API_KEY=""
WANDB_PROJECT_NAME="CHOOSE_YOUR_WANDB_PROJECT_NAME"

project_name="CHOOSE_YOUR_TEST_PROJECT_NAME"

DATA_ROOT_DIR="/directory/to/training_data_dir"
LOGS_DIR="/directory/to/logs_dir"
TEST_SAMPLE_DIRECTORY="/directory/to/test_sample_directory"

# One checkpoint parent dir per run (4 runs total).
# This script will automatically use the *latest* run subdirectory under each parent dir,
# and then look for checkpoints under its `checkpoint/` folder (fallback: `${parent}/checkpoint`).
checkpoint_parent_dir_list=(
  "CHOOSE_YOUR_CHECKPOINT_PARENT_DIR_1"
  "CHOOSE_YOUR_CHECKPOINT_PARENT_DIR_2"
  "CHOOSE_YOUR_CHECKPOINT_PARENT_DIR_3"
  "CHOOSE_YOUR_CHECKPOINT_PARENT_DIR_4"
)

# The solver-behavior checkpoint filename inside the checkpoint folder.
solver_behavior_checkpoint_name="model-best_validation.pt"

# One solver-behavior checkpoint parent dir per run (4 runs total).
# Same logic as `checkpoint_parent_dir_list`: uses the latest run subdirectory if present.
solver_behavior_checkpoint_parent_dir_list=(
  "CHOOSE_YOUR_SOLVER_BEHAVIOR_CHECKPOINT_PARENT_DIR_1"
  "CHOOSE_YOUR_SOLVER_BEHAVIOR_CHECKPOINT_PARENT_DIR_2"
  "CHOOSE_YOUR_SOLVER_BEHAVIOR_CHECKPOINT_PARENT_DIR_3"
  "CHOOSE_YOUR_SOLVER_BEHAVIOR_CHECKPOINT_PARENT_DIR_4"
)

# One stat file per run (4 runs total)
data_stat_file_path_list=(
  "CHOOSE_YOUR_DATA_STAT_FILE_PATH_1"
  "CHOOSE_YOUR_DATA_STAT_FILE_PATH_2"
  "CHOOSE_YOUR_DATA_STAT_FILE_PATH_3"
  "CHOOSE_YOUR_DATA_STAT_FILE_PATH_4"
)

export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB__INIT_TIMEOUT="${WANDB__INIT_TIMEOUT:-300}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

condition_encoder_data_type_strings_list=("parameter_minmax_raw_concat_32"
"parameter_minmax_raw_concat_32" "parameter_minmax_radius_scale10_raw_concat_32"
"parameter_minmax_radius_scale10_raw_concat_32")
condition_encoder_data_type_list=('parameter' 'parameter' 'parameter,radius' 'parameter,radius')
condition_y_parameter_dim_list=(1 1 1 1)
unet_dim_list=(64 64 64 64)
unet_dim_mults_list=('(1,2,4)' '(1,2,4)' '(1,2,4)' '(1,2,4)')
embed_y_all_dim_list=(32 32 32 32)
auto_normalize_list=(False False False False)

cond_scale_list=(1.5 1.5 1.5 1.5)
downsample_initial_guess_ratio_list=(0.01 0.01 0.01 0.01)
k_neighbor_to_use_list=(0 0 0 0)
condition_fusion_type_list=('raw_concat' 'raw_concat' 'raw_concat' 'raw_concat')
parameter_data_process_type_list=('min_max' 'min_max' 'min_max' 'min_max')
# One comma-separated string per run (parsed by parse_float_list in Python); do not reuse these names inside the loop.
parameter_data_min_list=('0.0' '0.0' '0.0' '0.0')
parameter_data_max_list=('30.0' '30.0' '30.0' '30.0')
radius_data_process_type_list=('scale' 'scale' 'scale' 'scale')
radius_data_scale_list=(10 10 10 10)
mlp_type_list=('silu_layer_no_output_norm' 'silu_layer_no_output_norm' 'silu_layer_no_output_norm' 'silu_layer_no_output_norm')

# Solver behavior model parameters
solver_behavior_data_type_list=('parameter,radius_square_grad' 'parameter,radius_square_grad' 'parameter,radius_square_grad' 'parameter,radius_square_grad')
solver_behavior_encoder_data_type_list=('parameter' 'parameter' 'parameter' 'parameter')
solver_behavior_condition_y_parameter_dim_list=(1 1 1 1)
solver_behavior_output_data_type_list=('radius_square_grad' 'radius_square_grad' 'radius_square_grad' 'radius_square_grad')
solver_behavior_unet_type_list=('1D' '1D' '1D' '1D')
solver_behavior_unet_model_dim_list=(64 64 64 64)
solver_behavior_unet_dim_mults_list=('(1,2,4)' '(1,2,4)' '(1,2,4)' '(1,2,4)')
solver_behavior_embed_y_all_dim_list=(64 64 64 64)
solver_behavior_dropout_list=(0.0 0.0 0.0 0.0)

radius_square_grad_data_process_type_list=('scale' 'scale' 'scale' 'scale')
radius_square_grad_data_scale_list=(10 10 10 10)

solver_behavior_apply_on_max_diffusion_steps_list=(-1 -1 -1 4)
solver_behavior_guidance_step_per_diffusion_step_list=(1 1 1 1)
solver_behavior_step_size_list=(10.0 10.0 10.0 100.0)

###########################################################################################################################
# Fixed parameter
problem_dim=100
checkpoint_name="model-best_validation.pt"

for i in "${!wandb_run_name_list[@]}"; do
    condition_encoder_data_type_strings=${condition_encoder_data_type_strings_list[$i]}
    condition_encoder_data_type=${condition_encoder_data_type_list[$i]}
    unet_dim=${unet_dim_list[$i]}
    unet_dim_mults=${unet_dim_mults_list[$i]}
    auto_normalize=${auto_normalize_list[$i]}
    cond_scale=${cond_scale_list[$i]}
    k_neighbor_to_use=${k_neighbor_to_use_list[$i]}
    condition_fusion_type=${condition_fusion_type_list[$i]}
    parameter_data_process_type=${parameter_data_process_type_list[$i]}
    parameter_data_min_str="${parameter_data_min_list[$i]}"
    parameter_data_max_str="${parameter_data_max_list[$i]}"
    radius_data_process_type=${radius_data_process_type_list[$i]}
    radius_data_scale=${radius_data_scale_list[$i]}
    embed_y_all_dim=${embed_y_all_dim_list[$i]}
    downsample_initial_guess_ratio=${downsample_initial_guess_ratio_list[$i]}
    mlp_type=${mlp_type_list[$i]}
    condition_y_parameter_dim=${condition_y_parameter_dim_list[$i]}

    data_root_dir="${DATA_ROOT_DIR}"
    test_sample_directory="${TEST_SAMPLE_DIRECTORY}"
    data_stat_file_path="${data_stat_file_path_list[$i]}"

    # find the timestamp directory and add to the checkpoint results folder
    # Checkpoint dir
    checkpoint_parent_dir="${checkpoint_parent_dir_list[$i]}"
    timestamp_dir=$(ls -td "${checkpoint_parent_dir}"/*/ 2>/dev/null | head -n 1)
    if [ -n "$timestamp_dir" ]; then
        checkpoint_results_folder="${timestamp_dir}checkpoint"
    else
        checkpoint_results_folder="${checkpoint_parent_dir}/checkpoint"
    fi

    # dynamics parameters
    solver_behavior_data_type=${solver_behavior_data_type_list[$i]}
    solver_behavior_encoder_data_type=${solver_behavior_encoder_data_type_list[$i]}
    solver_behavior_condition_y_parameter_dim=${solver_behavior_condition_y_parameter_dim_list[$i]}
    solver_behavior_output_data_type=${solver_behavior_output_data_type_list[$i]}
    solver_behavior_unet_type=${solver_behavior_unet_type_list[$i]}
    solver_behavior_unet_model_dim=${solver_behavior_unet_model_dim_list[$i]}
    solver_behavior_unet_dim_mults=${solver_behavior_unet_dim_mults_list[$i]}
    solver_behavior_embed_y_all_dim=${solver_behavior_embed_y_all_dim_list[$i]}
    solver_behavior_dropout=${solver_behavior_dropout_list[$i]}
    solver_behavior_checkpoint_parent_dir=${solver_behavior_checkpoint_parent_dir_list[$i]}
    solver_behavior_timestamp_dir=$(ls -td "${solver_behavior_checkpoint_parent_dir}"/*/ 2>/dev/null | head -n 1)
    if [ -n "$solver_behavior_timestamp_dir" ]; then
        solver_behavior_checkpoint_folder="${solver_behavior_timestamp_dir}checkpoint"
    else
        solver_behavior_checkpoint_folder="${solver_behavior_checkpoint_parent_dir}/checkpoint"
    fi
    solver_behavior_checkpoint_path="${solver_behavior_checkpoint_folder}/${solver_behavior_checkpoint_name}"
    solver_behavior_apply_on_max_diffusion_steps=${solver_behavior_apply_on_max_diffusion_steps_list[$i]}
    solver_behavior_guidance_step_per_diffusion_step=${solver_behavior_guidance_step_per_diffusion_step_list[$i]}
    solver_behavior_step_size=${solver_behavior_step_size_list[$i]}
    radius_square_grad_data_process_type=${radius_square_grad_data_process_type_list[$i]}
    radius_square_grad_data_scale=${radius_square_grad_data_scale_list[$i]}

    python run/test/diffusion/test_diffusion_sample_with_solver_behavior.py \
        --task_name qp_constrained \
        --project_name ${project_name} \
        --result_folder ${LOGS_DIR}/qp_constrained \
        --wandb_mode offline \
        --wandb_api_key "$WANDB_API_KEY" \
        --wandb_run_name "${wandb_run_name_list[$i]}" \
        --data_root_dir ${data_root_dir} \
        --dataset_downsample_condition_ratio 1.0 \
        --dataset_downsample_initial_guess_ratio ${downsample_initial_guess_ratio} \
        --condition_y_parameter_dim ${condition_y_parameter_dim} \
        --condition_y_radius_dim 1 \
        --embed_y_all_dim ${embed_y_all_dim} \
        --unet_model_dim ${unet_dim} \
        --unet_dim_mults ${unet_dim_mults} \
        --test_sample_num 100 \
        --training_timesteps 50 \
        --sampling_timesteps 50 \
        --checkpoint_results_folder ${checkpoint_results_folder} \
        --checkpoint_name ${checkpoint_name} \
        --main_model_type diffusion \
        --data_x_channel 1 \
        --data_x_size_h ${problem_dim} \
        --data_x_size_w 1 \
        --data_stat_type all_neighbor_per_dim_stat \
        --data_stat_file_path ${data_stat_file_path} \
        --training_data_dimension ${problem_dim} \
        --k_neighbor_to_use ${k_neighbor_to_use} \
        --max_k_neighbor_to_encode 10 \
        --unet_type 1D \
        --condition_encoder_data_type ${condition_encoder_data_type} \
        --use_lr_scheduler False \
        --test_data_type test \
        --test_batch_size 32 \
        --condition_encoder_type MLP \
        --test_sample_directory ${test_sample_directory} \
        --auto_normalize ${auto_normalize} \
        --cond_scale ${cond_scale} \
        --condition_fusion_type ${condition_fusion_type} \
        --parameter_data_process_type ${parameter_data_process_type} \
        --parameter_data_min_list "${parameter_data_min_str}" \
        --parameter_data_max_list "${parameter_data_max_str}" \
        --radius_data_process_type ${radius_data_process_type} \
        --radius_data_scale ${radius_data_scale} \
        --radius_square_grad_data_process_type ${radius_square_grad_data_process_type} \
        --radius_square_grad_data_scale ${radius_square_grad_data_scale} \
        --solver_behavior_data_type ${solver_behavior_data_type} \
        --solver_behavior_encoder_data_type ${solver_behavior_encoder_data_type} \
        --solver_behavior_condition_y_parameter_dim ${solver_behavior_condition_y_parameter_dim} \
        --solver_behavior_output_data_type ${solver_behavior_output_data_type} \
        --solver_behavior_unet_type ${solver_behavior_unet_type} \
        --solver_behavior_unet_model_dim ${solver_behavior_unet_model_dim} \
        --solver_behavior_unet_dim_mults ${solver_behavior_unet_dim_mults} \
        --solver_behavior_embed_y_all_dim ${solver_behavior_embed_y_all_dim} \
        --solver_behavior_dropout ${solver_behavior_dropout} \
        --solver_behavior_checkpoint_path ${solver_behavior_checkpoint_path} \
        --solver_behavior_apply_on_max_diffusion_steps ${solver_behavior_apply_on_max_diffusion_steps} \
        --solver_behavior_guidance_step_per_diffusion_step ${solver_behavior_guidance_step_per_diffusion_step} \
        --solver_behavior_step_size ${solver_behavior_step_size} \
        --mlp_type ${mlp_type}
done

