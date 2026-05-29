#!/usr/bin/env bash

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate glens

# Hard-coded W&B run names (task = robot; same order as the 4 runs below).
wandb_run_name_list=(
  "robot_diffusolve_k_1"
)

# Placeholders (edit these before running)
# Fill `checkpoint_parent_dir_list`, `solver_behavior_checkpoint_parent_dir_list`, and
# `data_stat_file_path_list` in this same order.
PROJECT_DIR="/directory/to/project/folder"
# W&B usage:
# - Always set project_name and wandb_run_name_list when using wandb.
# - Enable wandb by setting WANDB_API_KEY.
# - Disable wandb by keeping WANDB_API_KEY="".
WANDB_API_KEY=""

project_name="CHOOSE_YOUR_TEST_PROJECT_NAME"

DATA_ROOT_DIR="/directory/to/training_data_dir"
LOGS_DIR="/directory/to/logs_dir"
TEST_SAMPLE_DIRECTORY="/directory/to/test_sample_directory"

# One checkpoint parent dir per run (4 runs total).
# Uses the latest timestamp subdirectory under each parent, then `checkpoint/` (fallback: `${parent}/checkpoint`).
checkpoint_parent_dir_list=(
  "CHOOSE_YOUR_CHECKPOINT_PARENT_DIR_1"
)

solver_behavior_checkpoint_name="model-best_validation.pt"

# One solver-behavior checkpoint parent dir per run (4 runs total).
solver_behavior_checkpoint_parent_dir_list=(
  "CHOOSE_YOUR_SOLVER_BEHAVIOR_CHECKPOINT_PARENT_DIR_1"
)

# One stat pickle per run (4 runs total). Often the same file is reused across k; set duplicates if needed.
data_stat_file_path_list=(
  "CHOOSE_YOUR_DATA_STAT_FILE_PATH_1"
)

export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB__INIT_TIMEOUT="${WANDB__INIT_TIMEOUT:-300}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

condition_encoder_data_type_strings_list=("parameter_original_raw_concat_64")
condition_encoder_data_type_list=('parameter')
unet_dim_list=(64)
unet_dim_mults_list=('(1,2,4)')
embed_y_all_dim_list=(64)
auto_normalize_list=(False)

cond_scale_list=(1.5)
downsample_condition_ratio_list=(1.0)
downsample_initial_guess_ratio_list=(0.01)
k_neighbor_to_use_list=(0)
condition_fusion_type_list=('raw_concat')
parameter_data_process_type_list=('original')
radius_data_process_type_list=('original')

data_x_process_type_list=('add_t_into_channel')
radius_square_grad_data_process_type_list=('original')

solver_behavior_data_type_list=('parameter,radius_square_grad')
solver_behavior_encoder_data_type_list=('parameter')
solver_behavior_condition_y_parameter_dim_list=(6)
solver_behavior_output_data_type_list=('radius_square_grad')
solver_behavior_unet_type_list=('1D')
solver_behavior_unet_model_dim_list=(64)
solver_behavior_unet_dim_mults_list=('(1,2,4)')
solver_behavior_embed_y_all_dim_list=(64)
solver_behavior_dropout_list=(0.0)

solver_behavior_apply_on_max_diffusion_steps_list=(-1)
solver_behavior_guidance_step_per_diffusion_step_list=(1)
solver_behavior_step_size_list=(10.0)
solver_behavior_k_to_use_list=(10)

###########################################################################################################################
# Fixed parameter (Robot dynamics live in 20-D state space for this experiment folder.)
problem_dim=20
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
    radius_data_process_type=${radius_data_process_type_list[$i]}
    embed_y_all_dim=${embed_y_all_dim_list[$i]}
    downsample_condition_ratio=${downsample_condition_ratio_list[$i]}
    downsample_initial_guess_ratio=${downsample_initial_guess_ratio_list[$i]}
    data_x_process_type=${data_x_process_type_list[$i]}
    radius_square_grad_data_process_type=${radius_square_grad_data_process_type_list[$i]}

    data_root_dir="${DATA_ROOT_DIR}"
    test_sample_directory="${TEST_SAMPLE_DIRECTORY}"
    data_stat_file_path="${data_stat_file_path_list[$i]}"

    checkpoint_parent_dir="${checkpoint_parent_dir_list[$i]}"
    timestamp_dir=$(ls -td "${checkpoint_parent_dir}"/*/ 2>/dev/null | head -n 1)
    if [ -n "$timestamp_dir" ]; then
        checkpoint_results_folder="${timestamp_dir}checkpoint"
    else
        checkpoint_results_folder="${checkpoint_parent_dir}/checkpoint"
    fi

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
    solver_behavior_k_to_use=${solver_behavior_k_to_use_list[$i]}

    python run/test/diffusion/test_diffusion_sample_with_solver_behavior.py \
        --task_name robot \
        --project_name "${project_name}" \
        --result_folder "${LOGS_DIR}/robot" \
        --wandb_mode offline \
        --wandb_api_key "$WANDB_API_KEY" \
        --wandb_run_name "${wandb_run_name_list[$i]}" \
        --data_root_dir "${data_root_dir}" \
        --dataset_downsample_condition_ratio "${downsample_condition_ratio}" \
        --dataset_downsample_initial_guess_ratio "${downsample_initial_guess_ratio}" \
        --condition_y_parameter_dim 6 \
        --condition_y_radius_dim 1 \
        --embed_y_all_dim "${embed_y_all_dim}" \
        --unet_model_dim "${unet_dim}" \
        --unet_dim_mults "${unet_dim_mults}" \
        --test_sample_num 100 \
        --training_timesteps 50 \
        --sampling_timesteps 50 \
        --checkpoint_results_folder "${checkpoint_results_folder}" \
        --checkpoint_name "${checkpoint_name}" \
        --main_model_type diffusion \
        --data_x_process_type "${data_x_process_type}" \
        --data_x_channel 5 \
        --data_x_size_h "${problem_dim}" \
        --data_stat_type min_max_stat \
        --data_stat_file_path "${data_stat_file_path}" \
        --training_data_dimension "${problem_dim}" \
        --k_neighbor_to_use "${k_neighbor_to_use}" \
        --max_k_neighbor_to_encode 10 \
        --unet_type 1D \
        --condition_encoder_data_type "${condition_encoder_data_type}" \
        --use_lr_scheduler False \
        --test_data_type test \
        --test_batch_size 32 \
        --condition_encoder_type MLP \
        --test_sample_directory "${test_sample_directory}" \
        --auto_normalize "${auto_normalize}" \
        --cond_scale "${cond_scale}" \
        --filter_data_by_objective False \
        --condition_fusion_type "${condition_fusion_type}" \
        --parameter_data_process_type "${parameter_data_process_type}" \
        --radius_data_process_type "${radius_data_process_type}" \
        --radius_square_grad_data_process_type "${radius_square_grad_data_process_type}" \
        --solver_behavior_data_type "${solver_behavior_data_type}" \
        --solver_behavior_encoder_data_type "${solver_behavior_encoder_data_type}" \
        --solver_behavior_condition_y_parameter_dim "${solver_behavior_condition_y_parameter_dim}" \
        --solver_behavior_output_data_type "${solver_behavior_output_data_type}" \
        --solver_behavior_unet_type "${solver_behavior_unet_type}" \
        --solver_behavior_unet_model_dim "${solver_behavior_unet_model_dim}" \
        --solver_behavior_unet_dim_mults "${solver_behavior_unet_dim_mults}" \
        --solver_behavior_embed_y_all_dim "${solver_behavior_embed_y_all_dim}" \
        --solver_behavior_dropout "${solver_behavior_dropout}" \
        --solver_behavior_checkpoint_path "${solver_behavior_checkpoint_path}" \
        --solver_behavior_apply_on_max_diffusion_steps "${solver_behavior_apply_on_max_diffusion_steps}" \
        --solver_behavior_guidance_step_per_diffusion_step "${solver_behavior_guidance_step_per_diffusion_step}" \
        --solver_behavior_step_size "${solver_behavior_step_size}"
done
