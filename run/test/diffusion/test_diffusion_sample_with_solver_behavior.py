from run.train.train_utils import set_random_seed, create_logger
from dataset import build_dataloader
from dataset.robot.legacy_car_keys import canonical_stat_dict

from accelerate import Accelerator, DataLoaderConfiguration

import torch

import datetime
import pickle
import os
import numpy as np

import logging
from ema_pytorch import EMA

from model.trajectory_solver_behavior_model import TrajectorySolverBehaviorModel
from run.diffusion_parse_args_utils import parse_args


def main():
    # Read the config from yaml file #######################################################################################
    args = parse_config()

    model_params = {}
    for key, val in vars(args).items():
        model_params[key] = val

    # Set up random seed for model initialization
    set_random_seed(args.random_seed)

    result_folder = f"{args.result_folder}/{args.project_name}/{args.wandb_run_name}"

    # Configure wandb mode
    if args.wandb_mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
    elif args.wandb_mode == "online":
        os.environ["WANDB_MODE"] = "online"
    else:
        raise ValueError(f"wandb_mode {args.wandb_mode} is not supported!")

    # Configure accelerator ##########################################################################################
    dataloader_config = DataLoaderConfiguration(split_batches=True)
    accelerator = Accelerator(
        dataloader_config=dataloader_config,
        mixed_precision='no',
    )

    # log output #########################################################################################################
    output_dir = f"{result_folder}/output"
    os.makedirs(output_dir, exist_ok=True)

    log_file = f"{output_dir}/log_test_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"

    # Determine logging level based on main process
    log_level = logging.INFO if accelerator.is_main_process else logging.WARN
    logger = create_logger(log_file, rank=0 if accelerator.is_main_process else 1, log_level=log_level)

    # log to file
    if accelerator.is_main_process:
        logger.info('**********************Start logging**********************')
        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
        logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

        for key, val in vars(args).items():
            logger.info('{:16} {}'.format(key, val))

    # Build dataset ####################################################################################################

    test_loader = build_dataloader(
        logger=logger,
        split=args.test_data_type,
        args=args,
    )

    #####################################################################################################################
    # Load and prepare all models and dataloader together
    # Load diffusion model
    if args.main_model_type != 'diffusion':
        raise ValueError(f"main_model_type must be 'diffusion', got: {args.main_model_type}")
    from model.trajectory_diffusion_model import TrajectoryDiffusion
    model = TrajectoryDiffusion(model_params=model_params)

    # Load solver behavior model
    solver_behavior_model = TrajectorySolverBehaviorModel(model_params=model_params)

    # Prepare all models and dataloader together
    model, solver_behavior_model, test_loader = accelerator.prepare(
        model, solver_behavior_model, test_loader
    )
    device = accelerator.device

    #####################################################################################################################
    # Load diffusion checkpoint
    data = torch.load(f"{args.checkpoint_results_folder}/{args.checkpoint_name}", map_location=device)
    logger.info(f"Loaded checkpoint from {args.checkpoint_results_folder}/{args.checkpoint_name}")

    # Load diffusion model weights
    model = accelerator.unwrap_model(model)
    model.load_state_dict(data['model'])

    # Load ema model for diffusion
    ema = EMA(model).to(device)
    ema.load_state_dict(data['ema'])
    ema_model = ema.model
    ema_model.eval()

    #####################################################################################################################
    # Load solver behavior checkpoint
    solver_behavior_model_data = torch.load(f"{args.solver_behavior_checkpoint_path}", map_location=device)
    logger.info(f"Loaded checkpoint from {args.solver_behavior_checkpoint_path}")

    solver_behavior_model = accelerator.unwrap_model(solver_behavior_model)
    solver_behavior_model.load_state_dict(solver_behavior_model_data['model'])

    solver_behavior_ema = EMA(solver_behavior_model).to(device)
    solver_behavior_ema.load_state_dict(solver_behavior_model_data['ema'])
    solver_behavior_ema_model = solver_behavior_ema.model
    solver_behavior_ema_model.eval()

    #####################################################################################################################
    # Process all batches in a single loop
    sample_data = {}
    if args.task_name in {'himmelblau', 'levy', 'rosenbrock', 'qp_constrained'}:
        sample_data['x'] = []
        sample_data['condition_lambda_value'] = []
    elif args.task_name == 'robot':
        sample_data['t_final'] = []
        sample_data['robot_0_u0'] = []
        sample_data['robot_0_u1'] = []
        sample_data['robot_1_u0'] = []
        sample_data['robot_1_u1'] = []
        sample_data['condition_data'] = []
    else:
        raise ValueError(
            f"Invalid task name: {args.task_name}. Must be one of: "
            "himmelblau, levy, rosenbrock, qp_constrained, robot"
        )

    sample_data['condition_seed_string'] = []

    logger.info(f"Processing {len(test_loader)} batches...")
    for batch in test_loader:

        # Get sample results
        sample_results = model.sample(
            batch=batch,
            sample_num=args.test_sample_num,
            args=args,
            solver_behavior_model=solver_behavior_ema_model,
        )
        sample_results_np = sample_results.cpu().numpy()
        sample_results_np = np.squeeze(sample_results_np)

        if args.training_data_type != 'k_neighbor_single_point':
            raise ValueError(f"Invalid training data type: {args.training_data_type}")

        # Unnormalize the data #########################################################################################################
        with open(args.data_stat_file_path, "rb") as f:
            data_stat = pickle.load(f)
        data_stat = canonical_stat_dict(data_stat)
        
        if args.data_stat_type == 'all_neighbor_per_dim_stat':
            if args.task_name in {'himmelblau', 'levy', 'rosenbrock', 'qp_constrained'}:
                original_x = sample_results_np * data_stat[f'x_std_all_neighbor_per_dim'][:args.training_data_dimension] + data_stat[f'x_mean_all_neighbor_per_dim'][:args.training_data_dimension]

                # Save sample results
                sample_data['x'].append(original_x)
            else:
                raise ValueError(f"Invalid task name: {args.task_name}")
        elif args.data_stat_type == 'min_max_stat':
            if args.task_name == 'robot':
                assert args.data_x_channel == 5, "Only support 5 channels for robot"
                # sample_results_np has shape (batch_size, sample_num, data_x_channel, data_x_size_h), e.g. (32, 100, 4, 40)

                # Note that t_final is at the first channel
                t_final = sample_results_np[:, :, 0, :].mean(axis=-1, keepdims=True) # (batch_size, sample_num, 1)
                robot_0_u0 = sample_results_np[:, :, 1, :] # (batch_size, sample_num, data_x_size_h)
                robot_0_u1 = sample_results_np[:, :, 2, :] # (batch_size, sample_num, data_x_size_h)
                robot_1_u0 = sample_results_np[:, :, 3, :] # (batch_size, sample_num, data_x_size_h)
                robot_1_u1 = sample_results_np[:, :, 4, :] # (batch_size, sample_num, data_x_size_h)

                # Unormalize t_final using min and max
                t_final = (t_final + 1.0) * 0.5 * (data_stat["t_final_max"] - data_stat["t_final_min"]) + data_stat["t_final_min"]
                robot_0_u0 = (robot_0_u0 + 1.0) * 0.5 * (data_stat["robot_0_u0_max"] - data_stat["robot_0_u0_min"]) + data_stat["robot_0_u0_min"]
                robot_0_u1 = (robot_0_u1 + 1.0) * 0.5 * (data_stat["robot_0_u1_max"] - data_stat["robot_0_u1_min"]) + data_stat["robot_0_u1_min"]
                robot_1_u0 = (robot_1_u0 + 1.0) * 0.5 * (data_stat["robot_1_u0_max"] - data_stat["robot_1_u0_min"]) + data_stat["robot_1_u0_min"]
                robot_1_u1 = (robot_1_u1 + 1.0) * 0.5 * (data_stat["robot_1_u1_max"] - data_stat["robot_1_u1_min"]) + data_stat["robot_1_u1_min"]

                # Save sample results
                sample_data['t_final'].append(t_final)  
                sample_data['robot_0_u0'].append(robot_0_u0)
                sample_data['robot_0_u1'].append(robot_0_u1)
                sample_data['robot_1_u0'].append(robot_1_u0)
                sample_data['robot_1_u1'].append(robot_1_u1)
            else:
                raise ValueError(f"Invalid task name: {args.task_name}")
        else:
            raise ValueError(f"Invalid data stat type: {args.data_stat_type}")

        # Save condition data
        if args.task_name in {'himmelblau', 'levy', 'rosenbrock', 'qp_constrained'}:
            sample_data['condition_lambda_value'].append(batch['condition_lambda_value'])
        elif args.task_name == 'robot':
            sample_data['condition_data'].append(batch['condition_data'])
        else:
            raise ValueError(f"Invalid task name: {args.task_name}")

        # Save condition seed string
        sample_data['condition_seed_string'].append(batch['condition_seed_string'])

    # Save sample data
    directory_to_save = f"{args.test_sample_directory}/{args.project_name}"
    if not os.path.exists(directory_to_save):
        os.makedirs(directory_to_save, exist_ok=True)
    
    file_name = f"{args.wandb_run_name}_sample_num_{args.test_sample_num}.pkl"
    with open(f"{directory_to_save}/{file_name}", "wb") as f:
        pickle.dump(sample_data, f)

    logger.info(f"Saved sample data to {directory_to_save}/{file_name}")

def parse_config():

    args = parse_args()

    return args

if __name__ == '__main__':
    main()
