from run.train.diffusion.trainer_diffusion import Trainer
from run.train.train_utils import set_random_seed, create_logger
from dataset import build_dataloader

from accelerate import Accelerator, DataLoaderConfiguration

import datetime
import os

import logging

from run.diffusion_parse_args_utils import parse_args

def main():
    args = parse_config()

    model_params = {}
    for key, val in vars(args).items():
        model_params[key] = val

    # Set up random seed for model initialization
    set_random_seed(args.random_seed)

    # Configure result_folder with time
    if args.previous_run_current_time is not None:
        current_datetime = args.enforce_current_time
    else:
        current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    result_folder = f"{args.result_folder}/{args.project_name}/{args.wandb_run_name}/{current_datetime}"

    os.makedirs(result_folder, exist_ok=True)

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

    log_file = f"{output_dir}/log_train_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"

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
    if accelerator.is_main_process:
        logger.info('Building training data loader...')
    
    train_loader = build_dataloader(
        logger=logger,
        split="train",
        args=args,
    )

    if accelerator.is_main_process:
        logger.info('Building validation data loader...')
    
    validation_loader = build_dataloader(
        logger=logger,
        split="val",
        args=args,
    )

    # Build model ########################################################################################################
    if accelerator.is_main_process:
        logger.info('Building model...')

    if args.main_model_type != 'diffusion':
        raise ValueError(f"main_model_type must be 'diffusion', got: {args.main_model_type}")
    from model.trajectory_diffusion_model import TrajectoryDiffusion
    model = TrajectoryDiffusion(model_params=model_params)

    total_data_len = len(train_loader)

    if args.training_steps_limit is not None:
        train_num_steps = args.training_steps_limit
    else:
        train_num_steps = total_data_len * args.epochs

    # Build trainer ####################################################################################################
    trainer = Trainer(
        model=model,
        train_data_loader=train_loader,
        validation_data_loader=validation_loader,
        train_lr=args.train_lr,
        train_num_steps=train_num_steps,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        results_folder=result_folder,
        project_name=args.project_name,
        model_params=model_params,
        max_grad_norm=args.max_grad_norm,
        curr_datetime=current_datetime,
        accelerator=accelerator,
        checkpoint_path_to_start=args.checkpoint_path_to_start,
        use_lr_scheduler=args.use_lr_scheduler,
        wandb_run_name=args.wandb_run_name,
        previous_run_id=args.previous_run_id,
        warmup_steps=args.warmup_steps,
        wandb_api_key=args.wandb_api_key,
    )

    trainer.train()

def parse_config():

    args = parse_args()

    return args


if __name__ == '__main__':
    main()
