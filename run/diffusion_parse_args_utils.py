import argparse
import ast

def parse_tuple(s):
    try:
        value = ast.literal_eval(s)
    except (SyntaxError, ValueError):
        raise argparse.ArgumentTypeError("Tuple argument must be a valid Python tuple")
    if not isinstance(value, tuple):
        raise argparse.ArgumentTypeError("Tuple argument must be a valid Python tuple")
    return value


def parse_string_list(s):
    """Parse a comma-separated string into a list of strings. Always returns a list, even for single values."""
    if isinstance(s, list):
        return s
    # Split by comma and strip whitespace, ensuring we always get a list
    items = [item.strip() for item in s.split(',')]
    # Remove empty strings that might result from trailing commas
    items = [item for item in items if item]
    return items


def parse_float_list(s):
    """Parse a comma-separated string into a list of floats.

    Examples:
      "1,3," -> [1.0, 3.0]
      " 0.1 , -2 " -> [0.1, -2.0]
    """
    if s is None:
        return None
    if isinstance(s, list):
        return [float(x) for x in s]
    if isinstance(s, tuple):
        return [float(x) for x in s]
    items = [item.strip() for item in str(s).split(',')]
    items = [item for item in items if item]  # allow trailing commas
    return [float(item) for item in items]


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')

    # General setup #################################################################
    parser.add_argument('--workers',
                        type=int,
                        default=4,
                        help='Number of dataloader worker processes.')
    parser.add_argument('--result_folder',
                        type=str,
                        default='results',
                        help='Root directory for results.')
    parser.add_argument('--data_root_dir',
                        type=str,
                        default=None,
                        help='Dataset root directory (task-specific).')
    parser.add_argument('--task_name',
                        type=str,
                        default='robot',
                        choices=['himmelblau', 'robot', 'rosenbrock', 'levy', 'qp_constrained'],
                        help='Task/dataset name.')
    parser.add_argument('--project_name',
                        type=str,
                        default="solver_info_diffusion",
                        help='Project name (used for organizing outputs and W&B project).')
    parser.add_argument('--wandb_mode',
                        type=str,
                        default='offline',
                        choices=['offline', 'online'],
                        help='W&B mode.')
    parser.add_argument('--wandb_run_name',
                        type=str,
                        default='run',
                        help='W&B run name.')
    parser.add_argument(
        '--wandb_api_key',
        type=str,
        default=None,
        help='W&B API key (optional). If provided, used for wandb.login().'
    )
    
    # Warmstart from previous checkpoint #########################################
    parser.add_argument('--previous_run_current_time', 
                        type=str,
                        default=None,
                        help='previous run current time, to warmstart from previous checkpoint')
    parser.add_argument('--previous_run_id',
                        type=str,
                        default=None,
                        help='previous run id, to warmstart from previous checkpoint')
    parser.add_argument('--checkpoint_path_to_start',
                        default=None,
                        type=str,
                        help='checkpoint path')

    # Training setup #################################################################
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=64,
                        help='batch size for train dataloader')
    parser.add_argument('--validation_batch_size',
                        type=int,
                        default=1024,
                        help='batch size for validation dataloader')
    parser.add_argument('--epochs',
                        type=int,
                        default=2,
                        help='number of epochs')
    parser.add_argument('--training_steps_limit',
                        type=int,
                        default=None,
                        help='Max training steps (overrides epochs if set).')
    parser.add_argument('--random_seed',
                        type=int,
                        default=0,
                        help='random seed to initialize model')
    parser.add_argument('--dataset_train_ratio',
                        type=float,
                        default=0.8,
                        help='train ratio of dataset')
    parser.add_argument('--dataset_val_ratio',
                        type=float,
                        default=0.1,
                        help='val ratio of dataset')
    parser.add_argument('--dataset_downsample_condition_ratio',
                        type=float,
                        default=1.,
                        help='downsample ratio of the condition during training')
    parser.add_argument('--dataset_downsample_initial_guess_ratio',
                        type=float,
                        default=1.,
                        help='downsample ratio of the initial guess during training')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=1.,
                        help='gradient norm clipping')
    parser.add_argument('--use_lr_scheduler',
                        type=str,
                        default="True",
                        choices=["True", "False"],
                        help="whether to use learning rate scheduler")
    parser.add_argument('--train_lr',
                        type=float,
                        default=8e-5,
                        help='learning rate for training')
    parser.add_argument('--warmup_steps',
                        type=int,
                        default=0,
                        help='linear LR warmup steps; 0 disables warmup')
    parser.add_argument('--sampling_timesteps',
                        type=int,
                        default=50,
                        help='Number of sampling timesteps during evaluation.')
    parser.add_argument('--training_timesteps',
                        type=int,
                        default=50,
                        help='Number of diffusion timesteps used during training.')
    parser.add_argument('--data_repeat_num',
                        type=int,
                        default=1,
                        help='Number of times to repeat the dataset in an epoch.')
    parser.add_argument('--cond_drop_prob',
                        type=float,
                        default=0.1,
                        help='probability of dropping condition during training')
    parser.add_argument('--auto_normalize',
                        type=lambda x: x.lower() == 'true',
                        default=False,
                        help='Auto-normalize inputs from [0,1] to [-1,1].')
    parser.add_argument('--filter_data_by_objective',
                        type=lambda x: x.lower() == 'true',
                        default=False,
                        help='whether to filter data by objective')
    parser.add_argument('--filter_data_by_objective_threshold',
                        type=float,
                        default=12.0,
                        help='threshold to filter data by objective')
    parser.add_argument('--filter_neighborhood_by_threshold',
                        type=lambda x: x.lower() == 'true',
                        default=False,
                        help='whether to filter neighborhood by threshold')
    parser.add_argument('--filter_neighborhood_by_threshold_threshold',
                        type=float,
                        default=0.5,
                        help='threshold to filter neighborhood by threshold')
    parser.add_argument('--parameter_data_process_type',
                        type=str, 
                        default='original',
                        choices=['original', 'min_max', 'standardize'],
                        help='type of parameter data processing to use, e.g., min-max normalization for better convergence representation')
    parser.add_argument('--parameter_data_min',
                        type=float,
                        default=0.,
                        help='minimum value for parameter data processing')
    parser.add_argument('--parameter_data_max',
                        type=float,
                        default=1.,
                        help='maximum value for parameter data processing')
    parser.add_argument('--parameter_data_min_list',
                        type=parse_float_list,
                        default=[50, -1.5],
                        help='comma-separated per-dimension min list for parameter data processing, e.g. "1,3,"')
    parser.add_argument('--parameter_data_max_list',
                        type=parse_float_list,
                        default=[200, 0.0],
                        help='comma-separated per-dimension max list for parameter data processing, e.g. "5,7,"')
    parser.add_argument('--radius_data_process_type',
                        type=str,
                        default='scale',
                        choices=['original', 'scale', 'log_transform'],
                        help='type of radius data processing to use, e.g., min-max normalization for better convergence representation')
    parser.add_argument('--radius_data_scale',
                        type=float,
                        default=10.,
                        help='scale value for radius data processing')
    parser.add_argument('--radius_square_grad_data_process_type',
                        type=str,
                        default='original',
                        choices=['original', 'scale', 'mean_std_normalize'],
                        help='type of radius square grad data processing to use, e.g., min-max normalization for better convergence representation')
    parser.add_argument('--radius_square_grad_data_scale',
                        type=float,
                        default=10.,
                        help='scale value for radius square grad data processing')
    parser.add_argument('--condition_fusion_type',
                        type=str,
                        default='raw_concat',
                        choices=['raw_concat'],
                        help='type of condition fusion to use, e.g., concat or add for better convergence representation')
    parser.add_argument('--data_x_process_type',
                        type=str,
                        default='add_t_into_channel',
                        choices=['add_t_into_channel', 'add_t_into_timestep'],
                        help='type of data x process to use, e.g., add t into channel or add t into timestep')
    # Model setup ################################################################
    # Schedule setup ############################################################
    parser.add_argument('--beta_schedule',
                        type=str,
                        default="cosine",
                        choices=["sigmoid", "cosine", "linear", "custom"],
                        help='type of schedule to use')
    parser.add_argument('--beta_schedule_type',
                        type=str,
                        default="0_to_1",
                        choices=["0_to_1"],
                        help='type of schedule to use')
    # Unet setup ##########################
    parser.add_argument('--unet_type',
                        type=str,
                        default="1D",
                        choices=["1D"],
                        help='type of unet model to use')
    parser.add_argument('--unet_model_dim',
                        type=int,
                        default=128,
                        help='dimensions for unet model')
    parser.add_argument('--unet_dim_mults',
                        type=parse_tuple,
                        default=(1, 2, 4),
                        help='Dimension multipliers for the U-Net.'
                        )
    parser.add_argument('--unet_attention_resolutions',
                        type=parse_tuple,
                        default=(1, 2, 4),
                        help='dimensions with attention layer in unet')
    
    # Condition encoder setup #########################################################
    parser.add_argument('--condition_encoder_data_type',
                        type=parse_string_list,
                        default=['parameter'],
                        help='what condition data to use, options: parameter, convergence, residual, objective, constraint_violation')
    parser.add_argument('--condition_y_convergence_encode_type',
                        type=str,
                        default='MLP',
                        choices=['MLP'],
                        help='type of condition encoder to use for convergence parameter (k-neighbor)')
    parser.add_argument('--condition_y_residual_encode_type',
                        type=str,
                        default='MLP',
                        choices=['MLP'],
                        help='type of condition encoder to use for residual parameter')
    parser.add_argument('--condition_encoder_type',
                        type=str,
                        default='MLP',
                        choices=['MLP'],
                        help='type of condition encoder to use')
    parser.add_argument('--data_x_channel',
                        type=int,
                        default=20,
                        help='channels for data x')
    parser.add_argument('--data_x_size_h',
                        type=int,
                        default=1,
                        help='size h of data x within (h, w)')
    parser.add_argument('--data_x_size_w',
                        type=int,
                        default=1,
                        help='size w of data x within (h, w)')
    parser.add_argument('--condition_y_parameter_dim',
                        type=int,
                        default=20,
                        help='dimensions for problem parameter in condition input y')
    parser.add_argument('--condition_y_convergence_dim',
                        type=int,
                        default=3,
                        help='dimensions for convergence information in condition input y')
    parser.add_argument('--condition_y_radius_dim',
                        type=int,
                        default=1,
                        help='dimensions for radius information in condition input y')
    parser.add_argument('--embed_x_dim',
                        type=int,
                        default=8,
                        help='Embed x dim before UNet')
    parser.add_argument('--embed_y_all_dim',
                        type=int,
                        default=80,
                        help='Embed all condition y dim (parameter + convergence) before UNet')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.,
                        help='dropout rate for unet and condition encoder')
    parser.add_argument('--batch_norm',
                        type=lambda x: x.lower() == 'true',
                        default=False,
                        help='whether to use batch normalization for condition encoder')
    parser.add_argument('--mlp_type',
                        type=str,
                        default='silu_layer_no_output_norm',
                        choices=['silu_layer_no_output_norm'],
                        help='type of MLP to use for condition encoder')
    
    # Training data setup, data statistics ############################################################
    parser.add_argument('--training_data_dimension',
                        type=int,
                        default=2,
                        help='in qp problem, the dimension of training data chosen to train, to decrease the problem complexity')
    parser.add_argument('--training_data_type',
                        type=str,
                        default='k_neighbor_single_point',
                        choices=['k_neighbor_single_point'],
                        help='type of training data to use, whether to learn on the single point of k-neighbor')
    parser.add_argument('--k_neighbor_to_use',
                        type=lambda x: [int(i) for i in x.split(',')],
                        default=[0, 1, 2, 3, 4],
                        help='k neighbor to use in training data')
    parser.add_argument('--max_k_neighbor_to_encode',
                        type=int,
                        default=10,
                        help='max k neighbor to encode in condition encoder')
    parser.add_argument('--data_stat_file_path',
                        type=str,
                        required=True,
                        help='Path to a precomputed data statistics pickle (task-specific).')
    parser.add_argument('--data_stat_type',
                        type=str,
                        default='per_neighbor_per_dim_stat',
                        choices=['all_neighbor_global_stat','all_neighbor_per_dim_stat', 'per_neighbor_per_dim_stat', 'per_neighbor_global_stat', 'min_max_stat'])
    parser.add_argument('--std_padding',
                        type=float,
                        default=0.0,
                        help='add a small value to std, avoid nan loss')
    parser.add_argument('--gamma',
                        type=str,
                        default="None",
                        help='gamma for data filtering')
    
    # Diffusion model setup ############################################################
    parser.add_argument('--main_model_type',
                        type=str,
                        default='diffusion',
                        choices=['diffusion'],
                        help='type of main model to generate trajectory')
    
    # Test setup ##########################################################################
    parser.add_argument('--checkpoint_results_folder',
                        type=str,
                        default=None,
                        help='Path to a results folder that contains a `checkpoint/` subdirectory (optional).')
    parser.add_argument('--checkpoint_name',
                        type=str,
                        default='model-best_validation_epoch-635.pt',
                        help='name of checkpoint to use')
    parser.add_argument('--test_sample_num',
                        type=int,
                        default=1000,
                        help='number of samples generated for testing for each condition input')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=1,
                        help='batch size for test dataloader')
    parser.add_argument('--to_plot_k_neighbor_accuracy',
                        type=lambda x: x.lower() == 'true',
                        default=False,
                        help='whether to plot results with k-neighbor accuracy')
    parser.add_argument('--to_plot_multiple_conditions_with_gt',
                        type=lambda x: x.lower() == 'true',
                        default=False,
                        help='whether to plot results with multiple conditions and gt')
    parser.add_argument('--test_data_type',
                        type=str,
                        default='test',
                        choices=['test', 'train', 'test_random'],
                        help='type of test data to use')
    parser.add_argument('--test_condition_seed_list',
                        type=lambda x: [int(i) for i in x.split(',')],
                        default=[900,901,902,903,904],
                        help='list of condition seed to test')
    parser.add_argument('--test_sample_directory',
                        type=str,
                        help='directory for test samples')
    parser.add_argument('--test_method_name',
                        type=str,
                        default="unconstrained_qp_test",
                        help='name of test method')
    parser.add_argument('--cond_scale',
                        type=float,
                        default=6.,
                        help='cond scale for classifier-free guidance')
    parser.add_argument('--rescaled_phi',
                        type=float,
                        default=0.7,
                        help='rescaled phi for interpolated rescaled logits')

    # Solver behavior model training setup ############################################################
    parser.add_argument('--solver_behavior_data_type',
                        type=parse_string_list,
                        default=['parameter', 'radius_square_grad'],
                        help='what data to use for the solver behavior model. Parameter is always used. Then radius_square_grad can be used.')
    parser.add_argument('--solver_behavior_encoder_data_type',
                        type=parse_string_list,
                        default=['parameter'],
                        help='what data to use for the encoder of the solver behavior model.')
    parser.add_argument('--solver_behavior_condition_y_parameter_dim',
                        type=int,
                        default=4,
                        help='dimensions for problem parameter in condition input y for the solver behavior model')
    parser.add_argument('--solver_behavior_embed_y_all_dim',
                        type=int,
                        default=64,
                        help='dimensions for the encoded condition y for the solver behavior model')
    parser.add_argument('--solver_behavior_output_data_type',
                        type=str,
                        default='radius_square_grad',
                        choices=['radius_square_grad'],
                        help='what data to use for the output of the solver behavior model.')
    parser.add_argument('--solver_behavior_unet_type',
                        type=str,
                        default='1D',
                        choices=['1D'],
                        help='type of unet model to use for the solver behavior model')
    parser.add_argument('--solver_behavior_unet_model_dim',
                        type=int,
                        default=64,
                        help='dimensions for unet model for the solver behavior model')
    parser.add_argument('--solver_behavior_unet_dim_mults',
                        type=parse_tuple,
                        default=(1, 2, 4),
                        help='dimensions mult for unet model for the solver behavior model')
    parser.add_argument('--solver_behavior_dropout',
                        type=float,
                        default=0.,
                        help='dropout rate for unet model for the solver behavior model')
    parser.add_argument('--solver_behavior_loss_type',
                        type=str,
                        default='mse',
                        choices=['mse', 'huber'],
                        help='type of loss function (mse or huber/SmoothL1) for the solver behavior model')
    parser.add_argument('--solver_behavior_loss_data_type',
                        type=str,
                        default='original',
                        choices=['original'],
                        help='how to aggregate loss: original (element-wise mean) or average_t_in_channel_then_sum')

    # Solver behavior model testing setup ############################################################
    parser.add_argument('--solver_behavior_checkpoint_path',
                        type=str,
                        default=None,
                        help='checkpoint path for the solver behavior model')
    parser.add_argument('--solver_behavior_apply_on_max_diffusion_steps',
                        type=int,
                        default=0,
                        help='the maximum diffusion steps to apply the solver behavior model')
    parser.add_argument('--solver_behavior_guidance_step_per_diffusion_step',
                        type=int,
                        default=5,
                        help='the number of solver behavior steps to apply the solver behavior model as guidance per diffusion step')
    parser.add_argument('--solver_behavior_step_size',
                        type=float,
                        default=0.1,
                        help='the step size for the solver behavior model')
    args = parser.parse_args()

    return args
