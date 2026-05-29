import os
import glob
import pickle
import numpy as np
import torch
from dataset.dataset import BaseDataset

class RosenbrockDataset(BaseDataset):
    def __init__(self, 
                 logger=None,
                 split='train', 
                 args=None):
        
        # Initialize the parent class
        super().__init__()
        
        self.logger = logger
        self.split = split
        self.k_neighbor_to_use = args.k_neighbor_to_use
        self.dataset_train_ratio = args.dataset_train_ratio
        self.dataset_val_ratio = args.dataset_val_ratio
        self.dataset_downsample_condition_ratio = args.dataset_downsample_condition_ratio
        self.dataset_downsample_initial_guess_ratio = args.dataset_downsample_initial_guess_ratio
        self.condition_encoder_data_type = args.condition_encoder_data_type

        # Data processing setup
        self.parameter_data_process_type = args.parameter_data_process_type
        self.parameter_data_min = args.parameter_data_min
        self.parameter_data_max = args.parameter_data_max
        self.parameter_data_min_list = args.parameter_data_min_list
        self.parameter_data_max_list = args.parameter_data_max_list
        self.condition_fusion_type = args.condition_fusion_type
        self.radius_data_process_type = args.radius_data_process_type
        self.radius_data_scale = args.radius_data_scale
        self.radius_square_grad_data_process_type = args.radius_square_grad_data_process_type
        self.radius_square_grad_data_scale = args.radius_square_grad_data_scale

        # Solver behavior model setup
        self.solver_behavior_data_type = args.solver_behavior_data_type

        self.random_seed = args.random_seed

        self.logger.info(f"Initializing {split} dataset from {args.data_root_dir}")

        # Get all condition seed directories
        condition_dirs = glob.glob(os.path.join(args.data_root_dir, "condition_seed_*"))
        # Sort by the number after "condition_seed_"
        condition_dirs = sorted(condition_dirs, key=lambda x: int(os.path.basename(x).split('_')[-1]))
        
        # Split directories based on ratios
        num_conditions = len(condition_dirs)

        self.logger.info(f"Found {num_conditions} condition directories for {split} dataset")

        indices = np.arange(num_conditions)
        
        train_size = int(self.dataset_train_ratio * num_conditions)
        val_size = int(self.dataset_val_ratio * num_conditions)
        
        if self.split == 'train':
            used_dirs = [condition_dirs[i] for i in indices[:train_size]]
        elif self.split == 'val':
            used_dirs = [condition_dirs[i] for i in indices[train_size:train_size+val_size]]
        elif self.split == 'test' or self.split == 'test_random':
            used_dirs = [condition_dirs[i] for i in indices[train_size+val_size:]]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        self.logger.info(f"{len(used_dirs)} condition directories for {split} dataset")

        rng = np.random.default_rng(self.random_seed)

        if self.dataset_downsample_condition_ratio < 1.0:
            downsample_size = int(len(used_dirs) * self.dataset_downsample_condition_ratio)
            used_dirs = rng.choice(used_dirs, downsample_size, replace=False).tolist()
        
        self.logger.info(f"After downsampling condition ratio {self.dataset_downsample_condition_ratio}, {len(used_dirs)} condition directories for {split} dataset")

        self.pickle_files = []
        for dir_path in used_dirs:
            initial_guess_files = glob.glob(os.path.join(dir_path, "*.pkl"))

            if self.dataset_downsample_initial_guess_ratio < 1.0:
                downsample_size = int(len(initial_guess_files) * self.dataset_downsample_initial_guess_ratio)
                initial_guess_files = rng.choice(initial_guess_files, downsample_size, replace=False).tolist()

            self.pickle_files.extend(initial_guess_files)
        
        self.logger.info(f"Found total of {len(self.pickle_files)} pickle files for {split} dataset after downsampling initial guess ratio {self.dataset_downsample_initial_guess_ratio}")
        
        # Create metadata mapping without loading full data
        self.logger.info(f"Creating metadata mapping for {len(self.pickle_files)} pickle files...")
        
        estimated_size = len(self.pickle_files) * len(self.k_neighbor_to_use)
        self.metadata_mapping = [(None, None)] * estimated_size
        
        curr_pickle_file_num = 0
        actual_index = 0
        
        for pkl_file in self.pickle_files:

            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

                # Only use converged data
                if not data["convergence_flag"]:
                    continue
                    
                try:
                    all_keys = sorted(data['solver_info'].keys(), reverse=True)
                except:
                    raise ValueError(f"Invalid pickle file: {pkl_file}")

                for idx in self.k_neighbor_to_use:
                    if idx < len(all_keys):
                        iter_to_use = all_keys[idx]
                        if actual_index < estimated_size:
                            self.metadata_mapping[actual_index] = (pkl_file, iter_to_use)
                            actual_index += 1
                        else:
                            self.metadata_mapping.append((pkl_file, iter_to_use))
                            actual_index += 1
                
            if curr_pickle_file_num % 500 == 0:
                self.logger.info(f"Processing file {curr_pickle_file_num} of {len(self.pickle_files)}")
            curr_pickle_file_num += 1
        
        if actual_index < estimated_size:
            self.metadata_mapping = self.metadata_mapping[:actual_index]
        
        self.logger.info(f"Original metadata mapping size: {len(self.metadata_mapping)}")
        
        # Apply data repeat
        self.metadata_mapping = self.metadata_mapping * args.data_repeat_num
        
        # Load data statistics
        self.logger.info(f"Loading data statistics from {args.data_stat_file_path}")
        
        self.data_stat_type = args.data_stat_type

        with open(args.data_stat_file_path, "rb") as f:
            data_stat = pickle.load(f)
        
        self.data_stat = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in data_stat.items()
        }

        self.training_data_dimension = args.training_data_dimension
        self.unet_type = args.unet_type
        self.data_x_channel = args.data_x_channel
        self.data_x_size_h = args.data_x_size_h
        self.data_x_size_w = args.data_x_size_w
        
        self.logger.info(f"{split} dataset initialization complete with {len(self.metadata_mapping)} samples")
    
    def __len__(self):
        return len(self.metadata_mapping)

    def __getitem__(self, idx):
        file_path, marjor_iter_idx = self.metadata_mapping[idx]

        with open(file_path, 'rb') as f:
            sol_hist_data = pickle.load(f)

        data_x_processed = self._prepare_data_x(sol_hist_data, marjor_iter_idx)
        condition_y_processed = self._prepare_condition_y(sol_hist_data, marjor_iter_idx)

        # Combine processed data
        processed_data = {}
        processed_data.update(data_x_processed)
        processed_data.update(condition_y_processed)

        # Add original data to processed data
        processed_data["condition_lambda_value"] = torch.tensor(sol_hist_data['condition_lambda_value'], dtype=torch.float32)
        processed_data["file_path"] = file_path

        condition_seed_string = file_path.split('/')[-2].split('_')[-1]
        processed_data["condition_seed_string"] = condition_seed_string

        return processed_data

    def _prepare_data_x(self, sol_hist_data, marjor_iter_idx):
        """Load, normalize, and reshape solver state into `data_x_*` tensors."""
        
        data_x_processed = {}

        # Load raw data
        x = torch.tensor(sol_hist_data['solver_info'][marjor_iter_idx]['x'][:self.training_data_dimension], dtype=torch.float32)

        max_major_iter = sol_hist_data['max_major_iter']
        if marjor_iter_idx == max_major_iter:
            x_next_iter = torch.tensor(sol_hist_data['solver_info'][marjor_iter_idx]['x'][:self.training_data_dimension], dtype=torch.float32)
        else:
            x_next_iter = torch.tensor(sol_hist_data['solver_info'][marjor_iter_idx + 1]['x'][:self.training_data_dimension], dtype=torch.float32)
        
        # Load final iter
        x_final_iter = torch.tensor(sol_hist_data['solver_info'][max_major_iter]['x'][:self.training_data_dimension], dtype=torch.float32)
        
        # Normalize data x based on data_stat_type
        if self.data_stat_type == 'all_neighbor_per_dim_stat':
            x = (x - self.data_stat[f'x_mean_all_neighbor_per_dim'][:self.training_data_dimension]) / self.data_stat[f'x_std_all_neighbor_per_dim'][:self.training_data_dimension]
            x_next_iter = (x_next_iter - self.data_stat[f'x_mean_all_neighbor_per_dim'][:self.training_data_dimension]) / self.data_stat[f'x_std_all_neighbor_per_dim'][:self.training_data_dimension]
            x_final_iter = (x_final_iter - self.data_stat[f'x_mean_all_neighbor_per_dim'][:self.training_data_dimension]) / self.data_stat[f'x_std_all_neighbor_per_dim'][:self.training_data_dimension]
        else:
            raise ValueError(f"Invalid data stat type: {self.data_stat_type}")

        # Reshape for UNet input.
        if self.unet_type == '1D':
            data_x_concat = x.unsqueeze(-1).reshape(self.data_x_channel, self.data_x_size_h)
            data_x_next_iter_concat = x_next_iter.unsqueeze(-1).reshape(self.data_x_channel, self.data_x_size_h)
            data_x_final_iter_concat = x_final_iter.unsqueeze(-1).reshape(self.data_x_channel, self.data_x_size_h)

        else:
            raise ValueError(f"Invalid unet type: {self.unet_type}")

        data_x_processed["data_x_concat"] = data_x_concat
        data_x_processed["data_x_next_iter_concat"] = data_x_next_iter_concat
        data_x_processed["data_x_final_iter_concat"] = data_x_final_iter_concat
        
        return data_x_processed

    def _prepare_condition_y(self, sol_hist_data, marjor_iter_idx):
        """Build condition tensors based on configured feature types."""
        condition_y_processed = {}
        
        # Parameters
        if 'parameter' in self.condition_encoder_data_type or 'parameter' in self.solver_behavior_data_type:
            # Load parameter data
            condition_y_parameter_concat = torch.tensor(sol_hist_data['condition_lambda_value'], dtype=torch.float32)

            # Process parameter data
            if self.parameter_data_process_type == 'original':
                pass
            elif self.parameter_data_process_type == 'min_max':
                assert self.parameter_data_min_list is not None and self.parameter_data_max_list is not None, "parameter_data_min_list and parameter_data_max_list must be set"

                parameter_data_min_list = torch.tensor(self.parameter_data_min_list, dtype=torch.float32)
                parameter_data_max_list = torch.tensor(self.parameter_data_max_list, dtype=torch.float32)
                condition_y_parameter_concat = (condition_y_parameter_concat - parameter_data_min_list) / (parameter_data_max_list - parameter_data_min_list)
            else:
                raise ValueError(f"Invalid parameter data process type: {self.parameter_data_process_type}")
            
            condition_y_processed['condition_y_parameter_concat'] = condition_y_parameter_concat

        # Relative iterate index (used as a generic neighbor indicator).
        max_major_iter = torch.tensor(sol_hist_data['max_major_iter'], dtype=torch.float32)
        curr_major_iter = torch.tensor(marjor_iter_idx, dtype=torch.float32)
        k_neighbor_idx_flatten = (max_major_iter - curr_major_iter).flatten()
        condition_y_processed['k_neighbor_idx_flatten'] = k_neighbor_idx_flatten
        
        # Radius-based features
        if (
            'radius' in self.condition_encoder_data_type
            or 'radius' in self.solver_behavior_data_type
            or 'radius_square_grad' in self.solver_behavior_data_type
        ):
            x = torch.tensor(sol_hist_data['solver_info'][marjor_iter_idx]['x'][:self.training_data_dimension], dtype=torch.float32)
            max_major_iter = sol_hist_data['max_major_iter']
            x_final_iter = torch.tensor(sol_hist_data['solver_info'][max_major_iter]['x'][:self.training_data_dimension], dtype=torch.float32)
            if 'radius' in self.condition_encoder_data_type or 'radius' in self.solver_behavior_data_type:
                radius = torch.norm(x - x_final_iter, dim=-1)

                # Process radius data
                if self.radius_data_process_type == 'original':
                    pass
                elif self.radius_data_process_type == 'scale':
                    radius = radius * self.radius_data_scale
                else:
                    raise ValueError(f"Invalid radius data process type: {self.radius_data_process_type}")

                radius = radius.unsqueeze(0)
                condition_y_processed['condition_y_radius_concat'] = radius
            if 'radius_square_grad' in self.solver_behavior_data_type:

                radius_square_grad = x - x_final_iter

                if self.radius_square_grad_data_process_type == 'original':
                    pass
                elif self.radius_square_grad_data_process_type == 'scale':
                    radius_square_grad = radius_square_grad * self.radius_square_grad_data_scale
                else:
                    raise ValueError(f"Invalid radius square grad data process type: {self.radius_square_grad_data_process_type}")

                radius_square_grad = radius_square_grad.unsqueeze(0)
                condition_y_processed['condition_y_radius_square_grad_concat'] = radius_square_grad
        
        return condition_y_processed