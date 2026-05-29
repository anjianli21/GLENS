import os
import glob
import pickle
import numpy as np
import torch
from dataset.dataset import BaseDataset
from dataset.robot.legacy_car_keys import canonical_stat_dict, get_solver_x_value

class RobotDataset(BaseDataset):
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
        self.condition_fusion_type = args.condition_fusion_type
        self.radius_data_process_type = args.radius_data_process_type
        self.radius_data_scale = args.radius_data_scale
        self.radius_square_grad_data_process_type = args.radius_square_grad_data_process_type
        self.radius_square_grad_data_scale = args.radius_square_grad_data_scale
        self.data_x_process_type = args.data_x_process_type

        # Filter setup
        self.filter_data_by_objective = args.filter_data_by_objective
        self.filter_data_by_objective_threshold = args.filter_data_by_objective_threshold
        self.filter_neighborhood_by_threshold = args.filter_neighborhood_by_threshold
        self.filter_neighborhood_by_threshold_threshold = args.filter_neighborhood_by_threshold_threshold

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

        # Keep condition ordering stable to avoid overlap across splits.
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

        # Initialize a random generator with the specified seed
        rng = np.random.default_rng(self.random_seed)

        if self.dataset_downsample_condition_ratio < 1.0:
            # Randomly downsample the used_dirs
            downsample_size = int(len(used_dirs) * self.dataset_downsample_condition_ratio)
            used_dirs = rng.choice(used_dirs, downsample_size, replace=False).tolist()
        
        self.logger.info(f"After downsampling condition ratio {self.dataset_downsample_condition_ratio}, {len(used_dirs)} condition directories for {split} dataset")

        # Get all pickle files from selected directories
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

                if (self.split == 'train' or self.split == 'val') and not data['convergence_flag']:
                    continue
                
                try:
                    all_keys = sorted(data['solver_info'].keys(), reverse=True)
                except:
                    raise ValueError(f"Invalid pickle file: {pkl_file}")
                
                if (self.split == 'train' or self.split == 'val') and self.filter_data_by_objective:
                    max_major_iter = data['max_major_iter']
                    obj_val = data['solver_info'][max_major_iter]['x']['t_final']
                    if obj_val > self.filter_data_by_objective_threshold:
                        continue

                # Get available neighbors based on k_neighbor_to_use list
                # Filter neighborhood by threshold
                if (self.split == 'train' or self.split == 'val') and self.filter_neighborhood_by_threshold:
                    num_neighbor_to_select = 0                    
                    for idx in all_keys:

                        assert len(self.k_neighbor_to_use) == 1 or len(self.k_neighbor_to_use) == 10, "k_neighbor_to_use should be 1 or 10 for the current filtering setup"
                            
                        # idx is the key (iterate index) when iterating over all_keys
                        iter_to_use = idx

                        if self.filter_data_by_objective:
                            if not data[f'filter_neighborhood_by_threshold_{self.filter_neighborhood_by_threshold_threshold}_obj_filter_12'][iter_to_use]:
                                continue
                        else:
                            if not data[f'filter_neighborhood_by_threshold_{self.filter_neighborhood_by_threshold_threshold}_obj_no_filter'][iter_to_use]:
                                continue

                        if actual_index < estimated_size:                                
                            self.metadata_mapping[actual_index] = (pkl_file, iter_to_use)
                            actual_index += 1
                        else:
                            # Fallback in case our estimate was too small
                            self.metadata_mapping.append((pkl_file, iter_to_use))
                            actual_index += 1
                        
                        num_neighbor_to_select += 1
                        if num_neighbor_to_select == len(self.k_neighbor_to_use):
                            break
                else:
                    for idx in self.k_neighbor_to_use:
                        if idx < len(all_keys):

                            iter_to_use = all_keys[idx]

                            if actual_index < estimated_size:                                
                                self.metadata_mapping[actual_index] = (pkl_file, iter_to_use)
                                actual_index += 1
                            else:
                                # Fallback in case our estimate was too small
                                self.metadata_mapping.append((pkl_file, iter_to_use))
                                actual_index += 1
            
            if curr_pickle_file_num % 500 == 0:
                self.logger.info(f"Processing file {curr_pickle_file_num} of {len(self.pickle_files)}")
            curr_pickle_file_num += 1
        
        # Trim the list to the actual size used if needed
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
        data_stat = canonical_stat_dict(data_stat)

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

        # Prepare data_x (with normalization and reshaping)
        data_x_processed = self._prepare_data_x(sol_hist_data, marjor_iter_idx)
        # Prepare condition_y (with flattening and concatenation)
        condition_y_processed = self._prepare_condition_y(
            sol_hist_data,
            marjor_iter_idx,
            data_x_processed=data_x_processed,
        )

        # Combine processed data
        processed_data = {}
        processed_data.update(data_x_processed)
        processed_data.update(condition_y_processed)

        # Only attach raw condition_data during testing (avoid extra CPU work in training/val).
        if self.split in {'test', 'test_random'}:
            condition_data = sol_hist_data['condition_data']
            processed_data["condition_data"] = {
                'obs_pos': torch.tensor(condition_data['obs_pos'], dtype=torch.float32),
                'obs_radius': torch.tensor(condition_data['obs_radius'], dtype=torch.float32),
            }
        processed_data["file_path"] = file_path

        condition_seed_string = file_path.split('/')[-2].split('_')[-1]
        processed_data["condition_seed_string"] = condition_seed_string

        return processed_data

    def _prepare_data_x(self, sol_hist_data, marjor_iter_idx):
        """Load, normalize, and reshape solver state into `data_x_*` tensors."""
        
        data_x_processed = {}

        # Load raw data (current iter); pickles may use legacy `car_*` keys
        x_curr = sol_hist_data["solver_info"][marjor_iter_idx]["x"]
        robot_0_u0 = torch.tensor(get_solver_x_value(x_curr, "robot_0_u0"), dtype=torch.float32)
        robot_0_u1 = torch.tensor(get_solver_x_value(x_curr, "robot_0_u1"), dtype=torch.float32)
        robot_1_u0 = torch.tensor(get_solver_x_value(x_curr, "robot_1_u0"), dtype=torch.float32)
        robot_1_u1 = torch.tensor(get_solver_x_value(x_curr, "robot_1_u1"), dtype=torch.float32)
        t_final = torch.tensor(get_solver_x_value(x_curr, "t_final"), dtype=torch.float32)

        # Load raw data (final iter) for reuse by radius / solver behavior condition features
        max_major_iter = sol_hist_data["max_major_iter"]
        x_final = sol_hist_data["solver_info"][max_major_iter]["x"]
        robot_0_u0_final = torch.tensor(get_solver_x_value(x_final, "robot_0_u0"), dtype=torch.float32)
        robot_0_u1_final = torch.tensor(get_solver_x_value(x_final, "robot_0_u1"), dtype=torch.float32)
        robot_1_u0_final = torch.tensor(get_solver_x_value(x_final, "robot_1_u0"), dtype=torch.float32)
        robot_1_u1_final = torch.tensor(get_solver_x_value(x_final, "robot_1_u1"), dtype=torch.float32)
        t_final_final = torch.tensor(get_solver_x_value(x_final, "t_final"), dtype=torch.float32)

        if self.data_stat_type == 'min_max_stat':
            # normalize to [-1, 1]
            robot_0_u0 = ((robot_0_u0 - self.data_stat["robot_0_u0_min"]) / (self.data_stat["robot_0_u0_max"] - self.data_stat["robot_0_u0_min"])) * 2.0 - 1.0
            robot_0_u1 = ((robot_0_u1 - self.data_stat["robot_0_u1_min"]) / (self.data_stat["robot_0_u1_max"] - self.data_stat["robot_0_u1_min"])) * 2.0 - 1.0
            robot_1_u0 = ((robot_1_u0 - self.data_stat["robot_1_u0_min"]) / (self.data_stat["robot_1_u0_max"] - self.data_stat["robot_1_u0_min"])) * 2.0 - 1.0
            robot_1_u1 = ((robot_1_u1 - self.data_stat["robot_1_u1_min"]) / (self.data_stat["robot_1_u1_max"] - self.data_stat["robot_1_u1_min"])) * 2.0 - 1.0
            t_final = ((t_final - self.data_stat["t_final_min"]) / (self.data_stat["t_final_max"] - self.data_stat["t_final_min"])) * 2.0 - 1.0

            robot_0_u0_final = ((robot_0_u0_final - self.data_stat["robot_0_u0_min"]) / (self.data_stat["robot_0_u0_max"] - self.data_stat["robot_0_u0_min"])) * 2.0 - 1.0
            robot_0_u1_final = ((robot_0_u1_final - self.data_stat["robot_0_u1_min"]) / (self.data_stat["robot_0_u1_max"] - self.data_stat["robot_0_u1_min"])) * 2.0 - 1.0
            robot_1_u0_final = ((robot_1_u0_final - self.data_stat["robot_1_u0_min"]) / (self.data_stat["robot_1_u0_max"] - self.data_stat["robot_1_u0_min"])) * 2.0 - 1.0
            robot_1_u1_final = ((robot_1_u1_final - self.data_stat["robot_1_u1_min"]) / (self.data_stat["robot_1_u1_max"] - self.data_stat["robot_1_u1_min"])) * 2.0 - 1.0
            t_final_final = ((t_final_final - self.data_stat["t_final_min"]) / (self.data_stat["t_final_max"] - self.data_stat["t_final_min"])) * 2.0 - 1.0

        else:
            raise ValueError(f"Invalid data stat type: {self.data_stat_type}")

        # Reshape for UNet input.
        if self.unet_type == '1D':
            if self.data_x_process_type == 'add_t_into_channel':
                timestep = robot_0_u0.shape[0]  # 20
                t_final_repeated = t_final.squeeze().expand(timestep)
                data_x_concat = torch.stack([t_final_repeated, robot_0_u0, robot_0_u1, robot_1_u0, robot_1_u1], dim=0)

                # Final iter version (same shape (5, 20)), used for radius-based condition features.
                t_final_final_repeated = t_final_final.squeeze().expand(timestep)
                data_x_final_concat = torch.stack([t_final_final_repeated, robot_0_u0_final, robot_0_u1_final, robot_1_u0_final, robot_1_u1_final], dim=0)

        else:
            raise ValueError(f"Invalid unet type: {self.unet_type}")

        data_x_processed["data_x_concat"] = data_x_concat
        data_x_processed["data_x_final_concat"] = data_x_final_concat
        
        return data_x_processed

    def _prepare_condition_y(self, sol_hist_data, marjor_iter_idx, data_x_processed=None):
        """Build condition tensors based on configured feature types."""
        condition_y_processed = {}
        
        # Parameters
        if 'parameter' in self.condition_encoder_data_type or 'parameter' in self.solver_behavior_data_type:
            condition_data = sol_hist_data['condition_data']
            obs_pos = torch.tensor(condition_data['obs_pos'], dtype=torch.float32)
            obs_radius = torch.tensor(condition_data['obs_radius'], dtype=torch.float32)

            # Process parameter data
            if self.parameter_data_process_type == 'original':
                pass
            elif self.parameter_data_process_type == 'min_max':
                # normalize to [0, 1]
                obs_pos = (obs_pos - self.data_stat["obs_pos_min"]) / (self.data_stat["obs_pos_max"] - self.data_stat["obs_pos_min"])
                obs_radius = (obs_radius - self.data_stat["obs_radius_min"]) / (self.data_stat["obs_radius_max"] - self.data_stat["obs_radius_min"])
            else:
                raise ValueError(f"Invalid parameter data process type: {self.parameter_data_process_type}")
            
            # Flatten all tensors before concatenation
            obs_pos_flat = obs_pos.flatten()
            obs_radius_flat = obs_radius.flatten()
            
            condition_y_parameter_concat = torch.cat([obs_pos_flat, obs_radius_flat], dim=0)
            
            condition_y_processed['condition_y_parameter_concat'] = condition_y_parameter_concat

        # Relative iterate index (used as a generic neighbor indicator).
        max_major_iter = torch.tensor(sol_hist_data['max_major_iter'], dtype=torch.float32)
        curr_major_iter = torch.tensor(marjor_iter_idx, dtype=torch.float32)
        k_neighbor_idx_flatten = (max_major_iter - curr_major_iter).flatten()
        condition_y_processed['k_neighbor_idx_flatten'] = k_neighbor_idx_flatten

        # Radius-based features.
        if (
            'radius' in self.condition_encoder_data_type
            or 'radius' in self.solver_behavior_data_type
            or 'radius_square_grad' in self.solver_behavior_data_type
        ):
            if data_x_processed is None:
                raise ValueError("data_x_processed must be provided to compute radius-based condition features.")

            data_x_curr = data_x_processed["data_x_concat"]
            data_x_final = data_x_processed["data_x_final_concat"]
            
            diff = data_x_curr - data_x_final
            
            if 'radius' in self.condition_encoder_data_type or 'radius' in self.solver_behavior_data_type:
                radius = torch.norm(diff)

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
                radius_square_grad = diff

                if self.radius_square_grad_data_process_type == 'original':
                    pass
                elif self.radius_square_grad_data_process_type == 'scale':
                    radius_square_grad = radius_square_grad * self.radius_square_grad_data_scale
                else:
                    raise ValueError(f"Invalid radius square grad data process type: {self.radius_square_grad_data_process_type}")

                condition_y_processed['condition_y_radius_square_grad_concat'] = radius_square_grad
        
        return condition_y_processed