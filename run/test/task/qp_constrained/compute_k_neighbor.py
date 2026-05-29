import os
import numpy as np
import pickle
import copy
import argparse
import torch

def compute_gradient_and_objective(x, problem_parameters):
    lambda_value = problem_parameters["lambda_value"]

    x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    lambda_torch = torch.tensor(lambda_value, dtype=torch.float64, requires_grad=False).reshape(-1)

    n = x_torch.shape[0]
    lambda_1 = lambda_torch[0]
    # lambda_2 = lambda_torch[1]
    # lambda_3 = lambda_torch[2]
    # lambda_4 = lambda_torch[3]

    # Quadratic term
    quadratic_term = 0.5 * torch.dot(x_torch, x_torch)

    # Pairwise bounds: n/2 terms, lambda_1 * sum_i softplus(L1 - (x_{2i-1} + x_{2i}))
    L1 = 0.0
    # L2 = 1.0
    # L3 = 1.0
    # L4 = 1.0
    x_1_torch = x_torch[0::2]
    x_2_torch = x_torch[1::2]
    penalty_term_1 = lambda_1 * torch.nn.functional.softplus(L1 - (x_1_torch + x_2_torch)).sum()
    # penalty_term_2 = lambda_2 * torch.nn.functional.softplus(L2 - (x_1_torch - x_2_torch)).sum()
    # penalty_term_3 = lambda_3 * torch.nn.functional.softplus(L3 - (- x_1_torch + x_2_torch)).sum()
    # penalty_term_4 = lambda_4 * torch.nn.functional.softplus(L4 - (- x_1_torch - x_2_torch)).sum()

    # objective_value = quadratic_term + penalty_term_1 + penalty_term_2 + penalty_term_3 + penalty_term_4
    objective_value = quadratic_term + penalty_term_1

    objective_value.backward()
    
    # Extract gradient and convert to numpy
    grad = x_torch.grad.detach().numpy()
    
    # Convert objective value to numpy scalar
    objective_value_np = objective_value.item()
    
    return grad, objective_value_np


def setup_problem_parameters(lambda_value):

    problem_parameters = {
            "lambda_value": lambda_value,
    }

    return problem_parameters

def solve_problem_and_get_k_neighbor_idx(x_i, problem_parameters, step_size=0.1, max_iter=100):
    # Initialize iterate storage
    curr_x = copy.deepcopy(x_i)
    prev_x = np.ones_like(curr_x) * np.inf

    # Fixed tolerance
    tolerance = 1e-2
    
    # Gradient descent iterations
    # Start at 0 to match original logic: increment after each update
    curr_iter = 0
    convergence_flag = False
    while curr_iter < max_iter:
        # Check convergence based on residual
        if np.linalg.norm(curr_x - prev_x) <= tolerance:
            convergence_flag = True
            break
        
        # Gradient descent update
        grad, _ = compute_gradient_and_objective(curr_x, problem_parameters)
        prev_x = copy.deepcopy(curr_x)
        curr_x = curr_x - step_size * grad
        
        # Update iteration counter
        curr_iter += 1
    
    # Return the last iteration index (k-neighbor)
    last_iter_idx = curr_iter

    return last_iter_idx
    

def collect_k_neighbor_results(file_path: str, problem_type: str, step_size=0.1, max_iter=100):
    # Load data, mapping CUDA tensors to CPU
    # Try torch.load first (most efficient for torch.save files)
    try:
        data = torch.load(file_path, map_location='cpu', weights_only=False)
    except:
        # Fallback: use pickle with CPU-mapped restore_location
        original_restore = torch.serialization.default_restore_location
        torch.serialization.default_restore_location = lambda storage, loc: original_restore(storage, 'cpu' if isinstance(loc, str) and loc.startswith('cuda') else loc)
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
        finally:
            torch.serialization.default_restore_location = original_restore

    x_list = data["x"]
    lambda_value_list = data["condition_lambda_value"]

    k_neighbor_list = []
    for x, lambda_value in zip(x_list, lambda_value_list):
        batch_num = x.shape[0]

        for i in range(batch_num):
            sample_num = x[i].shape[0]

            for j in range(sample_num):
                
                # Setup problem parameters
                problem_parameters = setup_problem_parameters(lambda_value[i])

                # Get the initial guess
                x_i = x[i][j]

                # Solve the problem and get the k-neighbor index
                k_neighbor_idx = solve_problem_and_get_k_neighbor_idx(x_i, problem_parameters, step_size=step_size, max_iter=max_iter)

                k_neighbor_list.append(k_neighbor_idx)

    return k_neighbor_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute k-neighbor results from input data files")
    parser.add_argument("--file_paths", 
                        type=lambda x: [s.strip() for s in x.split(',')], 
                        required=True,
                        help="comma-separated list of input pickle file paths")
    parser.add_argument("--output_paths", 
                        type=lambda x: [s.strip() for s in x.split(',')], 
                        required=True,
                        help="comma-separated list of output pickle file paths (must match number of input file paths)")
    parser.add_argument("--problem_type", 
                        type=str, 
                        default="qp_constrained", 
                        help="problem type (for internal use)")
    parser.add_argument('--step_size',
                        type=float,
                        default=0.1,
                        help="step size for gradient descent")
    parser.add_argument('--max_iter',
                        type=int,
                        default=100,
                        help="maximum number of iterations")

    args = parser.parse_args()

    # Validate that file_paths and output_paths have the same length
    if len(args.file_paths) != len(args.output_paths):
        raise ValueError(f"Number of input file paths ({len(args.file_paths)}) must equal number of output file paths ({len(args.output_paths)}).")

    # Process each file path
    for idx, (input_path, output_path) in enumerate(zip(args.file_paths, args.output_paths)):
        if not os.path.exists(input_path):
            print(f"Warning: Input file not found: {input_path}")
            continue

        print(f"Processing file {idx+1}/{len(args.file_paths)}: {input_path}")

        print(f"Step size: {args.step_size}, Max iter: {args.max_iter}, Tolerance: 1e-2")
        # Collect k-neighbor results
        k_neighbors = collect_k_neighbor_results(input_path, problem_type=args.problem_type, step_size=args.step_size, max_iter=args.max_iter)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save k-neighbor results
        with open(output_path, "wb") as f:
            pickle.dump({"k_neighbors": k_neighbors}, f)
        
        print(f"Saved {len(k_neighbors)} k-neighbor results to {output_path}")
