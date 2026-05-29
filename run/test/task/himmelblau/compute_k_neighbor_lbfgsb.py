import os
import numpy as np
import pickle
import copy
import argparse
import torch
from scipy.optimize import minimize

def compute_gradient_and_objective(x, problem_parameters):
    lambda_value = problem_parameters["lambda_value"]
    chain_coupling_weight = problem_parameters.get("chain_coupling_weight", 0.0)

    # Convert x to torch tensor with gradient tracking
    x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=True)

    # Convert lambda to torch tensor (no gradient needed for lambda)
    lambda_torch = torch.tensor(lambda_value, dtype=torch.float64, requires_grad=False)

    # x is grouped in pairs (x1, x2); each pair corresponds to (lambda1, lambda2)
    if x_torch.numel() % 2 != 0:
        raise ValueError("The dimension of x must be even, since each objective term uses (x1, x2).")

    # Reshape into pairs
    x_pairs = x_torch.view(-1, 2)              # shape: (n_pairs, 2)

    x1 = x_pairs[:, 0]
    x2 = x_pairs[:, 1]
    lambda1 = lambda_torch[0]
    lambda2 = lambda_torch[1]

    # For each pair:
    # (x1^2 + x2 - lambda1)^2 + (x1 + x2^2 - lambda2)^2
    term1 = (x1**2 + x2 - lambda1) ** 2
    term2 = (x1 + x2**2 - lambda2) ** 2

    # Final objective is the average over all pairs
    num_pairs = x_pairs.shape[0]
    objective_value = (term1 + term2).sum() / num_pairs

    # Chain coupling term: weight * (1/(m-1)) * sum_{i=1}^{m-1} [(x_{2i+1} - x_{2i-1})^2 + (x_{2i+2} - x_{2i})^2]
    if chain_coupling_weight != 0.0 and num_pairs >= 2:
        first_diff = x_pairs[1:, 0] - x_pairs[:-1, 0]   # x_{2i+1} - x_{2i-1} in 1-indexed
        second_diff = x_pairs[1:, 1] - x_pairs[:-1, 1]  # x_{2i+2} - x_{2i} in 1-indexed
        num_chain_terms = num_pairs - 1
        chain_term = chain_coupling_weight * ((first_diff ** 2) + (second_diff ** 2)).sum() / num_chain_terms
        objective_value = objective_value + chain_term

    # Compute gradient using autograd
    objective_value.backward()
    
    # Extract gradient and convert to numpy
    grad = x_torch.grad.detach().numpy()
    
    # Convert objective value to numpy scalar
    objective_value_np = objective_value.item()
    
    return grad, objective_value_np


def setup_problem_parameters(lambda_value, chain_coupling_weight=0.0):
    problem_parameters = {
        "lambda_value": lambda_value,
        "chain_coupling_weight": chain_coupling_weight,
    }
    return problem_parameters

def solve_problem_and_get_k_neighbor_idx(x_i, problem_parameters, error_type="tolerance", error_tolerance=1e-3):
    # Initialize iterate storage similar to gradient_descent_torch_solver
    curr_x = copy.deepcopy(x_i)
    prev_x = np.ones_like(curr_x) * np.inf

    results = {}
    results["solver_info"] = {}

    # Iteration 0 (initial point)
    curr_iter = 0
    grad, objective_value = compute_gradient_and_objective(curr_x, problem_parameters)
    residual = np.linalg.norm(curr_x - prev_x)

    results["solver_info"][curr_iter] = {}
    results["solver_info"][curr_iter]["x"] = curr_x
    results["solver_info"][curr_iter]["grad"] = grad
    results["solver_info"][curr_iter]["objective_value"] = objective_value
    results["solver_info"][curr_iter]["residual"] = residual

    # State for callback
    convergence_flag = False

    ###########################################################################################
    
    def obj_fun(x):
        # Objective for scipy: returns scalar
        _, f_val = compute_gradient_and_objective(x, problem_parameters)
        return f_val

    def obj_jac(x):
        # Gradient (Jacobian) for scipy: returns numpy array
        g, _ = compute_gradient_and_objective(x, problem_parameters)
        return g
    
    def callback(xk):
        nonlocal curr_iter, curr_x, prev_x, grad, objective_value

        # Update iteration counter
        curr_iter += 1

        # Track previous and current x
        prev_x = curr_x.copy()
        curr_x = xk.copy()

        # Compute gradient and objective at current point
        grad, objective_value = compute_gradient_and_objective(curr_x, problem_parameters)
        residual_local = np.linalg.norm(curr_x - prev_x)

        # Save history
        results["solver_info"][curr_iter] = {}
        results["solver_info"][curr_iter]["x"] = curr_x
        results["solver_info"][curr_iter]["grad"] = grad
        results["solver_info"][curr_iter]["objective_value"] = objective_value
        results["solver_info"][curr_iter]["residual"] = residual_local
    
    # Set optimizer options
    options = {
        "maxiter": 300,
        # 'ftol': 1e-4,
    }

    # Run optimization
    if error_type == "tolerance":
        result = minimize(
            fun=obj_fun,
            x0=x_i,
            method="L-BFGS-B",
            jac=obj_jac,
            callback=callback,
            options=options,
            tol=error_tolerance,
        )
    else:
        result = minimize(
            fun=obj_fun,
            x0=x_i,
            method="L-BFGS-B",
            jac=obj_jac,
            callback=callback,
            options=options,
        )
    
    convergence_flag = bool(result.success)
    if convergence_flag:
        last_iter_idx = max(results["solver_info"].keys())
    else:
        last_iter_idx = 300

    return last_iter_idx
    

def collect_k_neighbor_results(file_path: str, problem_type: str, error_type="tolerance", error_tolerance=1e-3, chain_coupling_weight=0.0):
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
                problem_parameters = setup_problem_parameters(lambda_value[i], chain_coupling_weight=chain_coupling_weight)

                # Get the initial guess
                x_i = x[i][j]

                # Solve the problem and get the k-neighbor index
                k_neighbor_idx = solve_problem_and_get_k_neighbor_idx(x_i, problem_parameters, error_type=error_type, error_tolerance=error_tolerance)

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
                        default="himmelblau", 
                        help="problem type (for internal use)")
    parser.add_argument('--error_type',
                        type=str,
                        default="none",
                        choices=["tolerance", "none"],
                        help="type of error to use")
    parser.add_argument('--error_tolerance',
                        type=float,
                        default=1e-3,
                        help="error tolerance for convergence")
    parser.add_argument('--chain_coupling_weight',
                        type=float,
                        default=0.0,
                        help="chain coupling weight for Himmelblau objective (default: 0.0)")

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

        print(f"Error type: {args.error_type}, Error tolerance: {args.error_tolerance}, Chain coupling weight: {args.chain_coupling_weight}")
        # Collect k-neighbor results
        k_neighbors = collect_k_neighbor_results(input_path, problem_type=args.problem_type, error_type=args.error_type, error_tolerance=args.error_tolerance, chain_coupling_weight=args.chain_coupling_weight)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save k-neighbor results
        with open(output_path, "wb") as f:
            pickle.dump({"k_neighbors": k_neighbors}, f)
        
        print(f"Saved {len(k_neighbors)} k-neighbor results to {output_path}")

