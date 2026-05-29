# GLENS: Global Search via Learning from Solver Iterates with Diffusion Models

This repository contains the official implementation of **GLENS: Global Search via Learning from Solver Iterates with Diffusion Models** by [Anjian Li](https://anjianli21.github.io/), [Bartolomeo Stellato](https://stella.to/), and [Ryne Beeson](https://beeson.princeton.edu/) at Princeton University.

GLENS is a diffusion-based method for generating diverse, high-quality initial guesses for nonconvex continuous optimization. It uses intermediate solver iterates as free data augmentation and combines two learned components:

- **Neighborhood Structure Model (NS):** learns the local geometry around optima conditioned on problem parameters.
- **Solver Behavior Model (SB):** learns refinement directions that guide samples toward nearby optima.

For questions, contact Anjian Li at [anjianl@princeton.edu](mailto:anjianl@princeton.edu).

## Repository Structure

```text
dataset/    Dataset loaders for all benchmark tasks.
model/      Diffusion, condition encoder, and solver-behavior model code.
run/        Python entry points for training, sampling, evaluation, and plotting.
script/     Shell scripts that reproduce the training and testing workflows.
```

## Environment

We recommend [Conda](https://docs.conda.io/). From the repository root:

```bash
conda create -n glens python=3.8
conda activate glens
pip install -r requirements.txt
```

The code uses PyTorch and will use a GPU when one is available.

## Data

Training data and precomputed data-statistics pickles are available on Hugging Face: [Anjian/GLENS](https://huggingface.co/datasets/Anjian/GLENS/tree/main). The scripts refer to these pickles through `DATA_STAT_FILE_PATH` or `data_stat_file_path_list`.

Use the statistics files as follows:

| Task | Statistics file rule |
| --- | --- |
| `qp_constrained`, `levy`, `himmelblau`, `rosenbrock` | Use the `k=1` statistics for DiffuSolve `k=1`. Use the `k=10` statistics for DiffuSolve `k=10`, NS `k=10`, and GLENS `k=10`. |
| `robot` | Use the same statistics file for all methods. |

## Training

Training scripts live under `script/train/<task>/`. The non-robot tasks use a `dim_100/` subdirectory; the robot task uses `script/train/robot/`.

Each task provides scripts for:

| Component | Script pattern |
| --- | --- |
| DiffuSolve, `k=1` | `*_diffusolve_k_1.sh` |
| DiffuSolve, `k=10` | `*_diffusolve_k_10.sh` |
| Neighborhood Structure Model (NS), `k=10` | `*_NS_k_10.sh` |
| Solver Behavior Model (SB), `k=10` | `*_dynamics_k_10.sh` |

Before running a script, edit the placeholders near the top: `PROJECT_DIR`, `DATA_ROOT_DIR`, `RESULT_FOLDER`, `DATA_STAT_FILE_PATH`, and the optional Weights & Biases settings (`WANDB_API_KEY`, `WANDB_PROJECT_NAME`, `WANDB_RUN_NAME`, `WANDB_MODE`).

Example: train the NS model for `qp_constrained` at dimension 100:

```bash
. script/train/qp_constrained/dim_100/qp_constrained_NS_k_10.sh
```

The directory `script/train/qp_constrained/example/` contains filled example scripts for the `qp_constrained` setup.

## Testing

Evaluation scripts live under `script/test/<task>/`. The standard pipeline has three stages:

1. `generate_test_sample.sh` generates test-sample pickles for all four methods.
2. `compute_k_neighbor.sh` computes k-neighborhood results from those sample pickles.
3. `plot_results_from_k_neighbor.sh` plots cumulative curves and writes summary statistics.

Across all three stages, method-specific arrays must stay in this order:

1. DiffuSolve, `k=1`
2. DiffuSolve, `k=10`
3. NS only, `k=10`
4. GLENS, `k=10` (`NS + SB`)

This ordering applies to arrays such as `checkpoint_parent_dir_list`, `solver_behavior_checkpoint_parent_dir_list`, `data_stat_file_path_list`, `input_file_paths`, `output_file_paths`, and `file_paths`.

Example: evaluate `qp_constrained` at dimension 100:

```bash
. script/test/qp_constrained/dim_100/generate_test_sample.sh
. script/test/qp_constrained/dim_100/compute_k_neighbor.sh
. script/test/qp_constrained/dim_100/plot_results_from_k_neighbor.sh
```

The directory `script/test/qp_constrained/example/` contains filled example scripts for the `qp_constrained` setup.

## Robot Evaluation Note

The robot k-neighborhood evaluation uses the commercial **SNOPT** solver. Without SNOPT, you can still visualize robot samples with:

```bash
. script/test/robot/plot_samples.sh
```

Code for computing robot k-neighborhood results can be requested separately.

## Citation

If you use this repository, please cite the GLENS paper. BibTeX will be added when available.
