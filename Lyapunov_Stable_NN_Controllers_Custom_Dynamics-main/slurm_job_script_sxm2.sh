#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1 # number of nodes ()
#SBATCH --gres=gpu:v100-sxm2:1 # type of GPU
#SBATCH --time=02:00:00
#SBATCH --job-name=quadrotor2d_output_v100-sxm2_train_controller
#SBATCH --mem=8GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=output/benezerg/quadrotor2d_output/job_logs/quadrotor2d_output_v100-sxm2_train_controller.%j.out
#SBATCH --error=output/benezerg/quadrotor2d_output/job_logs/quadrotor2d_output_v100-sxm2_train_controller.%j.err

export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/alpha-beta-CROWN:$(pwd)/alpha-beta-CROWN/complete_verifier"
python examples/quadrotor2d_output_training.py user=benezerg_quadrotor2d_output_training_sxm2
