#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1 # number of nodes ()
#SBATCH --gres=gpu:v100-pcie:1 # type of GPU
#SBATCH --time=02:00:00
#SBATCH --job-name=pendulum_state_v100-pcie_train_controller_symbolic
#SBATCH --mem=8GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=output/benezerg/pendulum_state/job_logs/pendulum_state_v100-pcie_train_controller_symbolic.%j.out
#SBATCH --error=output/benezerg/pendulum_state/job_logs/pendulum_state_v100-pcie_train_controller_symbolic.%j.err

export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/alpha-beta-CROWN:$(pwd)/alpha-beta-CROWN/complete_verifier"
python examples/pendulum_state_training_symbolic.py user=benezerg_pendulum_state_training_pcie
