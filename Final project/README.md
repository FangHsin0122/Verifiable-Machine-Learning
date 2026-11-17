# Final project

Group project: extend the code from *Lyapunov-stable Neural Control for State and Output Feedback: A Novel Formulation* (Yang, Dai, Shi, Hsieh, Tedrake, Zhang) so we can train and verify controllers for user-defined nonlinear systems, not just the four hard-coded ones. The paper is in [../Papers/Lyapunov barrier and contraction certificates/](<../Papers/Lyapunov barrier and contraction certificates/README.md>).

| File / folder | What it is |
|---|---|
| [Lyapunov_Stable_NN_Controllers-main/](Lyapunov_Stable_NN_Controllers-main/README.md) | The original authors' code. Trains a neural controller and Lyapunov function together, then verifies them with α,β-CROWN. Systems: pendulum, path tracking, PVTOL, 2D quadrotor |
| [Lyapunov_Stable_NN_Controllers_Custom_Dynamics-main/](Lyapunov_Stable_NN_Controllers_Custom_Dynamics-main/README.md) | Our group's fork (see below) |
| [CS 7268 Group Project Presentation.pdf](<CS 7268 Group Project Presentation.pdf>) | Our final presentation (Gil Benezer and Cindy Li): *Specifying Dynamical Systems for Neural CLF Training: Moving Towards User-Friendly Interfaces*. Exported from Google Slides |
| [Lyapunov_Stable_NN_Controllers_Custom_Dynamics-main.zip](Lyapunov_Stable_NN_Controllers_Custom_Dynamics-main.zip) | Zip of the fork |

## What the fork adds

- `neural_lyapunov_training/symbolic_dynamics.py`, `symbolic_systems.py`: define a system's dynamics as symbolic equations instead of a hard-coded class.
- `examples/*_symbolic.py` + matching `examples/config/*_symbolic.yaml`: training scripts that use the symbolic dynamics, including a new cartpole system.
- `neural_lyapunov_training/roa_metrics.py`, `lyapunov_roa_visualization.py`: measure and plot the region of attraction (ROA) to compare against the original.
- `neural_lyapunov_training/rho_bisect.py`: bisection on the ROA level ρ (replaces the original `bisect.py`).
- `examples/config/user/`: each group member's run configs.
- `slurm_job_script_*.sh`: cluster job scripts for different GPU types (H200, PCIe, SXM2).
- `alpha-beta-CROWN/`: empty here. Clone it as the fork's README says before running verification.
