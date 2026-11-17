# Final project – User-Defined Dynamics for Neural Lyapunov Control

Group: Gil Benezer and Fang-Hsin "Cindy" Li. Presented 12/10/2025.
Fork: <https://github.com/gbenezer/Lyapunov_Stable_NN_Controllers_Custom_Dynamics>

## The big idea

HW3 checked safety by bounding where a closed-loop system can go, step by step. A **certificate function** takes a different route: find one function that *proves* a property for the whole region at once. A **control Lyapunov function (CLF)** V(x) proves the controller drives the system to an equilibrium: V is zero at the equilibrium, positive everywhere else, and always decreases along trajectories. The set of states the proof covers is the **region of attraction (ROA)**.

The state-of-the-art method we built on, *Lyapunov-stable Neural Control for State and Output Feedback* (Yang, Dai, Shi, Hsieh, Tedrake, Zhang), trains a neural controller and a Lyapunov function together, then verifies the Lyapunov conditions with α,β-CROWN (the bound-propagation tools from HW2) instead of slower MILP/SMT solvers.

**The problem we tackled:** that code is hard to use on a new system. The four systems (pendulum, path tracking, PVTOL, 2D quadrotor) are hard-coded Python classes, some example scripts don't reproduce the paper by default, the plots show the *open-loop* Lyapunov function, and there's no way to measure ROA size.

**Our goal:** let a user describe *any* nonlinear system as symbolic equations and plug it into the same training and verification pipeline, then check that this gives the same results as the hard-coded versions.

## What we built

- **Symbolic system interface** (`symbolic_dynamics.py`, `symbolic_systems.py`): a `SymbolicDynamicalSystem` superclass where the user defines state, control, and output variables as SymPy symbols and writes the dynamics. It can evaluate the dynamics symbolically and numerically, linearize at an equilibrium, generate NumPy/PyTorch functions, check stability, and compute continuous-time LQR and Kalman gains.
- **`GenericDiscreteTimeSystem` wrapper**: makes a symbolic system work with the existing training code (backward compatible with its `DiscreteTimeSystem`). Adds trajectory simulation, discrete-time LQR/Kalman gains, and interactive HTML trajectory and phase-portrait plots.
- **Lyapunov plots**: 2D and 3D interactive plots of the learned Lyapunov function and its closed-loop derivative.
- **ROA measurement** (`roa_metrics.py`, `rho_bisect.py`): quasi-Monte Carlo sampling to estimate what fraction of the domain the ROA covers, and to flag a candidate as invalid if V increases anywhere inside it.
- **New training scripts and configs**: `examples/*_symbolic.py` and `examples/config/*_symbolic.yaml`, plus SLURM job scripts for cluster GPUs.

## Results

- **Pendulum**: symbolic and hard-coded versions perform about the same. The plots looked different, so we compared them with numbers. Training doesn't always find a verified Lyapunov function, and in this case it was a learnable quadratic, not a neural network.
- **Path tracking**: also matches, this time with a neural-network Lyapunov function. With the right hyperparameters it finds a valid one every time. The difference from the pendulum is that path tracking has a stable equilibrium; the inverted pendulum is unstable.
- **New systems (cartpole, PVTOL)**: training didn't converge. Most likely cause: hyperparameter tuning. Training is a bi-level optimization with competing objectives, so the settings make or break it.

## Deliverables

| Deliverable | Status |
|---|---|
| Symbolic dynamics framework, integrated into the original code base | Done (fork) |
| Lyapunov/derivative plotting and ROA measurement | Done |
| Comparison of hard-coded vs. symbolic systems (pendulum, path tracking) | Done |
| Training on systems without author examples (cartpole, PVTOL) | Attempted; did not converge |
| Final presentation | Done: `CS 7268 Group Project Presentation.pdf` |
| Fork README (modifications, usage, results, scalability) | Not done: those sections are still `TODO` |
| Scalability study (state dimension, network size/depth) | Planned in the README; no results |
| Running the verification step (α,β-CROWN) on our candidates | Not done: the current code only produces candidates |

## Next steps (from the presentation)

- Finish the README.
- Find easier, more intuitive ways to set training hyperparameters.
- Show candidate generation and controller synthesis on new systems.
- Run the authors' verification step on our candidates.
- Better support for partially observable systems and multiple equilibria.
