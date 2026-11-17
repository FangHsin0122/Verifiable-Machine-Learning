"""
Control and Observer Extensions for Symbolic Dynamical Systems

Add these methods to SymbolicDynamicalSystem and GenericDiscreteTimeSystem classes
to provide LQR, LQG, and EKF functionality.
"""

import numpy as np
import scipy
import scipy.linalg
import torch
from typing import Tuple, Optional
from neural_lyapunov_training.symbolic_dynamics import (
    GenericDiscreteTimeSystem,
    IntegrationMethod,
    ExtendedKalmanFilter,
    LinearController,
    LinearObserver
)
from neural_lyapunov_training.symbolic_systems import SymbolicPendulum


def demo_lqr_lqg():
    """Demonstrate LQR/LQG functionality"""

    print("=" * 70)
    print("LQR/LQG Demo")
    print("=" * 70)

    # Create system
    pendulum_ct = SymbolicPendulum(m=1.0, l=1.0, beta=0.1, g=9.81)
    pendulum_dt = GenericDiscreteTimeSystem(
        pendulum_ct, dt=0.01, integration_method=IntegrationMethod.RK4
    )

    # Design LQR controller
    print("\n--- LQR Control Design ---")
    Q = np.diag([10.0, 1.0])  # Penalize angle more than velocity
    R = np.array([[0.1]])  # Control effort cost

    K_ct, S_ct = pendulum_ct.lqr_control(Q, R)
    print(f"Continuous-time LQR gain K: {K_ct}")
    print(f"Riccati solution S:\n{S_ct}")

    K_dt, S_dt = pendulum_dt.dlqr_control(Q, R)
    print(f"\nDiscrete-time LQR gain K: {K_dt}")

    # Design Kalman filter
    print("\n--- Kalman Filter Design ---")
    Q_process = np.eye(2) * 0.01
    R_meas = np.array([[0.1]])

    L_ct = pendulum_ct.kalman_gain(Q_process, R_meas)
    print(f"Continuous-time Kalman gain L: {L_ct.T}")

    L_dt = pendulum_dt.discrete_kalman_gain(Q_process, R_meas)
    print(f"Discrete-time Kalman gain L: {L_dt.T}")

    # Design LQG controller
    print("\n--- LQG Control Design ---")
    K_lqg, L_lqg = pendulum_dt.dlqg_control(Q, R, Q_process, R_meas)
    print(f"LQG Control gain K: {K_lqg}")
    print(f"LQG Observer gain L: {L_lqg.T}")

    # Closed-loop analysis
    print("\n--- Closed-Loop Analysis ---")
    A_cl = pendulum_dt.dlqg_closed_loop_matrix(K_lqg, L_lqg)
    eigs_cl = np.linalg.eigvals(A_cl)
    print(f"Closed-loop eigenvalues: {eigs_cl}")
    print(f"Max eigenvalue magnitude: {np.abs(eigs_cl).max():.4f}")
    print(f"Stable? {np.all(np.abs(eigs_cl) < 1.0)}")

    # Test LQR controller
    print("\n--- Testing LQR Controller ---")
    controller = LinearController(
        K_dt, pendulum_dt.x_equilibrium, pendulum_dt.u_equilibrium
    )

    x0 = torch.tensor([0.5, 0.0])
    u0 = controller(x0)
    print(f"State: {x0.numpy()}")
    print(f"Control: {u0.numpy()}")

    # Simulate closed-loop with LQR
    print("\n--- Simulating Closed-Loop System ---")
    x = x0.clone()
    trajectory = [x.numpy()]
    controls = []

    for t in range(100):
        u = controller(x)
        controls.append(u.numpy())
        x = pendulum_dt(x, u)
        trajectory.append(x.numpy())

    trajectory = np.array(trajectory)
    controls = np.array(controls)

    print(f"Initial state: {trajectory[0]}")
    print(f"Final state: {trajectory[-1]}")
    print(f"Converged to equilibrium? {np.allclose(trajectory[-1], [0, 0], atol=1e-2)}")

    # Test EKF
    print("\n--- Testing Extended Kalman Filter ---")
    ekf = ExtendedKalmanFilter(pendulum_dt, Q_process, R_meas)

    x_true = torch.tensor([0.3, 0.2])
    ekf.reset(x0=torch.tensor([0.5, 0.5]))  # Wrong initial guess

    for t in range(20):
        u = torch.zeros(1)

        # Simulate true system (with noise)
        x_true = pendulum_dt(x_true, u)
        y_meas = pendulum_ct.h(x_true.unsqueeze(0)).squeeze(0)
        y_meas = y_meas + torch.randn_like(y_meas) * 0.05  # Add noise

        # EKF update
        ekf.predict(u)
        ekf.update(y_meas)

        if t % 5 == 0:
            error = torch.norm(ekf.x_hat - x_true).item()
            print(f"  Step {t:2d}: estimation error = {error:.4f}")

    print("\n✓ LQR/LQG/EKF demo complete!")


if __name__ == "__main__":
    demo_lqr_lqg()
