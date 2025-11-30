import os
from path import Path
import pdb
from datetime import datetime

import argparse
import hydra
import logging
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import scipy.linalg
import torch
import torch.nn as nn
import wandb

import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.models as models
import neural_lyapunov_training.pendulum as pendulum
import neural_lyapunov_training.train_utils as train_utils
import neural_lyapunov_training.symbolic_dynamics as sd
import neural_lyapunov_training.symbolic_systems as ss
import neural_lyapunov_training.lyapunov_roa_visualization as lrv
import neural_lyapunov_training.roa_metrics as rmet

device = torch.device("cuda")
dtype = torch.float


def generate_candidate_states(limit, num_per_dim=3):
    """Generate grid of candidate states"""
    ranges = []
    for i in range(len(limit)):
        ranges.append(torch.linspace(-limit[i], limit[i], num_per_dim, device=device))

    # Create meshgrid
    grids = torch.meshgrid(*ranges, indexing="ij")
    candidates = torch.stack([g.flatten() for g in grids], dim=1)

    # Remove equilibrium
    candidates = candidates[(candidates.abs().sum(dim=1) > 0.01)]

    return candidates


def main():

    train_utils.set_seed(42)

    dt = 0.01
    cartpole_continuous = ss.CartPole(
        m_cart=1.0, m_pole=0.1, length=1.0, gravity=9.81, friction=0.1
    )

    dynamics = sd.GenericDiscreteTimeSystem(
        cartpole_continuous,
        dt=dt,
        integration_method=sd.IntegrationMethod.RK4,
        position_integration=sd.IntegrationMethod.RK4,
    )

    controller = controllers.NeuralNetworkController(
        nlayer=4,
        in_dim=4,
        out_dim=1,
        hidden_dim=8,
        clip_output="clamp",
        u_lo=torch.tensor([-30.0]),
        u_up=torch.tensor([30.0]),
        x_equilibrium=cartpole_continuous.x_equilibrium,
        u_equilibrium=cartpole_continuous.u_equilibrium,
    )
    controller.train()

    absolute_output = True
    # lyapunov_nn = lyapunov.NeuralNetworkLyapunov(
    #     goal_state=torch.tensor([0.0, 0.0, 0.0, 0.0]),
    #     hidden_widths=[16, 16, 8],
    #     x_dim=4,
    #     R_rows=6,
    #     absolute_output=absolute_output,
    #     eps=0.01,
    #     activation=nn.LeakyReLU,
    #     V_psd_form="L1",
    # )

    Q = np.diag(np.array([1.0, 1.0, 10.0, 10.0]))
    R = np.diag(np.array([10.0]))
    K, S = cartpole_continuous.lqr_control(Q, R)
    K_torch = torch.from_numpy(K).type(dtype).to(device)
    S_torch = torch.from_numpy(S).type(dtype).to(device)
    R = torch.linalg.cholesky(S_torch)
    lyapunov_nn = lyapunov.NeuralNetworkQuadraticLyapunov(
        goal_state=torch.zeros(4, dtype=dtype).to(device),
        x_dim=4,
        R_rows=4,
        eps=0.01,
        R=R,
    )
    lyapunov_nn.train()

    kappa = 0.01
    rho_multiplier = 2.25
    # Place holder for lyaloss
    derivative_lyaloss = lyapunov.LyapunovDerivativeLoss(
        dynamics,
        controller,
        lyapunov_nn,
        box_lo=0,
        box_up=0,
        rho_multiplier=rho_multiplier,
        kappa=kappa,
        hard_max=True,
    )

    dynamics.to(device)
    controller.to(device)
    lyapunov_nn.to(device)
    grid_size = torch.tensor([50, 50, 50, 50], device=device)
    # approximate_controller(controller_target, controller, 2, limit, 0, 0, "examples/pendulum_controller.pth", batch_size=10000, max_iter=500)
    # approximate_controller(lyapunov_target, lyapunov_nn, 2, limit, 0, 0, "examples/pendulum_lyapunov.pth", batch_size=10000, max_iter=500)
    logger = logging.getLogger(__name__)

    if absolute_output:
        positivity_lyaloss = None
    else:
        positivity_lyaloss = lyapunov.LyapunovPositivityLoss(
            lyapunov_nn, 0.01 * torch.eye(2, device=device)
        )

    wandb.init(
        project="CS-7268-Group-Project",
        entity="GB-Northeastern-Projects",
        name=f"{datetime.now():%Y-%m-%d}_{datetime.now():%H-%M-%S}_cartpole_state",
    )

    candidate_scale = 2.0
    candidate_roa_states_weight = 1.0e-05
    suffix = "symbolic"
    data_folder = f"./output/benezerg/cartpole_state/{datetime.now():%Y-%m-%d}/{datetime.now():%H-%M-%S}_{suffix}"
    os.makedirs(data_folder, exist_ok=True)
    save_lyaloss = True
    V_decrease_within_roa = True
    save_lyaloss_path = None
    save_name = (
        f"lyaloss_{kappa}kappa_{candidate_scale}_{candidate_roa_states_weight}.pth"
    )
    if save_lyaloss:
        save_lyaloss_path = f"{data_folder}/{save_name}"
    else:
        save_lyaloss_path = None

    limit_scale = [0.1, 0.25, 0.5, 1.0]
    limit_base = torch.tensor([0.5, 0.3, 0.5, 0.5], dtype=dtype, device=device)
    for scale in limit_scale:
        limit = scale * limit_base
        lower_limit = -1 * limit
        upper_limit = 1 * limit

        derivative_lyaloss = lyapunov.LyapunovDerivativeLoss(
            dynamics,
            controller,
            lyapunov_nn,
            box_lo=lower_limit,
            box_up=upper_limit,
            rho_multiplier=rho_multiplier,
            kappa=kappa,
            hard_max=True,
        )

        # if save_lyaloss:
        #     save_lyaloss_path = os.path.join(os.getcwd(), f"lyaloss_{limit_scale}.pth")
        # else:
        #     save_lyaloss_path = None

        # candidate_roa_state_list = [
        #     [0.0, 0.0, 0.0, 0.0],
        #     [0.01 * 5, 0.01 * np.pi, 0.01 * 1.0, 0.01 * 1.0],
        #     [-0.01 * 5, -0.01 * np.pi, -0.01 * 1.0, -0.01 * 1.0],
        #     [0.01 * 5, -0.01 * np.pi, -0.01 * 1.0, -0.01 * 1.0],
        #     [-0.01 * 5, 0.01 * np.pi, -0.01 * 1.0, -0.01 * 1.0],
        #     [-0.01 * 5, -0.01 * np.pi, 0.01 * 1.0, -0.01 * 1.0],
        #     [-0.01 * 5, -0.01 * np.pi, -0.01 * 1.0, 0.01 * 1.0],
        #     [0.01 * 5, 0.01 * np.pi, -0.01 * 1.0, -0.01 * 1.0],
        #     [0.01 * 5, -0.01 * np.pi, 0.01 * 1.0, -0.01 * 1.0],
        #     [0.01 * 5, -0.01 * np.pi, -0.01 * 1.0, 0.01 * 1.0],
        #     [-0.01 * 5, 0.01 * np.pi, 0.01 * 1.0, -0.01 * 1.0],
        #     [-0.01 * 5, 0.01 * np.pi, -0.01 * 1.0, 0.01 * 1.0],
        #     [-0.01 * 5, -0.01 * np.pi, 0.01 * 1.0, 0.01 * 1.0],
        #     [0.1 * 5, 0.1 * np.pi, 0.1 * 1.0, 0.1 * 1.0],
        #     [-0.1 * 5, -0.1 * np.pi, -0.1 * 1.0, -0.1 * 1.0],
        #     [0.1 * 5, -0.1 * np.pi, -0.1 * 1.0, -0.1 * 1.0],
        #     [-0.1 * 5, 0.1 * np.pi, -0.1 * 1.0, -0.1 * 1.0],
        #     [-0.1 * 5, -0.1 * np.pi, 0.1 * 1.0, -0.1 * 1.0],
        #     [-0.1 * 5, -0.1 * np.pi, -0.1 * 1.0, 0.1 * 1.0],
        #     [0.1 * 5, 0.1 * np.pi, -0.1 * 1.0, -0.1 * 1.0],
        #     [0.1 * 5, -0.1 * np.pi, 0.1 * 1.0, -0.1 * 1.0],
        #     [0.1 * 5, -0.1 * np.pi, -0.1 * 1.0, 0.1 * 1.0],
        #     [-0.1 * 5, 0.1 * np.pi, 0.1 * 1.0, -0.1 * 1.0],
        #     [-0.1 * 5, 0.1 * np.pi, -0.1 * 1.0, 0.1 * 1.0],
        #     [-0.1 * 5, -0.1 * np.pi, 0.1 * 1.0, 0.1 * 1.0],
        # ]

        candidate_roa_states = generate_candidate_states(limit * 0.9, num_per_dim=3)
        # candidate_roa_states = current_limit_scale * torch.tensor(
        #     candidate_roa_state_list,
        #     device=device,
        # )

        train_utils.train_lyapunov_with_buffer(
            derivative_lyaloss=derivative_lyaloss,
            positivity_lyaloss=positivity_lyaloss,
            observer_loss=None,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            grid_size=grid_size,
            learning_rate=0.0005,
            weight_decay=0.0,
            max_iter=200,
            enable_wandb=True,
            derivative_ibp_ratio=0,
            derivative_sample_ratio=1,
            positivity_ibp_ratio=0,
            positivity_sample_ratio=0,
            save_best_model=save_lyaloss_path,
            pgd_steps=150,
            buffer_size=131072,
            batch_size=1024,
            epochs=200,
            samples_per_iter=8192,
            l1_reg=0,
            num_samples_per_boundary=1024,
            V_decrease_within_roa=V_decrease_within_roa,
            Vmin_x_boundary_weight=0,
            Vmax_x_boundary_weight=0,
            candidate_roa_states=candidate_roa_states,
            candidate_roa_states_weight=1.0e-05,
            logger=logger,
            always_candidate_roa_regulizer=True,
        )

    torch.save(
        {
            "state_dict": lyapunov_nn.state_dict(),
            "rho": derivative_lyaloss.get_rho(),
        },
        os.path.join(os.getcwd(), "lyapunov_nn.pth"),
    )
    torch.save(
        {
            "state_dict": controller.state_dict(),
            "rho": derivative_lyaloss.get_rho(),
        },
        os.path.join(os.getcwd(), "controller_nn.pth"),
    )

    lyapunov_nn.eval()
    controller.eval()
    # Check with pgd attack.
    derivative_lyaloss_check = lyapunov.LyapunovDerivativeLoss(
        dynamics,
        controller,
        lyapunov_nn,
        box_lo=lower_limit,
        box_up=upper_limit,
        rho_multiplier=rho_multiplier,
        kappa=0.0,
        hard_max=True,
    )
    pgd_verifier_find_counterexamples = False
    for seed in range(50):
        train_utils.set_seed(seed)
        if V_decrease_within_roa:
            x_min_boundary = train_utils.calc_V_extreme_on_boundary_pgd(
                lyapunov_nn,
                lower_limit,
                upper_limit,
                num_samples_per_boundary=1024,
                eps=limit,
                steps=100,
                direction="minimize",
            )
            if derivative_lyaloss.x_boundary is not None:
                derivative_lyaloss_check.x_boundary = torch.cat(
                    (x_min_boundary, derivative_lyaloss.x_boundary), dim=0
                )
        x_check_start = (
            (
                torch.rand((50000, 2), device=device)
                - torch.full((2,), 0.5, device=device)
            )
            * limit
            * 2
        )
        adv_x = train_utils.pgd_attack(
            x_check_start,
            derivative_lyaloss_check,
            eps=limit,
            steps=300,
            lower_boundary=lower_limit,
            upper_boundary=upper_limit,
            direction="minimize",
        ).detach()
        adv_lya = derivative_lyaloss_check(adv_x)
        adv_output = torch.clamp(-adv_lya, min=0.0)
        max_adv_violation = adv_output.max().item()
        msg = f"pgd attack max violation {max_adv_violation}, total violation {adv_output.sum().item()}"
        if max_adv_violation > 0:
            pgd_verifier_find_counterexamples = True
        logger.info(msg)
    logger.info(
        f"PGD verifier finds counter examples? {pgd_verifier_find_counterexamples}"
    )

    rho = derivative_lyaloss.get_rho().item()
    print("rho = ", rho)

    computed_roa_metrics = rmet.compute_roa_area_qmc_sobol(
        lyapunov_nn=lyapunov_nn,
        state_limits=(
            (lower_limit[0], upper_limit[0]),
            (lower_limit[1], upper_limit[1]),
            (lower_limit[2], upper_limit[2]),
            (lower_limit[3], upper_limit[3]),
        ),
        rho=rho,
        device=device,
    )
    rmet.print_roa_metrics(
        computed_roa_metrics,
        title="Computed Region of Attraction for Constructed Lyapunov Function and Given Rho",
    )

    # plots
    lrv.plot_lyapunov_2d(
        lyapunov_nn=lyapunov_nn,
        controller_nn=controller,
        dynamics_system=dynamics,
        state_limits=(
            (lower_limit[0], upper_limit[0]),
            (lower_limit[1], upper_limit[1]),
        ),
        state_names=("x", "theta"),
        rho=rho,
        title="Lyapunov Function for Symbolic Cartpole Inverted Pendulum System, X vs Angle",
        save_html=os.path.join(data_folder, "lyapunov_2d_x_theta.html"),
        show=False,
    )
    lrv.plot_lyapunov_3d_surface(
        lyapunov_nn=lyapunov_nn,
        controller_nn=controller,
        dynamics_system=dynamics,
        state_limits=(
            (lower_limit[0], upper_limit[0]),
            (lower_limit[1], upper_limit[1]),
        ),
        state_names=("x", "theta"),
        rho=rho,
        title="Lyapunov Function for Symbolic Cartpole Inverted Pendulum System, X vs Angle",
        save_html=os.path.join(data_folder, "lyapunov_3d_x_theta.html"),
        show=False,
        show_derivative=True,
    )

    lrv.plot_lyapunov_2d(
        lyapunov_nn=lyapunov_nn,
        controller_nn=controller,
        dynamics_system=dynamics,
        state_limits=(
            (lower_limit[0], upper_limit[0]),
            (lower_limit[2], upper_limit[2]),
        ),
        state_names=("x", "x_dot"),
        rho=rho,
        title="Lyapunov Function for Symbolic Cartpole Inverted Pendulum System, X vs X Derivative",
        save_html=os.path.join(data_folder, "lyapunov_2d_x_x_dot.html"),
        show=False,
    )
    lrv.plot_lyapunov_3d_surface(
        lyapunov_nn=lyapunov_nn,
        controller_nn=controller,
        dynamics_system=dynamics,
        state_limits=(
            (lower_limit[0], upper_limit[0]),
            (lower_limit[2], upper_limit[2]),
        ),
        state_names=("x", "x_dot"),
        rho=rho,
        title="Lyapunov Function for Symbolic Cartpole Inverted Pendulum System, X vs X Derivative",
        save_html=os.path.join(data_folder, "lyapunov_3d_x_x_dot.html"),
        show=False,
        show_derivative=True,
    )

    lrv.plot_lyapunov_2d(
        lyapunov_nn=lyapunov_nn,
        controller_nn=controller,
        dynamics_system=dynamics,
        state_limits=(
            (lower_limit[1], upper_limit[1]),
            (lower_limit[3], upper_limit[3]),
        ),
        state_names=("theta", "theta_dot"),
        rho=rho,
        title="Lyapunov Function for Symbolic Cartpole Inverted Pendulum System, Angle vs Angle Derivative",
        save_html=os.path.join(data_folder, "lyapunov_2d_theta_theta_dot.html"),
        show=False,
    )
    lrv.plot_lyapunov_3d_surface(
        lyapunov_nn=lyapunov_nn,
        controller_nn=controller,
        dynamics_system=dynamics,
        state_limits=(
            (lower_limit[1], upper_limit[1]),
            (lower_limit[3], upper_limit[3]),
        ),
        state_names=("theta", "theta_dot"),
        rho=rho,
        title="Lyapunov Function for Symbolic Cartpole Inverted Pendulum System, Angle vs Angle Derivative",
        save_html=os.path.join(data_folder, "lyapunov_3d_theta_theta_dot.html"),
        show=False,
        show_derivative=True,
    )

    lrv.plot_lyapunov_2d(
        lyapunov_nn=lyapunov_nn,
        controller_nn=controller,
        dynamics_system=dynamics,
        state_limits=(
            (lower_limit[2], upper_limit[2]),
            (lower_limit[3], upper_limit[3]),
        ),
        state_names=("x_dot", "theta_dot"),
        rho=rho,
        title="Lyapunov Function for Symbolic Cartpole Inverted Pendulum System, Derivatives",
        save_html=os.path.join(data_folder, "lyapunov_2d_x_dot_theta_dot.html"),
        show=False,
    )
    lrv.plot_lyapunov_3d_surface(
        lyapunov_nn=lyapunov_nn,
        controller_nn=controller,
        dynamics_system=dynamics,
        state_limits=(
            (lower_limit[2], upper_limit[2]),
            (lower_limit[3], upper_limit[3]),
        ),
        state_names=("x_dot", "theta_dot"),
        rho=rho,
        title="Lyapunov Function for Symbolic Cartpole Inverted Pendulum System, Derivatives",
        save_html=os.path.join(data_folder, "lyapunov_3d_x_dot_theta_dot.html"),
        show=False,
        show_derivative=True,
    )

    pass


if __name__ == "__main__":
    main()
