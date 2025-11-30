import os
from path import Path
import pdb
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
import itertools
import logging

import neural_lyapunov_training.arguments as arguments
import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.models as models
import neural_lyapunov_training.pvtol as pvtol
import neural_lyapunov_training.train_utils as train_utils
import neural_lyapunov_training.output_train_utils as output_train_utils
import neural_lyapunov_training.symbolic_dynamics as sd
import neural_lyapunov_training.symbolic_systems as ss
import neural_lyapunov_training.lyapunov_roa_visualization as lrv
import neural_lyapunov_training.roa_metrics as rmet

device = torch.device("cuda")
dtype = torch.float


def plot_V(V, lower_limit, upper_limit):
    x_ticks = torch.linspace(lower_limit[0], upper_limit[0], 50, device=device)
    y_ticks = torch.linspace(lower_limit[1], upper_limit[1], 50, device=device)
    grid_x, grid_y = torch.meshgrid(x_ticks, y_ticks)
    with torch.no_grad():
        V_val = V(torch.stack((grid_x, grid_y), dim=2)).squeeze(2)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(grid_x.numpy(), grid_y.numpy(), V_val.numpy())
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\dot{\theta}$")
    ax.set_zlabel("V")
    return fig, ax


def plot_V_heatmap(
    V,
    lower_limit,
    upper_limit,
    nx,
    labels,
    x_boundary=None,
    plot_idx=[0, 2],
    mode="boundary",
    V_lqr=None,
):
    x_ticks = torch.linspace(
        lower_limit[plot_idx[0]], upper_limit[plot_idx[0]], 50, device=device
    )
    y_ticks = torch.linspace(
        lower_limit[plot_idx[1]], upper_limit[plot_idx[1]], 50, device=device
    )
    grid_x, grid_y = torch.meshgrid(x_ticks, y_ticks)
    if mode == "boundary":
        X = torch.ones(2500, nx, device=device) * x_boundary
    elif isinstance(mode, float):
        X = torch.ones(2500, nx, device=device) * upper_limit * mode
    X[:, plot_idx[0]] = grid_x.flatten()
    X[:, plot_idx[1]] = grid_y.flatten()

    with torch.no_grad():
        V_val = V(X)

    V_val = V_val.cpu().reshape(50, 50)
    grid_x = grid_x.cpu()
    grid_y = grid_y.cpu()

    # Compute min V(x) on the boundary
    x_pgd_boundary_min = train_utils.calc_V_extreme_on_boundary_pgd(
        V,
        lower_limit,
        upper_limit,
        num_samples_per_boundary=1000,
        eps=(upper_limit - lower_limit) / 2,
        steps=100,
        direction="minimize",
    )
    rho_roa = torch.min(V(x_pgd_boundary_min)).item()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.pcolor(grid_x, grid_y, V_val)
    ax.contour(grid_x, grid_y, V_val, [rho_roa], colors="red")
    if V_lqr is not None:
        V_lqr_val = V_lqr(X).reshape(50, 50).cpu()
        x_pgd_boundary_min = train_utils.calc_V_extreme_on_boundary_pgd(
            V_lqr,
            lower_limit,
            upper_limit,
            num_samples_per_boundary=1000,
            eps=(upper_limit - lower_limit) / 2,
            steps=100,
            direction="minimize",
        )
        rho_lqr_roa = torch.min(V_lqr(x_pgd_boundary_min)).item()
        ax.contour(grid_x, grid_y, V_lqr_val, [rho_lqr_roa], colors="cyan")
    lower_limit = lower_limit.cpu()
    upper_limit = upper_limit.cpu()
    ax.set_xlim(lower_limit[plot_idx[0]], upper_limit[plot_idx[0]])
    ax.set_ylim(lower_limit[plot_idx[1]], upper_limit[plot_idx[1]])
    cbar = fig.colorbar(im, ax=ax)
    return fig, ax, cbar


if __name__ == "__main__":

    train_utils.set_seed(42)
    logger = logging.getLogger(__name__)
    dt = 0.05
    pvtol_continuous = ss.PVTOL()
    dynamics = sd.GenericDiscreteTimeSystem(
        pvtol_continuous,
        dt=dt,
        integration_method=sd.IntegrationMethod.RK4,
        position_integration=sd.IntegrationMethod.RK4,
    )

    grid_size = torch.tensor([4, 4, 6, 5, 5, 6], device=device)

    Q = np.diag(np.array([1, 1, 1, 10, 10, 10.0]))
    R = np.diag(np.array([10, 10.0]))
    K, S = pvtol_continuous.lqr_control(Q, R)
    K_torch = torch.from_numpy(K).type(dtype).to(device)
    S_torch = torch.from_numpy(S).type(dtype).to(device)
    max_u = torch.tensor([1, 1.0], device=device) * 39.2

    controller = controllers.NeuralNetworkController(
        nlayer=4,
        in_dim=6,
        out_dim=2,
        hidden_dim=8,
        clip_output="clamp",
        u_lo=torch.tensor([0, 0.0], dtype=dtype, device=device),
        u_up=max_u,
        x_equilibrium=(dynamics.x_equilibrium).to(device).to(dtype),
        u_equilibrium=(dynamics.u_equilibrium).to(device).to(dtype),
    )
    controller.train()

    absolute_output = True
    # lyapunov_nn = lyapunov.NeuralNetworkLyapunov(
    #     goal_state=torch.zeros(6, dtype=dtype).to(device),
    #     hidden_widths=[16, 16, 8],
    #     x_dim=6,
    #     R_rows=3,
    #     absolute_output=absolute_output,
    #     eps=0.01,
    #     activation=nn.LeakyReLU,
    #     V_psd_form="L1",
    # )
    R = torch.linalg.cholesky(S_torch)
    lyapunov_nn = lyapunov.NeuralNetworkQuadraticLyapunov(
        goal_state=torch.zeros(6, dtype=dtype).to(device),
        x_dim=6,
        R_rows=6,
        eps=0.01,
        R=R,
    )
    lyapunov_nn.train()

    dynamics.to(device).to(dtype)
    controller.to(device).to(dtype)
    lyapunov_nn.to(device).to(dtype)

    # output_train_utils.approximate_controller(V, lyapunov_nn, 6, limit, 0, 0, "examples/data/pvtol/lyapunov_{}.pth".format(lyapunov_hidden_widths), max_iter=500, lr=0.05, l1_reg=0.01)

    # controller.load_state_dict(
    #     torch.load("examples/data/pvtol/controller_lqr_[8, 8].pth")
    # )
    # controller.eval()

    kappa = 0.01
    hard_max = True
    derivative_lyaloss = lyapunov.LyapunovDerivativeLoss(
        dynamics,
        controller,
        lyapunov_nn,
        kappa=kappa,
        hard_max=hard_max,
        box_lo=0,
        box_up=0,
        rho_multiplier=2.25,
    )

    if absolute_output:
        positivity_lyaloss = None
    else:
        positivity_lyaloss = lyapunov.LyapunovPositivityLoss(
            lyapunov_nn, 0.01 * torch.eye(2, dtype=dtype, device=device)
        )

    candidate_scale = 2.0
    candidate_roa_states_weight = 1.0e-05
    suffix = "symbolic"
    data_folder = f"./output/benezerg/pvtol_state/{datetime.now():%Y-%m-%d}/{datetime.now():%H-%M-%S}_{suffix}"
    os.makedirs(data_folder, exist_ok=True)
    save_lyaloss = True
    V_decrease_within_roa = True
    save_lyaloss_path = None
    save_name = (
        f"lyaloss_{kappa}kappa_{candidate_scale}_{candidate_roa_states_weight}.pth"
    )
    if save_lyaloss:
        save_lyaloss_path = f"{data_folder}/{save_name}"

    wandb.init(
        project="CS-7268-Group-Project",
        entity="GB-Northeastern-Projects",
        name=f"{datetime.now():%Y-%m-%d}_{datetime.now():%H-%M-%S}_pvtol_state",
    )

    permute_array = [[-1, 1]] * pvtol_continuous.nx
    permute_array_torch = torch.tensor(
        list(itertools.product(*permute_array)), device=device
    )

    limit_base = torch.tensor(
        [0.3, 0.3, 0.3, 1.0, 1.0, 1.0], dtype=dtype, device=device
    )
    limit_scale = [0.25, 0.5, 1.0, 2.0]
    for scale in limit_scale:
        limit = limit_scale * limit_base
        lower_limit = -1 * limit
        upper_limit = 1 * limit

        
        candidate_roa_states = permute_array_torch * upper_limit
        x_min_boundary = train_utils.calc_V_extreme_on_boundary_pgd(
            lyapunov_nn,
            lower_limit,
            upper_limit,
            num_samples_per_boundary=1000,
            eps=limit,
            steps=100,
            direction="minimize",
        )
        rho = lyapunov_nn(x_min_boundary).min().item()
        # Sample slightly outside the current ROA
        V_candidate = lyapunov_nn(candidate_roa_states).clone().detach()
        candidate_roa_states = (
            candidate_roa_states / torch.sqrt(V_candidate / rho) * candidate_scale
        )
        candidate_roa_states = torch.clamp(
            candidate_roa_states, min=lower_limit, max=upper_limit
        )
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
            hard_max=hard_max,
            lr_scheduler=False,
            logger=logger,
        )

    controller.eval()
    lyapunov_nn.eval()
    derivative_lyaloss_check = lyapunov.LyapunovDerivativeLoss(
        dynamics,
        controller,
        lyapunov_nn,
        kappa=0.0,
        hard_max=hard_max,
        box_lo=0,
        box_up=0,
        rho_multiplier=2.25,
    )
    fig, ax = plt.subplots(1, 2)
    # Check with pgd attack.
    pgd_verifier_find_counterexamples = False
    for seed in range(200, 300):
        train_utils.set_seed(seed)
        if V_decrease_within_roa:
            x_min_boundary = train_utils.calc_V_extreme_on_boundary_pgd(
                lyapunov_nn,
                lower_limit,
                upper_limit,
                num_samples_per_boundary=1000,
                eps=limit,
                steps=100,
                direction="minimize",
            )
            derivative_lyaloss_check.x_boundary = x_min_boundary
        x_check_start = (
            (torch.rand((50000, pvtol_continuous.nx), dtype=dtype, device=device) - 0.5)
            * limit
            * 2
        )
        adv_x = train_utils.pgd_attack(
            x_check_start,
            derivative_lyaloss_check,
            eps=limit,
            steps=100,
            lower_boundary=lower_limit,
            upper_boundary=upper_limit,
            direction="minimize",
        ).detach()
        adv_lya = derivative_lyaloss_check(adv_x)
        adv_output = torch.clamp(-adv_lya, min=0.0)
        max_adv_violation = adv_output.max().item()
        msg = f"pgd attack max violation {max_adv_violation}, total violation {adv_output.sum().item()}"
        print(msg)
        x_adv = adv_x[(adv_lya < 0).squeeze()]
        print(adv_lya.min().item())
        if max_adv_violation > 0:
            pgd_verifier_find_counterexamples = True
        logger.info(msg)
    logger.info(
        f"PGD verifier finds counter examples? {pgd_verifier_find_counterexamples}"
    )

    plt.clf()
    rho = lyapunov_nn(x_min_boundary).min().item()
    x0 = (
        (torch.rand((40, pvtol_continuous.nx), dtype=dtype, device=device) - 0.5)
        * 2
        * limit
    )
    x_traj, V_traj = models.simulate(derivative_lyaloss, 500, x0)
    V_traj = torch.stack(V_traj[100:]).cpu().detach().squeeze().numpy()
    V_traj = V_traj[:, V_traj[0, :] <= rho]
    plt.plot(dt * np.arange(400), V_traj)
    plt.savefig(f"{data_folder}/V_traj_{kappa}_{candidate_scale}.png")

    print("rho = ", rho)
    x_boundary = x_min_boundary[torch.argmin(lyapunov_nn(x_min_boundary))]
    print("Boundary state ratio = ", x_boundary / limit)
    labels = [r"$x$", r"$y$", r"$\theta$", r"$\dot x$", r"$\dot y$", r"$\dot \theta$"]
    for plot_idx in [[0, 1], [0, 2], [3, 4], [4, 5]]:
        fig2, axis2, cbar2 = plot_V_heatmap(
            lyapunov_nn,
            lower_limit,
            upper_limit,
            6,
            labels,
            x_boundary,
            plot_idx=plot_idx,
            mode=0.0,
        )
        # plt.xticks([-0.75, -0.75/2, 0, 0.75/2, 0.75], [r"$-0.75$", r"$-0.375$", r"$0$", r"$0.375$", r"$0.75$"], fontsize=15)
        # plt.yticks([-0.75, -0.75/2, 0, 0.75/2, 0.75], [r"$-0.75$", r"$-0.375$", r"$0$", r"$0.375$", r"$0.75$"], fontsize=15)

        # plt.xticks([-4, -4/2, 0, 4/2, 4], [r"$-4$", r"$-2$", r"$0$", r"$2$", r"$4$"], fontsize=15)
        # plt.yticks([-4, -4/2, 0, 4/2, 4], [r"$-4$", r"$-2$", r"$0$", r"$2$", r"$4$"], fontsize=15)

        # plt.xticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2], [r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"], fontsize=15)
        # plt.yticks([-3, -3/2, 0, 3/2, 3], [r"$-3$", r"$-1.5$", r"$0$", r"$1.5$", r"$3$"], fontsize=15)

        # plt.title(f"rho = {rho}")
        plt.savefig(
            f"{data_folder}/V_{kappa}_{candidate_scale}_roa_{str(plot_idx)}.png"
        )

    quadrotor_state_limits = tuple(
        (lower_limit[i], upper_limit[i]) for i in range(len(lower_limit))
    )

    computed_roa_metrics = rmet.compute_roa_area_qmc_sobol(
        lyapunov_nn=lyapunov_nn,
        state_limits=quadrotor_state_limits,
        rho=rho,
        device=device,
    )

    rmet.print_roa_metrics(
        computed_roa_metrics,
        title="Computed Region of Attraction for Constructed Lyapunov Function and Given Rho",
    )

    labels = [r"$x$", r"$y$", r"$\theta$", r"$\dot x$", r"$\dot y$", r"$\dot \theta$"]

    plot_indices = (
        (2, 5),  # theta vs dot_theta
        (0, 3),  # x vs dot_x
        (1, 4),  # y vs dot_y
        (0, 2),  # x vs theta
        (1, 2),  # y vs theta
        (0, 1),  # x vs y
        (3, 4),  # dot_x vs dot_y
        (4, 5),  # dot_y vs dot_theta
        (3, 5),  # dot_x vs dot_theta
    )

    suffixes = (
        "theta_v_dot_theta",
        "x_v_dot_x",
        "y_v_dot_y",
        "x_v_theta",
        "y_v_theta",
        "x_v_y",
        "dot_x_v_dot_y",
        "dot_y_v_dot_theta",
        "dot_x_v_dot_theta",
    )

    titles = (
        "Angle and Angle Derivative",
        "X and X Derivative",
        "Y and Y Derivative",
        "X and Angle",
        "Y and Angle",
        "X and Y",
        "X Derivative and Y Derivative",
        "Y Derivative and Angle Derivative",
        "X Derivative and Angle Derivative",
    )

    for idx_index, plot_idxes in enumerate(plot_indices):

        name_tuple = (labels[plot_idxes[0]], labels[plot_idxes[1]])

        state_lims = (
            quadrotor_state_limits[plot_idxes[0]],
            quadrotor_state_limits[plot_idxes[1]],
        )

        lrv.plot_lyapunov_2d(
            lyapunov_nn=lyapunov_nn,
            controller_nn=controller,
            dynamics_system=dynamics,
            state_limits=state_lims,
            state_names=name_tuple,
            state_indices=plot_idxes,
            rho=rho,
            title=f"2D Lyapunov Function, 2D PVTOL, State Feedback, {titles[idx_index]}",
            save_html=os.path.join(
                data_folder, f"lyapunov_2d_{suffixes[idx_index]}.html"
            ),
            show=False,
        )

        lrv.plot_lyapunov_3d_surface(
            lyapunov_nn=lyapunov_nn,
            controller_nn=controller,
            dynamics_system=dynamics,
            state_limits=state_lims,
            state_names=name_tuple,
            state_indices=plot_idxes,
            rho=rho,
            nx=6,
            title=f"3D Lyapunov Function, 2D PVTOL, State Feedback, {titles[idx_index]}",
            save_html=os.path.join(
                data_folder, f"lyapunov_3d_{suffixes[idx_index]}.html"
            ),
            show=False,
            show_derivative=True,
        )
