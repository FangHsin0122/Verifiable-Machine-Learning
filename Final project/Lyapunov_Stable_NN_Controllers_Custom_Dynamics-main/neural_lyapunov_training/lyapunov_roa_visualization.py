"""
Lyapunov Function and Region of Attraction Visualization

This module provides comprehensive visualization tools for:
- Lyapunov function value fields
- Regions of Attraction (ROA)
- Closed-loop trajectories with neural controllers
- Lyapunov derivative fields
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Callable
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Convert state_limits to CPU numpy if they're tensors
def to_float(val):
    """Convert tensor or array to float"""
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().item()
    elif isinstance(val, (np.ndarray, np.generic)):
        return float(val)
    else:
        return float(val)

def plot_lyapunov_2d(
    lyapunov_nn,
    controller_nn,
    dynamics_system,
    state_limits: Tuple[Tuple[float, float], Tuple[float, float]],
    state_indices: Tuple[int, int] = (0, 1),
    state_names: Optional[Tuple[str, str]] = None,
    rho: Optional[float] = None,
    grid_resolution: int = 100,
    observer_nn = None,
    trajectories: Optional[List[torch.Tensor]] = None,
    title: Optional[str] = None,
    save_html: Optional[str] = None,
    show: bool = True,
    colorscale: str = 'Viridis'
):
    """
    Plot Lyapunov function value field and Region of Attraction in 2D
    
    Args:
        lyapunov_nn: Neural network Lyapunov function V(x)
        controller_nn: Neural network controller u = π(x) or π(x̂)
        dynamics_system: Dynamical system (GenericDiscreteTimeSystem)
        state_limits: ((x_min, x_max), (y_min, y_max)) for the two plotted states
        state_indices: Which two state dimensions to plot
        state_names: Names for the axes
        rho: ROA threshold (if None, compute from boundary)
        grid_resolution: Number of grid points per dimension
        observer_nn: Optional observer for output feedback (x̂ = obs(y))
        trajectories: Optional list of trajectories to overlay
        title: Plot title
        save_html: Filename to save interactive HTML
        show: Whether to display the plot
        colorscale: Plotly colorscale name
    
    Returns:
        Plotly figure object
    """
    
    state_limits = tuple(
        (to_float(lim[0]), to_float(lim[1])) for lim in state_limits
    )
    
    device = next(lyapunov_nn.parameters()).device if hasattr(lyapunov_nn, 'parameters') else 'cpu'
    
    # Create grid
    idx0, idx1 = state_indices
    x0_range = np.linspace(state_limits[0][0], state_limits[0][1], grid_resolution)
    x1_range = np.linspace(state_limits[1][0], state_limits[1][1], grid_resolution)
    X0, X1 = np.meshgrid(x0_range, x1_range)
    
    # Initialize state grid
    nx = dynamics_system.nx
    states_grid = torch.zeros((grid_resolution * grid_resolution, nx), device=device)
    
    # Fill in the two dimensions we're plotting
    states_grid[:, idx0] = torch.tensor(X0.flatten(), dtype=torch.float32, device=device)
    states_grid[:, idx1] = torch.tensor(X1.flatten(), dtype=torch.float32, device=device)
    
    # Other dimensions set to equilibrium or zero
    x_eq = dynamics_system.x_equilibrium.to(device)
    for i in range(nx):
        if i not in state_indices:
            states_grid[:, i] = x_eq[i]
    
    # Evaluate Lyapunov function
    with torch.no_grad():
        V_values = lyapunov_nn(states_grid).squeeze()
        V_grid = V_values.reshape(grid_resolution, grid_resolution).cpu().numpy()
    
    # Compute Lyapunov derivative (V̇)
    with torch.no_grad():
        if observer_nn is not None:
            # Output feedback: y = h(x), x̂ = obs(y), u = π(x̂)
            y = dynamics_system.continuous_time_system.h(states_grid)
            x_hat = observer_nn(y)
            u = controller_nn(x_hat)
        else:
            # State feedback: u = π(x)
            u = controller_nn(states_grid)
        
        # Compute x_next and Lyapunov derivative
        x_next = dynamics_system(states_grid, u)
        V_next = lyapunov_nn(x_next).squeeze()
        V_dot = V_next - V_values  # Discrete-time Lyapunov derivative
        V_dot_grid = V_dot.reshape(grid_resolution, grid_resolution).cpu().numpy()
    
    # Determine ROA threshold
    if rho is None:
        # Compute rho from boundary values
        boundary_mask = (
            (states_grid[:, idx0] == state_limits[0][0]) |
            (states_grid[:, idx0] == state_limits[0][1]) |
            (states_grid[:, idx1] == state_limits[1][0]) |
            (states_grid[:, idx1] == state_limits[1][1])
        )
        if boundary_mask.any():
            rho = V_values[boundary_mask].min().item()
        else:
            rho = V_values.max().item() * 0.8
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Lyapunov Function V(x)', 'Lyapunov Derivative ΔV(x)'),
        specs=[[{'type': 'contour'}, {'type': 'contour'}]]
    )
    
    # Plot 1: Lyapunov function
    fig.add_trace(
        go.Contour(
            x=x0_range,
            y=x1_range,
            z=V_grid,
            colorscale=colorscale,
            contours=dict(
                start=0,
                end=V_grid.max(),
                size=V_grid.max() / 20,
            ),
            colorbar=dict(title="V(x)", x=0.45),
            hovertemplate='%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<br>V: %{z:.3f}<extra></extra>',
        ),
        row=1, col=1
    )
    
    # Add ROA boundary (V(x) = rho)
    fig.add_trace(
        go.Contour(
            x=x0_range,
            y=x1_range,
            z=V_grid,
            contours=dict(
                start=rho,
                end=rho,
                size=1,
                coloring='none'
            ),
            line=dict(color='red', width=4),
            showscale=False,
            name=f'ROA (ρ={rho:.3f})',
            hovertemplate='ROA Boundary<extra></extra>',
        ),
        row=1, col=1
    )
    
    # Plot 2: Lyapunov derivative
    fig.add_trace(
        go.Contour(
            x=x0_range,
            y=x1_range,
            z=V_dot_grid,
            colorscale='RdBu_r',  # Red for positive, blue for negative
            contours=dict(
                start=V_dot_grid.min(),
                end=V_dot_grid.max(),
                size=(V_dot_grid.max() - V_dot_grid.min()) / 20,
            ),
            colorbar=dict(title="ΔV(x)", x=1.05),
            hovertemplate='%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<br>ΔV: %{z:.3f}<extra></extra>',
        ),
        row=1, col=2
    )
    
    # Add zero contour for V_dot (should be negative everywhere in ROA)
    fig.add_trace(
        go.Contour(
            x=x0_range,
            y=x1_range,
            z=V_dot_grid,
            contours=dict(
                start=0,
                end=0,
                size=1,
                coloring='none'
            ),
            line=dict(color='black', width=3, dash='dash'),
            showscale=False,
            name='ΔV=0',
            hovertemplate='ΔV=0 Contour<extra></extra>',
        ),
        row=1, col=2
    )
    
    # Add equilibrium point to both plots
    x_eq_np = x_eq.cpu().numpy()
    for col in [1, 2]:
        fig.add_trace(
            go.Scatter(
                x=[x_eq_np[idx0]],
                y=[x_eq_np[idx1]],
                mode='markers',
                marker=dict(size=12, color='lime', symbol='star', line=dict(width=2, color='black')),
                name='Equilibrium',
                showlegend=(col == 1),
                hovertemplate='Equilibrium<extra></extra>',
            ),
            row=1, col=col
        )
    
    # Overlay trajectories if provided
    if trajectories is not None:
        colors = ['white', 'yellow', 'cyan', 'magenta', 'orange']
        for i, traj in enumerate(trajectories):
            traj_np = traj.detach().cpu().numpy()
            color = colors[i % len(colors)]
            
            for col in [1, 2]:
                # Trajectory line
                fig.add_trace(
                    go.Scatter(
                        x=traj_np[:, idx0],
                        y=traj_np[:, idx1],
                        mode='lines',
                        line=dict(color=color, width=2),
                        name=f'Trajectory {i+1}',
                        showlegend=(col == 1),
                        hovertemplate=f'Traj {i+1}<extra></extra>',
                    ),
                    row=1, col=col
                )
                
                # Start point
                fig.add_trace(
                    go.Scatter(
                        x=[traj_np[0, idx0]],
                        y=[traj_np[0, idx1]],
                        mode='markers',
                        marker=dict(size=10, color=color, symbol='circle'),
                        showlegend=False,
                        hovertemplate='Start<extra></extra>',
                    ),
                    row=1, col=col
                )
    
    # Update axes
    if state_names is None:
        state_names = (f'x{idx0}', f'x{idx1}')
    
    fig.update_xaxes(title_text=state_names[0], row=1, col=1)
    fig.update_yaxes(title_text=state_names[1], row=1, col=1)
    fig.update_xaxes(title_text=state_names[0], row=1, col=2)
    fig.update_yaxes(title_text=state_names[1], row=1, col=2)
    
    # Update layout
    if title is None:
        title = "Lyapunov Function and Region of Attraction"
    
    fig.update_layout(
        title=title,
        height=650,
        width=1600,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=1.15,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=11)
        ),
        margin=dict(l=60, r=250, t=90, b=60)
    )

    # Add more horizontal spacing between subplots
    fig.update_xaxes(domain=[0.0, 0.42], row=1, col=1)
    fig.update_xaxes(domain=[0.59, 0.99], row=1, col=2)
    
    if save_html:
        fig.write_html(save_html)
        print(f"Lyapunov visualization saved to {save_html}")
    
    if show:
        fig.show()
    
    return fig
    
def plot_lyapunov_3d_surface(
    lyapunov_nn,
    state_limits: Tuple[Tuple[float, float], Tuple[float, float]],
    controller_nn=None,
    dynamics_system=None,
    observer_nn=None,
    state_indices: Tuple[int, int] = (0, 1),
    state_names: Optional[Tuple[str, str]] = None,
    rho: Optional[float] = None,
    grid_resolution: int = 80,
    nx: int = 2,
    x_equilibrium: Optional[torch.Tensor] = None,
    title: Optional[str] = None,
    save_html: Optional[str] = None,
    show: bool = True,
    colorscale: str = "Viridis",
    show_derivative: bool = False
):
    """
    Plot Lyapunov function as a 3D surface, optionally with derivative surface

    Args:
        lyapunov_nn: Neural network Lyapunov function
        state_limits: Limits for the two plotted states
        controller_nn: Optional controller (required if show_derivative=True)
        dynamics_system: Optional dynamics (required if show_derivative=True)
        observer_nn: Optional observer for output feedback
        state_indices: Which two states to plot
        state_names: Names for axes
        rho: ROA threshold to highlight
        grid_resolution: Grid density
        nx: Total state dimension
        x_equilibrium: Equilibrium point
        title: Plot title
        save_html: Filename to save
        show: Whether to display
        colorscale: Plotly colorscale
        show_derivative: If True, create side-by-side plot with V and ΔV

    Returns:
        Plotly figure
    """
    
    state_limits = tuple(
        (to_float(lim[0]), to_float(lim[1])) for lim in state_limits
    )
    
    device = next(lyapunov_nn.parameters()).device if hasattr(lyapunov_nn, 'parameters') else 'cpu'
    
    # Create grid
    idx0, idx1 = state_indices
    x0_range = np.linspace(state_limits[0][0], state_limits[0][1], grid_resolution)
    x1_range = np.linspace(state_limits[1][0], state_limits[1][1], grid_resolution)
    X0, X1 = np.meshgrid(x0_range, x1_range)
    
    # Initialize state grid
    states_grid = torch.zeros((grid_resolution * grid_resolution, nx), device=device)
    states_grid[:, idx0] = torch.tensor(X0.flatten(), dtype=torch.float32, device=device)
    states_grid[:, idx1] = torch.tensor(X1.flatten(), dtype=torch.float32, device=device)
    
    # Set other dimensions to equilibrium
    if x_equilibrium is not None:
        for i in range(nx):
            if i not in state_indices:
                states_grid[:, i] = x_equilibrium[i]
    
    # Evaluate Lyapunov function
    with torch.no_grad():
        V_values = lyapunov_nn(states_grid).squeeze()
        V_grid = V_values.reshape(grid_resolution, grid_resolution).cpu().numpy()
    
    # Optionally compute Lyapunov derivative
    if show_derivative:
        if controller_nn is None or dynamics_system is None:
            raise ValueError("controller_nn and dynamics_system required when show_derivative=True")
        
        with torch.no_grad():
            if observer_nn is not None:
                y = dynamics_system.continuous_time_system.h(states_grid)
                x_hat = observer_nn(y)
                u = controller_nn(x_hat)
            else:
                u = controller_nn(states_grid)
            
            x_next = dynamics_system(states_grid, u)
            V_next = lyapunov_nn(x_next).squeeze()
            V_dot = V_next - V_values
            V_dot_grid = V_dot.reshape(grid_resolution, grid_resolution).cpu().numpy()
    
    # Create figure - dual view if showing derivative
    if show_derivative:
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=('Lyapunov Function V(x)', 'Lyapunov Derivative ΔV(x)')
        )
        
        # Left: V(x) surface
        fig.add_trace(
            go.Surface(
                x=X0, y=X1, z=V_grid,
                colorscale=colorscale,
                name='V(x)',
                colorbar=dict(title="V(x)", x=0.42, len=0.85, thickness=20),
                hovertemplate='%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<br>V: %{z:.3f}<extra></extra>',
            ),
            row=1, col=1
        )
        
        # Right: ΔV(x) surface
        fig.add_trace(
            go.Surface(
                x=X0, y=X1, z=V_dot_grid,
                colorscale='RdBu_r',
                name='ΔV(x)',
                colorbar=dict(title="ΔV(x)", x=1.02, len=0.85, thickness=20),
                hovertemplate='%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<br>ΔV: %{z:.3f}<extra></extra>',
            ),
            row=1, col=2
        )
        
        # Add ROA plane to V(x) plot
        if rho is not None:
            rho_plane = np.full_like(V_grid, rho)
            fig.add_trace(
                go.Surface(
                    x=X0, y=X1, z=rho_plane,
                    opacity=0.4,
                    colorscale=[[0, 'red'], [1, 'red']],
                    showscale=False,
                    name=f'ROA (ρ={rho:.3f})',
                    hovertemplate=f'ROA boundary<extra></extra>',
                ),
                row=1, col=1
            )
        
        # Add zero plane to ΔV(x) plot
        zero_plane = np.zeros_like(V_dot_grid)
        fig.add_trace(
            go.Surface(
                x=X0, y=X1, z=zero_plane,
                opacity=0.4,
                colorscale=[[0, 'black'], [1, 'black']],
                showscale=False,
                name='ΔV=0',
                hovertemplate='ΔV=0 plane<extra></extra>',
            ),
            row=1, col=2
        )
        
        # Add equilibrium to both plots
        if x_equilibrium is not None:
            x_eq_np = x_equilibrium.cpu().numpy()
            V_eq = lyapunov_nn(x_equilibrium.unsqueeze(0).to(device)).item()
            
            for col in [1, 2]:
                z_val = V_eq if col == 1 else 0.0
                fig.add_trace(
                    go.Scatter3d(
                        x=[x_eq_np[idx0]], 
                        y=[x_eq_np[idx1]], 
                        z=[z_val],
                        mode='markers',
                        marker=dict(size=8, color='lime', symbol='diamond', 
                                   line=dict(width=2, color='black')),
                        showlegend=(col == 1),
                        name='Equilibrium',
                        hovertemplate='Equilibrium<extra></extra>',
                    ),
                    row=1, col=col
                )
        
        # Update scene settings for both plots
        if state_names is None:
            state_names = (f'x{idx0}', f'x{idx1}')
        
        for col in [1, 2]:
            fig.update_scenes(
                xaxis_title=state_names[0],
                yaxis_title=state_names[1],
                zaxis_title='V(x)' if col == 1 else 'ΔV(x)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
                row=1, col=col
            )
        
        if title is None:
            title = "Lyapunov Function and Derivative (3D)"
        
        fig.update_layout(
            title=title,
            height=700,
            width=1500,
            showlegend=True,
            margin=dict(l=50, r=50, t=80, b=50)
        )
    else:
        # Single plot - just V(x)
        fig = go.Figure()
        
        # Lyapunov surface
        fig.add_trace(
            go.Surface(
                x=X0, y=X1, z=V_grid,
                colorscale=colorscale,
                name='V(x)',
                colorbar=dict(title="V(x)", thickness=20),
                hovertemplate='%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<br>V: %{z:.3f}<extra></extra>',
            )
        )
        
        # Add ROA plane at V = rho
        if rho is not None:
            rho_plane = np.full_like(V_grid, rho)
            fig.add_trace(
                go.Surface(
                    x=X0, y=X1, z=rho_plane,
                    opacity=0.5,
                    colorscale=[[0, 'red'], [1, 'red']],
                    showscale=False,
                    name=f'ROA (ρ={rho:.3f})',
                    hovertemplate=f'ROA threshold: {rho:.3f}<extra></extra>',
                )
            )
        
        # Add equilibrium point
        if x_equilibrium is not None:
            x_eq_np = x_equilibrium.cpu().numpy()
            V_eq = lyapunov_nn(x_equilibrium.unsqueeze(0).to(device)).item()
            fig.add_trace(
                go.Scatter3d(
                    x=[x_eq_np[idx0]], 
                    y=[x_eq_np[idx1]], 
                    z=[V_eq],
                    mode='markers',
                    marker=dict(size=8, color='lime', symbol='diamond',
                               line=dict(width=2, color='black')),
                    name='Equilibrium',
                    hovertemplate='Equilibrium<extra></extra>',
                )
            )
        
        # Set axis labels
        if state_names is None:
            state_names = (f'x{idx0}', f'x{idx1}')
        
        if title is None:
            title = "Lyapunov Function V(x)"
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=state_names[0],
                yaxis_title=state_names[1],
                zaxis_title='V(x)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            height=700,
            width=900,
            showlegend=True,
            margin=dict(l=50, r=50, t=80, b=50)
        )
    
    fig.update_layout(
        legend=dict(
            x=1.14,  # Moved further right
            y=0.5,
            xanchor='left',
            yanchor='middle',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=11)
        )
    )

    if save_html:
        fig.write_html(save_html)
        print(f"3D Lyapunov surface saved to {save_html}")
    
    if show:
        fig.show()
    
    return fig