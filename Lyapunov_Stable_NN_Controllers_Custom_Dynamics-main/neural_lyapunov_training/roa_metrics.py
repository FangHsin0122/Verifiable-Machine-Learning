"""
Region of Attraction (ROA) Metrics

Quantitative measurement of ROA area/volume using:
- Monte Carlo (random sampling)
- Quasi-Monte Carlo (low-discrepancy sequences: Sobol, Halton)
- Grid-based methods
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Callable
from dataclasses import dataclass


@dataclass
class ROAMetrics:
    """Container for ROA metrics"""

    rho: float  # ROA threshold (V(x) <= rho)
    area_roa: float  # Area/volume of ROA
    area_domain: float  # Area/volume of full domain
    coverage_ratio: float  # ROA area / domain area
    num_samples_in_roa: int  # Number of samples inside ROA
    num_samples_total: int  # Total number of samples
    domain_bounds: Tuple  # Domain boundaries used
    method: str  # Computation method ('monte_carlo', 'qmc_sobol', 'qmc_halton', 'grid')
    grid_resolution: Optional[int] = None  # For grid method
    discrepancy: Optional[float] = None  # For QMC methods


# Convert state_limits to CPU numpy if they're tensors
def to_float(val):
    """Convert tensor or array to float"""
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().item()
    elif isinstance(val, (np.ndarray, np.generic)):
        return float(val)
    else:
        return float(val)


def round_to_power_of_2(n: int, direction: str = "nearest") -> int:
    """
    Round number to nearest power of 2

    Args:
        n: Input number
        direction: 'up', 'down', or 'nearest'

    Returns:
        Closest power of 2
    """
    if n <= 0:
        return 1

    log2_n = np.log2(n)

    if direction == "up":
        return int(2 ** np.ceil(log2_n))
    elif direction == "down":
        return int(2 ** np.floor(log2_n))
    else:  # nearest
        return int(2 ** np.round(log2_n))


def generate_sobol_samples(
    n_samples: int,
    n_dims: int,
    bounds: Tuple[Tuple[float, float], ...],
    device: str = "cpu",
    round_to_pow2: bool = True,
) -> torch.Tensor:
    """
    Generate samples using Sobol low-discrepancy sequence

    Sobol sequences have better uniformity than random sampling,
    leading to faster convergence (O(1/N) vs O(1/sqrt(N)))

    Args:
        n_samples: Number of samples (will be rounded to power of 2 if round_to_pow2=True)
        n_dims: Number of dimensions
        bounds: (min, max) for each dimension
        device: Computing device
        round_to_pow2: Whether to round n_samples to nearest power of 2

    Returns:
        Tensor of samples (actual_n_samples, n_dims)
        Note: actual_n_samples may differ from n_samples if rounded
    """
    from scipy.stats import qmc

    # Round to power of 2 for optimal Sobol properties
    if round_to_pow2:
        n_samples_actual = round_to_power_of_2(n_samples, direction="nearest")
        if n_samples_actual != n_samples:
            import warnings

            warnings.warn(
                f"Rounded n_samples from {n_samples} to {n_samples_actual} (power of 2) "
                f"for optimal Sobol sequence properties.",
                UserWarning,
            )
    else:
        n_samples_actual = n_samples

    # Generate samples in unit hypercube [0,1]^d
    sobol_engine = qmc.Sobol(d=n_dims, scramble=True, seed=42)
    samples_unit = sobol_engine.random(n_samples_actual)

    # Scale to actual bounds
    samples = np.zeros_like(samples_unit)
    for i in range(n_dims):
        low, high = bounds[i]
        samples[:, i] = samples_unit[:, i] * (high - low) + low

    return torch.tensor(samples, dtype=torch.float32, device=device)


def generate_halton_samples(
    n_samples: int,
    n_dims: int,
    bounds: Tuple[Tuple[float, float], ...],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate samples using Halton low-discrepancy sequence

    Halton sequences are another type of low-discrepancy sequence,
    often better for lower dimensions (d <= 10)

    Args:
        n_samples: Number of samples
        n_dims: Number of dimensions
        bounds: (min, max) for each dimension
        device: Computing device

    Returns:
        Tensor of samples (n_samples, n_dims)
    """
    from scipy.stats import qmc

    # Generate samples in unit hypercube
    halton_engine = qmc.Halton(d=n_dims, scramble=True, seed=42)
    samples_unit = halton_engine.random(n_samples)

    # Scale to actual bounds
    samples = np.zeros_like(samples_unit)
    for i in range(n_dims):
        low, high = bounds[i]
        samples[:, i] = samples_unit[:, i] * (high - low) + low

    return torch.tensor(samples, dtype=torch.float32, device=device)


def compute_discrepancy(samples: np.ndarray, method: str = "star") -> float:
    """
    Compute discrepancy of a point set

    Lower discrepancy = better uniformity
    Random sequences: D ≈ O(sqrt(log N / N))
    QMC sequences: D ≈ O((log N)^d / N)

    Args:
        samples: Samples in [0,1]^d (n_samples, n_dims)
        method: 'star', 'L2', or 'centered'

    Returns:
        Discrepancy measure
    """
    from scipy.stats import qmc

    try:
        if method == "star":
            disc = qmc.discrepancy(samples, method="SD")  # Star discrepancy
        elif method == "L2":
            disc = qmc.discrepancy(samples, method="L2-star")
        elif method == "centered":
            disc = qmc.discrepancy(samples, method="CD")  # Centered discrepancy
        else:
            disc = qmc.discrepancy(samples)  # Default

        # If discrepancy is suspiciously small, try alternate computation
        if disc < 1e-10 and len(samples) < 1e6:
            # Compute a simple uniformity metric as backup
            # Check coverage of hypercube subdivisions
            n_bins = int(len(samples) ** (1 / samples.shape[1]))
            hist, _ = np.histogramdd(samples, bins=[n_bins] * samples.shape[1])
            expected = len(samples) / (n_bins ** samples.shape[1])
            disc_alternate = np.sqrt(np.mean((hist - expected) ** 2)) / expected

            if disc_alternate > disc:
                return disc_alternate

        return disc
    except Exception as e:
        # Fallback: compute simple uniformity metric
        print(f"Warning: scipy discrepancy failed ({e}), using fallback metric")
        n_bins = max(10, int(len(samples) ** (1 / samples.shape[1])))
        hist, _ = np.histogramdd(samples, bins=[n_bins] * samples.shape[1])
        expected = len(samples) / (n_bins ** samples.shape[1])
        return np.sqrt(np.mean((hist - expected) ** 2)) / expected


def compute_roa_area_qmc_sobol(
    lyapunov_nn,
    state_limits: Tuple[Tuple[float, float], ...],
    rho: float,
    num_samples: int = 100000,
    device: str = "cpu",
    state_indices: Optional[Tuple[int, ...]] = None,
    compute_discrepancy_metric: bool = False,
    round_to_pow2: bool = True,
) -> ROAMetrics:
    """
    Compute ROA area using Quasi-Monte Carlo with Sobol sequence

    Sobol sequences provide better coverage than random sampling,
    typically requiring 10-100x fewer samples for same accuracy.

    Args:
        lyapunov_nn: Lyapunov function V(x)
        state_limits: Tuple of (min, max) for each dimension
        rho: ROA threshold
        num_samples: Number of Sobol samples
        device: Computing device
        state_indices: Which state dimensions to consider
        compute_discrepancy_metric: Whether to compute discrepancy (slow for large N)

    Returns:
        ROAMetrics object with area measurements
    """

    # convert to floats if not already floats
    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    # Determine dimensions
    if state_indices is None:
        state_indices = tuple(range(len(state_limits)))

    n_dims = len(state_indices)

    # Compute domain volume
    domain_volume = 1.0
    for idx in state_indices:
        domain_volume *= state_limits[idx][1] - state_limits[idx][0]

    # Infer total state dimension
    if hasattr(lyapunov_nn, "parameters"):
        try:
            first_layer = next(lyapunov_nn.parameters())
            nx_total = (
                first_layer.shape[1]
                if len(first_layer.shape) > 1
                else first_layer.shape[0]
            )
        except StopIteration:
            # No parameters - assume nx equals number of limits
            nx_total = len(state_limits)
    else:
        nx_total = len(state_limits)

    # Generate Sobol samples for the dimensions of interest
    bounds_for_sobol = tuple(state_limits[idx] for idx in state_indices)
    sobol_samples = generate_sobol_samples(
        num_samples, n_dims, bounds_for_sobol, device, round_to_pow2
    )

    ## CRITICAL FIX: Use actual number of samples generated, not requested
    num_samples_actual = sobol_samples.shape[0]

    # Create full state samples with correct size
    samples = torch.zeros((num_samples_actual, nx_total), device=device)
    for i, idx in enumerate(state_indices):
        samples[:, idx] = sobol_samples[:, i]

    # Evaluate Lyapunov function
    with torch.no_grad():
        V_values = lyapunov_nn(samples).squeeze()

    # Count samples in ROA
    in_roa = (V_values <= rho).sum().item()

    # Estimate ROA volume
    roa_volume = domain_volume * (in_roa / num_samples_actual)
    coverage_ratio = in_roa / num_samples_actual

    # Optionally compute discrepancy
    discrepancy_val = None
    if compute_discrepancy_metric:
        # Normalize samples to [0,1]^d for discrepancy computation
        samples_normalized = np.zeros((num_samples_actual, n_dims))
        for i, idx in enumerate(state_indices):
            low, high = state_limits[idx]
            samples_normalized[:, i] = (sobol_samples[:, i].cpu().numpy() - low) / (
                high - low
            )

        # Clip to [0,1] to handle any floating point errors
        samples_normalized = np.clip(samples_normalized, 0.0, 1.0)
        discrepancy_val = compute_discrepancy(
            samples_normalized, method="CD"
        )  # Use centered discrepancy

    return ROAMetrics(
        rho=rho,
        area_roa=roa_volume,
        area_domain=domain_volume,
        coverage_ratio=coverage_ratio,
        num_samples_in_roa=in_roa,
        num_samples_total=num_samples_actual,  # Use actual count
        domain_bounds=state_limits,
        method="qmc_sobol",
        discrepancy=discrepancy_val,
    )


def compute_roa_area_qmc_halton(
    lyapunov_nn,
    state_limits: Tuple[Tuple[float, float], ...],
    rho: float,
    num_samples: int = 100000,
    device: str = "cpu",
    state_indices: Optional[Tuple[int, ...]] = None,
    compute_discrepancy_metric: bool = False,
    round_to_pow2: bool = False,  # Halton doesn't require power of 2, but support for consistency
) -> ROAMetrics:
    """
    Compute ROA area using Quasi-Monte Carlo with Halton sequence

    Halton sequences are particularly good for lower dimensions (d <= 10).

    Args:
        lyapunov_nn: Lyapunov function V(x)
        state_limits: Tuple of (min, max) for each dimension
        rho: ROA threshold
        num_samples: Number of Halton samples
        device: Computing device
        state_indices: Which state dimensions to consider
        compute_discrepancy_metric: Whether to compute discrepancy
        round_to_pow2: For consistency with Sobol (Halton doesn't require it)

    Returns:
        ROAMetrics object with area measurements
    """

    # convert to floats if not already floats
    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    # Determine dimensions
    if state_indices is None:
        state_indices = tuple(range(len(state_limits)))

    n_dims = len(state_indices)

    # Compute domain volume
    domain_volume = 1.0
    for idx in state_indices:
        domain_volume *= state_limits[idx][1] - state_limits[idx][0]

    # Infer total state dimension
    if hasattr(lyapunov_nn, "parameters"):
        try:
            first_layer = next(lyapunov_nn.parameters())
            nx_total = (
                first_layer.shape[1]
                if len(first_layer.shape) > 1
                else first_layer.shape[0]
            )
        except StopIteration:
            nx_total = len(state_limits)
    else:
        nx_total = len(state_limits)

    # Generate Halton samples
    bounds_for_halton = tuple(state_limits[idx] for idx in state_indices)
    halton_samples = generate_halton_samples(
        num_samples, n_dims, bounds_for_halton, device
    )

    # Halton doesn't require power of 2, so num_samples stays the same
    num_samples_actual = halton_samples.shape[0]

    # Create full state samples
    samples = torch.zeros((num_samples_actual, nx_total), device=device)
    for i, idx in enumerate(state_indices):
        samples[:, idx] = halton_samples[:, i]

    # Evaluate Lyapunov function
    with torch.no_grad():
        V_values = lyapunov_nn(samples).squeeze()

    # Count samples in ROA
    in_roa = (V_values <= rho).sum().item()

    # Estimate ROA volume
    roa_volume = domain_volume * (in_roa / num_samples_actual)
    coverage_ratio = in_roa / num_samples_actual

    # Optionally compute discrepancy
    discrepancy_val = None
    if compute_discrepancy_metric:
        samples_normalized = np.zeros((num_samples_actual, n_dims))
        for i, idx in enumerate(state_indices):
            low, high = state_limits[idx]
            samples_normalized[:, i] = (halton_samples[:, i].cpu().numpy() - low) / (
                high - low
            )

        # Clip to [0,1]
        samples_normalized = np.clip(samples_normalized, 0.0, 1.0)
        discrepancy_val = compute_discrepancy(samples_normalized, method="CD")

    return ROAMetrics(
        rho=rho,
        area_roa=roa_volume,
        area_domain=domain_volume,
        coverage_ratio=coverage_ratio,
        num_samples_in_roa=in_roa,
        num_samples_total=num_samples_actual,
        domain_bounds=state_limits,
        method="qmc_halton",
        discrepancy=discrepancy_val,
    )


def compute_roa_area_monte_carlo(
    lyapunov_nn,
    state_limits: Tuple[Tuple[float, float], ...],
    rho: float,
    num_samples: int = 100000,
    device: str = "cpu",
    state_indices: Optional[Tuple[int, ...]] = None,
) -> ROAMetrics:
    """
    Compute ROA area using Monte Carlo sampling

    Args:
        lyapunov_nn: Lyapunov function V(x)
        state_limits: Tuple of (min, max) for each dimension
        rho: ROA threshold (points where V(x) <= rho are in ROA)
        num_samples: Number of random samples
        device: Computing device
        state_indices: Which state dimensions to consider (None = all)

    Returns:
        ROAMetrics object with area measurements
    """

    # convert to floats if not already floats
    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    # Determine dimensions
    if state_indices is None:
        state_indices = tuple(range(len(state_limits)))

    n_dims = len(state_indices)

    # Compute domain volume (hyperrectangle)
    domain_volume = 1.0
    for idx in state_indices:
        domain_volume *= state_limits[idx][1] - state_limits[idx][0]

    # Get total state dimension from Lyapunov network
    if hasattr(lyapunov_nn, "parameters"):
        try:
            first_layer = next(lyapunov_nn.parameters())
            nx_total = (
                first_layer.shape[1]
                if len(first_layer.shape) > 1
                else first_layer.shape[0]
            )
        except StopIteration:
            # No parameters - assume nx equals number of limits
            nx_total = len(state_limits)
    else:
        nx_total = len(state_limits)

    # Generate random samples in domain
    samples = torch.zeros((num_samples, nx_total), device=device)

    for i, idx in enumerate(state_indices):
        low, high = state_limits[idx]
        samples[:, idx] = torch.rand(num_samples, device=device) * (high - low) + low

    # Set other dimensions to zero or equilibrium
    for idx in range(nx_total):
        if idx not in state_indices:
            samples[:, idx] = 0.0

    # Evaluate Lyapunov function
    with torch.no_grad():
        V_values = lyapunov_nn(samples).squeeze()

    # Count samples in ROA
    in_roa = (V_values <= rho).sum().item()

    # Estimate ROA volume
    roa_volume = domain_volume * (in_roa / num_samples)
    coverage_ratio = in_roa / num_samples

    return ROAMetrics(
        rho=rho,
        area_roa=roa_volume,
        area_domain=domain_volume,
        coverage_ratio=coverage_ratio,
        num_samples_in_roa=in_roa,
        num_samples_total=num_samples,
        domain_bounds=state_limits,
        method="monte_carlo",
    )


def compute_roa_area_grid(
    lyapunov_nn,
    state_limits: Tuple[Tuple[float, float], Tuple[float, float]],
    rho: float,
    grid_resolution: int = 200,
    device: str = "cpu",
    state_indices: Tuple[int, int] = (0, 1),
) -> ROAMetrics:
    """
    Compute ROA area using uniform grid (more accurate for 2D)

    Args:
        lyapunov_nn: Lyapunov function V(x)
        state_limits: ((x_min, x_max), (y_min, y_max))
        rho: ROA threshold
        grid_resolution: Number of grid points per dimension
        device: Computing device
        state_indices: Which two state dimensions to use

    Returns:
        ROAMetrics object with area measurements
    """

    # convert to floats if not already floats
    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    idx0, idx1 = state_indices

    # Create grid
    x0_range = np.linspace(state_limits[0][0], state_limits[0][1], grid_resolution)
    x1_range = np.linspace(state_limits[1][0], state_limits[1][1], grid_resolution)
    X0, X1 = np.meshgrid(x0_range, x1_range, indexing='ij')

    # Cell area
    dx0 = (state_limits[0][1] - state_limits[0][0]) / grid_resolution
    dx1 = (state_limits[1][1] - state_limits[1][0]) / grid_resolution
    cell_area = dx0 * dx1

    # Total domain area
    domain_area = (state_limits[0][1] - state_limits[0][0]) * (
        state_limits[1][1] - state_limits[1][0]
    )

    # Infer total state dimension
    if hasattr(lyapunov_nn, "parameters"):
        first_layer = next(lyapunov_nn.parameters())
        nx_total = (
            first_layer.shape[1] if len(first_layer.shape) > 1 else first_layer.shape[0]
        )
    else:
        nx_total = max(state_indices) + 1

    # Initialize state grid
    num_points = grid_resolution * grid_resolution
    states_grid = torch.zeros((num_points, nx_total), device=device)
    states_grid[:, idx0] = torch.tensor(
        X0.flatten(), dtype=torch.float32, device=device
    )
    states_grid[:, idx1] = torch.tensor(
        X1.flatten(), dtype=torch.float32, device=device
    )

    # Evaluate Lyapunov function
    with torch.no_grad():
        V_values = lyapunov_nn(states_grid).squeeze()

    # Count cells in ROA
    in_roa = (V_values <= rho).sum().item()

    # Compute ROA area
    roa_area = in_roa * cell_area
    coverage_ratio = in_roa / num_points

    return ROAMetrics(
        rho=rho,
        area_roa=roa_area,
        area_domain=domain_area,
        coverage_ratio=coverage_ratio,
        num_samples_in_roa=in_roa,
        num_samples_total=num_points,
        domain_bounds=state_limits,
        method="grid",
        grid_resolution=grid_resolution,
    )


def compute_roa_volume_nd(
    lyapunov_nn,
    state_limits: Tuple[Tuple[float, float], ...],
    rho: float,
    samples_per_dim: int = 50,
    device: str = "cpu",
) -> ROAMetrics:
    """
    Compute ROA volume for arbitrary dimensional systems using grid method

    WARNING: Computational cost grows exponentially with dimension!
    For n_dims=3, samples_per_dim=50 -> 125,000 evaluations
    For n_dims=4, samples_per_dim=50 -> 6,250,000 evaluations

    Args:
        lyapunov_nn: Lyapunov function
        state_limits: Limits for each dimension
        rho: ROA threshold
        samples_per_dim: Grid points per dimension
        device: Computing device

    Returns:
        ROAMetrics with volume computation
    """

    # convert to floats if not already floats
    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    n_dims = len(state_limits)

    # Warn for high-dimensional cases
    total_points = samples_per_dim**n_dims
    if total_points > 1e7:
        import warnings

        warnings.warn(
            f"Computing {total_points:.2e} points in {n_dims}D. This may be slow. "
            f"Consider using Monte Carlo method instead."
        )

    # Create grid for each dimension
    grids = []
    cell_volume = 1.0
    domain_volume = 1.0

    for idx in range(n_dims):
        low, high = state_limits[idx]
        grid = np.linspace(low, high, samples_per_dim)
        grids.append(grid)

        dx = (high - low) / samples_per_dim
        cell_volume *= dx
        domain_volume *= high - low

    # Create meshgrid
    mesh = np.meshgrid(*grids, indexing="ij")

    # Flatten and create state tensor
    states = torch.zeros((total_points, n_dims), device=device, dtype=torch.float32)
    for i in range(n_dims):
        states[:, i] = torch.tensor(
            mesh[i].flatten(), dtype=torch.float32, device=device
        )

    # Evaluate Lyapunov function in batches to avoid memory issues
    batch_size = 10000
    V_values = []

    with torch.no_grad():
        for i in range(0, total_points, batch_size):
            batch = states[i : i + batch_size]
            V_batch = lyapunov_nn(batch).squeeze()
            V_values.append(V_batch)

    V_values = torch.cat(V_values)

    # Count points in ROA
    in_roa = (V_values <= rho).sum().item()

    # Compute volume
    roa_volume = in_roa * cell_volume
    coverage_ratio = in_roa / total_points

    return ROAMetrics(
        rho=rho,
        area_roa=roa_volume,
        area_domain=domain_volume,
        coverage_ratio=coverage_ratio,
        num_samples_in_roa=in_roa,
        num_samples_total=int(total_points),
        domain_bounds=state_limits,
        method=f"grid_{n_dims}d",
        grid_resolution=samples_per_dim,
    )


def estimate_rho_from_boundary(
    lyapunov_nn,
    state_limits: Tuple[Tuple[float, float], ...],
    grid_resolution: int = 100,
    device: str = "cpu",
    margin_factor: float = 0.9,
) -> float:
    """
    Estimate rho (ROA threshold) from minimum Lyapunov value on domain boundary

    Args:
        lyapunov_nn: Lyapunov function
        state_limits: Domain limits
        grid_resolution: Points per dimension on boundary
        device: Computing device
        margin_factor: Safety margin (rho = margin_factor * V_min_boundary)

    Returns:
        Estimated rho value
    """

    # convert to floats if not already floats
    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    n_dims = len(state_limits)

    # Sample points on each face of the hyperrectangle boundary
    boundary_points = []

    for dim in range(n_dims):
        # Create grid on lower face (dim-th coordinate at minimum)
        grids = []
        for d in range(n_dims):
            if d == dim:
                grids.append([state_limits[d][0]])  # Fixed at min
            else:
                grids.append(
                    np.linspace(state_limits[d][0], state_limits[d][1], grid_resolution)
                )

        mesh = np.meshgrid(*grids, indexing="ij")
        points_lower = np.stack([m.flatten() for m in mesh], axis=1)
        boundary_points.append(points_lower)

        # Create grid on upper face (dim-th coordinate at maximum)
        grids[dim] = [state_limits[dim][1]]  # Fixed at max
        mesh = np.meshgrid(*grids, indexing="ij")
        points_upper = np.stack([m.flatten() for m in mesh], axis=1)
        boundary_points.append(points_upper)

    # Concatenate all boundary points
    boundary_points = np.vstack(boundary_points)
    boundary_tensor = torch.tensor(boundary_points, dtype=torch.float32, device=device)

    # Evaluate Lyapunov function on boundary
    with torch.no_grad():
        V_boundary = lyapunov_nn(boundary_tensor).squeeze()

    # Minimum value on boundary
    rho = V_boundary.min().item() * margin_factor

    return rho


def compare_roa_metrics(
    lyapunov_nn,
    state_limits: Tuple[Tuple[float, float], ...],
    rho: float,
    device: str = "cpu",
    mc_samples: int = 100000,
    grid_resolution: int = 200,
    state_indices: Optional[Tuple[int, int]] = None,
    compute_discrepancy: bool = False,
) -> Dict[str, ROAMetrics]:
    """
    Compare ROA area estimates using different methods

    Args:
        lyapunov_nn: Lyapunov function
        state_limits: Domain limits
        rho: ROA threshold
        device: Computing device
        mc_samples: Samples for Monte Carlo and QMC
        grid_resolution: Resolution for grid method
        state_indices: Which dimensions to analyze (for 2D visualization)
        compute_discrepancy: Whether to compute discrepancy metric

    Returns:
        Dict with metrics from different methods
    """

    # convert to floats if not already floats
    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    results = {}

    # Random Monte Carlo
    print("Computing ROA area via Monte Carlo (random)...")
    mc_metrics = compute_roa_area_monte_carlo(
        lyapunov_nn, state_limits, rho, num_samples=mc_samples, device=device
    )
    results["monte_carlo"] = mc_metrics

    # Sobol QMC
    print("Computing ROA area via Quasi-Monte Carlo (Sobol)...")
    sobol_metrics = compute_roa_area_qmc_sobol(
        lyapunov_nn,
        state_limits,
        rho,
        num_samples=mc_samples,
        device=device,
        state_indices=state_indices,
        compute_discrepancy_metric=compute_discrepancy,
    )
    results["qmc_sobol"] = sobol_metrics

    # Halton QMC
    print("Computing ROA area via Quasi-Monte Carlo (Halton)...")
    halton_metrics = compute_roa_area_qmc_halton(
        lyapunov_nn,
        state_limits,
        rho,
        num_samples=mc_samples,
        device=device,
        state_indices=state_indices,
        compute_discrepancy_metric=compute_discrepancy,
    )
    results["qmc_halton"] = halton_metrics

    # Grid method (for 2D only)
    if state_indices is not None or len(state_limits) == 2:
        print("Computing ROA area via grid method...")
        if state_indices is None:
            state_indices = (0, 1)
            limits_2d = (state_limits[0], state_limits[1])
        else:
            limits_2d = (state_limits[state_indices[0]], state_limits[state_indices[1]])

        grid_metrics = compute_roa_area_grid(
            lyapunov_nn,
            limits_2d,
            rho,
            grid_resolution=grid_resolution,
            device=device,
            state_indices=state_indices,
        )
        results["grid"] = grid_metrics

    # Print comparison
    print("\n" + "=" * 70)
    print("Method Comparison")
    print("=" * 70)
    print(f"{'Method':<20} {'Area':<12} {'Coverage':<12} {'Samples in ROA':<15}")
    print("-" * 70)
    for method_name, metrics in results.items():
        print(
            f"{method_name:<20} {metrics.area_roa:<12.6f} {metrics.coverage_ratio*100:<11.2f}% "
            f"{metrics.num_samples_in_roa:>14,}"
        )

    if compute_discrepancy:
        print("\nDiscrepancy (lower = more uniform):")
        for method_name, metrics in results.items():
            if metrics.discrepancy is not None:
                print(f"  {method_name:<20}: {metrics.discrepancy:.6f}")

    print("=" * 70)

    return results


def compute_roa_area_with_controller(
    lyapunov_nn,
    controller_nn,
    dynamics_system,
    state_limits: Tuple[Tuple[float, float], ...],
    rho: Optional[float] = None,
    method: str = "monte_carlo",
    num_samples: int = 100000,
    grid_resolution: int = 200,
    device: str = "cpu",
    verify_lyapunov_decrease: bool = True,
    state_indices: Optional[Tuple[int, int]] = None,
) -> Dict:
    """
    Compute ROA area and verify Lyapunov decrease condition

    Args:
        lyapunov_nn: Lyapunov function
        controller_nn: Neural controller
        dynamics_system: Dynamical system
        state_limits: Domain limits
        rho: ROA threshold (auto-computed if None)
        method: 'monte_carlo' or 'grid'
        num_samples: For Monte Carlo
        grid_resolution: For grid method
        device: Computing device
        verify_lyapunov_decrease: Check ΔV < 0 in ROA
        state_indices: Dimensions to analyze

    Returns:
        Dict with metrics and verification results
    """

    # convert to floats if not already floats
    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    # Auto-compute rho if not provided
    if rho is None:
        print("Estimating rho from boundary...")
        rho = estimate_rho_from_boundary(
            lyapunov_nn, state_limits, grid_resolution=50, device=device
        )
        print(f"Estimated rho = {rho:.4f}")

    # Compute ROA area
    if method == "monte_carlo":
        metrics = compute_roa_area_monte_carlo(
            lyapunov_nn,
            state_limits,
            rho,
            num_samples=num_samples,
            device=device,
            state_indices=state_indices,
        )
    elif method == "grid" and len(state_limits) == 2:
        metrics = compute_roa_area_grid(
            lyapunov_nn,
            state_limits,
            rho,
            grid_resolution=grid_resolution,
            device=device,
            state_indices=state_indices or (0, 1),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    result = {"metrics": metrics}

    # Verify Lyapunov decrease condition
    if verify_lyapunov_decrease:
        print("\nVerifying Lyapunov decrease condition ΔV < 0...")

        # Sample points in ROA
        if method == "monte_carlo":
            # Generate samples
            n_verify = min(10000, num_samples)
            if state_indices is None:
                state_indices = tuple(range(len(state_limits)))

            nx_total = max(state_indices) + 1
            samples = torch.zeros((n_verify, nx_total), device=device)

            for idx in state_indices:
                low, high = state_limits[idx]
                samples[:, idx] = (
                    torch.rand(n_verify, device=device) * (high - low) + low
                )
        else:
            # Use grid samples
            idx0, idx1 = state_indices
            x0_range = np.linspace(
                state_limits[0][0], state_limits[0][1], grid_resolution
            )
            x1_range = np.linspace(
                state_limits[1][0], state_limits[1][1], grid_resolution
            )
            X0, X1 = np.meshgrid(x0_range, x1_range, indexing="ij")

            nx_total = dynamics_system.nx
            samples = torch.zeros((grid_resolution**2, nx_total), device=device)
            samples[:, idx0] = torch.tensor(
                X0.flatten(), dtype=torch.float32, device=device
            )
            samples[:, idx1] = torch.tensor(
                X1.flatten(), dtype=torch.float32, device=device
            )

        # Evaluate V and ΔV
        with torch.no_grad():
            V_samples = lyapunov_nn(samples).squeeze()
            in_roa_mask = V_samples <= rho

            # Only check points in ROA
            samples_in_roa = samples[in_roa_mask]

            if len(samples_in_roa) > 0:
                # Compute control and next state
                u_samples = controller_nn(samples_in_roa)
                x_next = dynamics_system(samples_in_roa, u_samples)
                V_next = lyapunov_nn(x_next).squeeze()

                # Lyapunov decrease
                V_current = V_samples[in_roa_mask]
                delta_V = V_next - V_current

                # Statistics
                num_decreasing = (delta_V < 0).sum().item()
                num_total = len(delta_V)
                decrease_ratio = num_decreasing / num_total
                max_violation = delta_V.max().item()
                mean_decrease = delta_V.mean().item()

                result["lyapunov_verification"] = {
                    "decrease_ratio": decrease_ratio,
                    "num_decreasing": num_decreasing,
                    "num_total": num_total,
                    "max_violation": max_violation,
                    "mean_delta_V": mean_decrease,
                    "all_decreasing": decrease_ratio == 1.0,
                }

                print(f"  Decreasing at {decrease_ratio*100:.2f}% of ROA points")
                print(f"  Max ΔV violation: {max_violation:.6f}")
                print(f"  Mean ΔV: {mean_decrease:.6f}")
            else:
                print("  Warning: No samples found in ROA!")
                result["lyapunov_verification"] = None

    return result


def compare_roa_sizes(
    lyapunov_models: Dict[str, torch.nn.Module],
    state_limits: Tuple[Tuple[float, float], ...],
    rho_values: Optional[Dict[str, float]] = None,
    method: str = "monte_carlo",
    num_samples: int = 100000,
    grid_resolution: int = 200,
    device: str = "cpu",
    **kwargs,
) -> Dict[str, ROAMetrics]:
    """
    Compare ROA sizes for multiple Lyapunov functions

    Useful for comparing different training runs or architectures

    Args:
        lyapunov_models: Dict of {name: lyapunov_nn}
        state_limits: Domain limits
        rho_values: Optional dict of {name: rho}
        method: 'monte_carlo', 'qmc_sobol', 'qmc_halton', or 'grid'
        num_samples: Number of samples for MC/QMC methods
        grid_resolution: Resolution for grid method
        device: Computing device
        **kwargs: Additional arguments for computation methods

    Returns:
        Dict of {name: ROAMetrics}
    """

    # convert to floats if not already floats
    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    results = {}

    for name, lyap_nn in lyapunov_models.items():
        print(f"\nComputing ROA for '{name}'...")

        rho = rho_values[name] if rho_values and name in rho_values else None

        if rho is None:
            rho = estimate_rho_from_boundary(lyap_nn, state_limits, device=device)
            print(f"  Estimated rho = {rho:.4f}")

        if method == "monte_carlo":
            metrics = compute_roa_area_monte_carlo(
                lyap_nn, state_limits, rho, num_samples=num_samples, device=device
            )
        elif method == "qmc_sobol":
            metrics = compute_roa_area_qmc_sobol(
                lyap_nn, state_limits, rho, num_samples=num_samples, device=device
            )
        elif method == "qmc_halton":
            metrics = compute_roa_area_qmc_halton(
                lyap_nn, state_limits, rho, num_samples=num_samples, device=device
            )
        elif method == "grid":
            if len(state_limits) != 2:
                raise ValueError("Grid method only supports 2D")
            metrics = compute_roa_area_grid(
                lyap_nn,
                state_limits,
                rho,
                grid_resolution=grid_resolution,
                device=device,
                state_indices=(0, 1),
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        results[name] = metrics
        print(f"  ROA area: {metrics.area_roa:.4f}")
        print(f"  Coverage: {metrics.coverage_ratio*100:.2f}%")

    return results


def print_roa_metrics(metrics: ROAMetrics, title: Optional[str] = None):
    """
    Pretty print ROA metrics

    Args:
        metrics: ROAMetrics object
        title: Optional title
    """
    if title:
        print(f"\n{'='*70}")
        print(f"{title}")
        print(f"{'='*70}")

    print(f"Method: {metrics.method}")
    if metrics.grid_resolution:
        print(f"Grid resolution: {metrics.grid_resolution}")
    print(f"Total samples: {metrics.num_samples_total:,}")
    print(f"Samples in ROA: {metrics.num_samples_in_roa:,}")
    print(f"\nROA threshold (ρ): {metrics.rho:.6f}")
    print(f"Domain area/volume: {metrics.area_domain:.6f}")
    print(f"ROA area/volume: {metrics.area_roa:.6f}")
    print(f"Coverage ratio: {metrics.coverage_ratio*100:.6f}%")
    print(f"{'='*70}")
