import numpy as np
import cvxpy as cp
from typing import Tuple, List

def verify_lp_single_output(Ws: List[np.ndarray], bs: List[np.ndarray], 
                            in_range: np.ndarray, output_idx: int, 
                            maximize: bool = True) -> float:
    """
    Verify a single output dimension using LP.
    Optimized version: compute bounds and build constraints in one pass.
    
    Args:
        Ws: List of weight matrices
        bs: List of bias vectors
        in_range: Input range as (n_inputs, 2) array [lower, upper]
        output_idx: Which output dimension to optimize
        maximize: If True, maximize; if False, minimize
        
    Returns:
        Optimal value (max or min) for the specified output
    """
    num_layers = len(Ws)
    constraints = []
    
    # Create variables for all layers at once
    z_vars = []
    
    # Input layer
    input_dim = in_range.shape[0]
    z_input = cp.Variable(input_dim)
    z_vars.append(z_input)
    
    # Add input constraints
    for i in range(input_dim):
        constraints.append(z_vars[0][i] >= in_range[i, 0])
        constraints.append(z_vars[0][i] <= in_range[i, 1])
    
    # Create variables for intermediate layers
    for i in range(num_layers - 1):
        layer_dim = Ws[i].shape[0]
        z_vars.append(cp.Variable(layer_dim))
    
    # Output layer (no ReLU)
    output_dim = Ws[-1].shape[0]
    z_output = cp.Variable(output_dim)
    z_vars.append(z_output)
    
    # Initialize bounds for layer 0 (input)
    z_low = in_range[:, 0].copy()
    z_up = in_range[:, 1].copy()
    
    # Process each layer: compute bounds AND add constraints in one pass
    for layer_idx in range(num_layers - 1):
        W, b = Ws[layer_idx], bs[layer_idx]
        z_prev = z_vars[layer_idx]
        z_curr = z_vars[layer_idx + 1]
        
        # Step 1: Propagate bounds through affine transformation (IBP)
        mu = (z_up + z_low) / 2
        r = (z_up - z_low) / 2
        
        mu_next = W @ mu + b
        r_next = np.abs(W) @ r
        
        z_low_affine = mu_next - r_next  # Pre-activation lower bound
        z_up_affine = mu_next + r_next    # Pre-activation upper bound
        
        # Step 2: For each neuron, classify and add constraints immediately
        pre_activation = W @ z_prev + b  # CVXPY expression
        
        for neuron_idx in range(len(z_low_affine)):
            l = z_low_affine[neuron_idx]
            u = z_up_affine[neuron_idx]
            
            # Classify neuron based on bounds
            if l > 0:
                # Active neuron: z = pre_activation
                constraints.append(z_curr[neuron_idx] == pre_activation[neuron_idx])
                
            elif u <= 0:
                # Inactive neuron: z = 0
                constraints.append(z_curr[neuron_idx] == 0)
                
            else:
                # Uncertain neuron: triangle relaxation
                # Constraint 1: z >= 0
                constraints.append(z_curr[neuron_idx] >= 0)
                
                # Constraint 2: z >= pre_activation
                constraints.append(z_curr[neuron_idx] >= pre_activation[neuron_idx])
                
                # Constraint 3: upper envelope
                # z <= u/(u-l) * (pre_activation - l)
                slope = u / (u - l)
                constraints.append(
                    z_curr[neuron_idx] <= slope * (pre_activation[neuron_idx] - l)
                )
        
        # Step 3: Update bounds for next layer (apply ReLU)
        z_low = np.maximum(z_low_affine, 0)
        z_up = np.maximum(z_up_affine, 0)
    
    # Final layer constraint (no ReLU)
    W_last, b_last = Ws[-1], bs[-1]
    constraints.append(z_vars[-1] == W_last @ z_vars[-2] + b_last)
    
    # Set objective
    if maximize:
        objective = cp.Maximize(z_vars[-1][output_idx])
    else:
        objective = cp.Minimize(z_vars[-1][output_idx])
    
    # Solve - try different solvers
    problem = cp.Problem(objective, constraints)
    
    solvers_to_try = [cp.CLARABEL, cp.ECOS, cp.SCS, cp.OSQP, cp.GLPK_MI, cp.SCIPY]
    
    solved = False
    for solver in solvers_to_try:
        try:
            problem.solve(solver=solver, verbose=False)
            if problem.status in ["optimal", "optimal_inaccurate"]:
                solved = True
                break
        except Exception as e:
            continue
    
    if not solved:
        # Try without specifying solver
        try:
            problem.solve(verbose=False)
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Warning: LP solver status: {problem.status}")
                return np.nan
        except Exception as e:
            print(f"Error solving LP: {e}")
            return np.nan
    
    return problem.value


def verify_lp(nnet_filename: str, in_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Verify neural network using LP-based method.
    
    Args:
        nnet_filename: Path to .nnet file
        in_range: Input range as (n_inputs, 2) array [lower, upper]
        
    Returns:
        lower_bounds: Lower bounds for each output
        upper_bounds: Upper bounds for each output
    """
    print('\n\n--- verify_lp ---')
    
    # Load network
    Ws, bs = nnet_to_weights_and_biases(nnet_filename)
    
    output_dim = Ws[-1].shape[0]
    
    lower_bounds = np.zeros(output_dim)
    upper_bounds = np.zeros(output_dim)
    
    # Solve LP for each output dimension
    for i in range(output_dim):
        print(f'Computing bounds for output {i}...')
        
        # Minimize to get lower bound
        lower_bounds[i] = verify_lp_single_output(Ws, bs, in_range, i, maximize=False)
        
        # Maximize to get upper bound
        upper_bounds[i] = verify_lp_single_output(Ws, bs, in_range, i, maximize=True)
    
    print('- LP bounds:')
    print(f'upper_bounds = {upper_bounds}')
    print(f'lower_bounds = {lower_bounds}\n')
    
    return lower_bounds, upper_bounds


# Helper function to load nnet files
def nnet_to_weights_and_biases(nnet_filename: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load a nnet text file and extract the weights and biases."""
    with open(nnet_filename, 'r') as f:
        line = f.readline()
        
        while '//' in line:  # skip comments
            line = f.readline()
        
        # number of layers
        nlayers = int(line.strip().split(',')[0])
        
        # read in layer sizes
        layer_sizes = [int(x) for x in f.readline().split(',')[1:nlayers+1]]
        
        # read past additional information
        for i in range(1, 6):
            line = f.readline()
        
        # i=1 corresponds to the input dimension, so it's ignored
        Ws = []
        bs = []
        for dim in layer_sizes:
            W = np.vstack([[float(x) for x in f.readline().rstrip(',\n ').split(',')] 
                          for i in range(dim)])
            b = np.array([float(f.readline().rstrip(',\n ')) for _ in range(dim)])
            Ws.append(W)
            bs.append(b)
    
    return Ws, bs


# Example usage
if __name__ == "__main__":
    nnet_filename = 'cartpole_nnet.nnet'
    
    # Use the nominal input from the problem
    nominal = np.array([0.0, 0.1, 0.2, 0.3])
    epsilon = 0.1
    in_range = np.vstack([nominal - epsilon, nominal + epsilon]).T
    
    # Run LP verification
    lower_bounds, upper_bounds = verify_lp(nnet_filename, in_range)
    
    print("\nFinal Results:")
    print(f"Output bounds: [{lower_bounds[0]:.4f}, {upper_bounds[0]:.4f}] and [{lower_bounds[1]:.4f}, {upper_bounds[1]:.4f}]")
