"""
Symbolic Dynamical Systems Framework

A framework for defining dynamical systems symbolically using SymPy and
automatically generating PyTorch-compatible numerical functions.
"""

import copy
import time
import sympy as sp
from sympy import pycode
import numpy as np
import scipy
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Callable
from enum import Enum
import control
import warnings
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Standard mapping from SymPy functions to PyTorch functions
# CRITICAL: SymPy uses capital letters for some functions (Abs, Min, Max, Pow)
def _torch_min(*args):
    """Handle Min for both scalars and tensors"""
    if len(args) == 2:
        a, b = args
        # Convert both to tensors to handle scalar/tensor combinations
        a_tensor = torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a
        b_tensor = torch.as_tensor(b) if not isinstance(b, torch.Tensor) else b
        return torch.minimum(a_tensor, b_tensor)
    else:
        # Multiple arguments - stack and take min
        tensors = [
            torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a for a in args
        ]
        return torch.min(torch.stack(tensors))


def _torch_max(*args):
    """Handle Max for both scalars and tensors"""
    if len(args) == 2:
        a, b = args
        # Convert both to tensors to handle scalar/tensor combinations
        a_tensor = torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a
        b_tensor = torch.as_tensor(b) if not isinstance(b, torch.Tensor) else b
        return torch.maximum(a_tensor, b_tensor)
    else:
        # Multiple arguments - stack and take max
        tensors = [
            torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a for a in args
        ]
        return torch.max(torch.stack(tensors))


def _identity_matrix(*args):
    """Handle ImmutableDenseMatrix - just return the args as tuple"""
    return args if len(args) > 1 else args[0]


SYMPY_TO_TORCH = {
    # Trigonometric
    "sin": torch.sin,
    "cos": torch.cos,
    "tan": torch.tan,
    "asin": torch.asin,
    "acos": torch.acos,
    "atan": torch.atan,
    "atan2": torch.atan2,
    "sinh": torch.sinh,
    "cosh": torch.cosh,
    "tanh": torch.tanh,
    # Exponential/Logarithmic
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    # Absolute value and sign - CRITICAL: SymPy uses 'Abs' not 'abs'
    "Abs": torch.abs,
    "abs": torch.abs,  # Include lowercase for safety
    "sign": torch.sign,
    # Min/Max - CRITICAL: SymPy uses 'Min' and 'Max' (capital letters)
    # Use helper functions to handle scalar/tensor combinations
    "Min": _torch_min,
    "Max": _torch_max,
    # Power - CRITICAL: SymPy uses 'Pow' (capital P)
    "Pow": torch.pow,
    # Rounding
    "floor": torch.floor,
    "ceil": torch.ceil,
    "round": torch.round,
    # Additional useful functions
    "clip": torch.clamp,
    "minimum": torch.minimum,
    "maximum": torch.maximum,
    # Matrix handling - SymPy sometimes uses these
    "ImmutableDenseMatrix": _identity_matrix,
    "MutableDenseMatrix": _identity_matrix,
    "Matrix": _identity_matrix,
}


class IntegrationMethod(Enum):
    """Available numerical integration methods"""

    ExplicitEuler = 1
    MidPoint = 2
    RK4 = 3


class SymbolicDynamicalSystem(ABC, nn.Module):
    """
    Base class for dynamical systems defined symbolically with SymPy.
    Provides automatic generation of numerical functions and linearizations.
    Compatible with the DiscreteTimeSystem interface.

    Attributes:
        state_vars: List of symbolic state variables
        control_vars: List of symbolic control variables
        output_vars: List of symbolic output variables
        parameters: Dict mapping SymPy symbols to numerical values
        order: System order (1=first-order, 2=second-order, etc.)
    """

    def __init__(self):
        super().__init__()
        # To be defined by subclasses
        self.state_vars: List[sp.Symbol] = []
        self.control_vars: List[sp.Symbol] = []
        self.output_vars: List[sp.Symbol] = []
        self.parameters: Dict[sp.Symbol, float] = {}  # Symbols as keys!

        # Symbolic expressions (to be defined)
        self._f_sym: Optional[sp.Matrix] = None  # State dynamics: dx/dt = f(x, u)
        self._h_sym: Optional[sp.Matrix] = None  # Output: y = h(x)

        # System order (1 for first-order, 2 for second-order, etc.)
        self.order: int = 1

        # Cached numerical functions
        self._f_numpy: Optional[Callable] = None
        self._f_torch: Optional[Callable] = None
        self._h_torch: Optional[Callable] = None

        # Cached Jacobians for efficiency
        self._A_sym_cached: Optional[sp.Matrix] = None
        self._B_sym_cached: Optional[sp.Matrix] = None
        self._C_sym_cached: Optional[sp.Matrix] = None

        # Flag to track if system has been properly initialized
        self._initialized: bool = False

        # Performance statistics
        self._perf_stats = {
            "forward_calls": 0,
            "forward_time": 0.0,
            "linearization_calls": 0,
            "linearization_time": 0.0,
        }

    @abstractmethod
    def define_system(self, *args, **kwargs):
        """
        Define the symbolic system. Must set:
        - self.state_vars: List of state symbols
        - self.control_vars: List of control symbols
        - self.output_vars: List of output symbols (optional)
        - self.parameters: Dict with Symbol keys (not strings!)
        - self._f_sym: Symbolic dynamics matrix
        - self._h_sym: Symbolic output matrix (optional)
        - self.order: System order (default: 1)

        CRITICAL: self.parameters must use SymPy Symbol objects as keys!
        Example: {m: 1.0, l: 0.5} NOT {'m': 1.0, 'l': 0.5}

        Args:
            *args, **kwargs: System-specific parameters
        """
        pass

    def _validate_system(self) -> bool:
        """Validate that the system is properly defined"""
        errors = []

        if not self.state_vars:
            errors.append("state_vars is empty")

        if not self.control_vars:
            errors.append("control_vars is empty")

        if self._f_sym is None:
            errors.append("_f_sym is not defined")

        if self.parameters:
            for key in self.parameters.keys():
                if not isinstance(key, sp.Symbol):
                    errors.append(f"Parameter key {key} is not a SymPy Symbol")

        if errors:
            error_msg = "System validation failed:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            error_msg += "\n\nHINT: Did you use Symbol objects as parameter keys?"
            error_msg += "\n  Correct:   {m: 1.0, l: 0.5}"
            error_msg += "\n  Incorrect: {'m': 1.0, 'l': 0.5}"
            raise ValueError(error_msg)

        self._initialized = True
        return True

    @property
    def nx(self) -> int:
        """Number of states"""
        return len(self.state_vars)

    @property
    def nu(self) -> int:
        """Number of controls"""
        return len(self.control_vars)

    @property
    def ny(self) -> int:
        """Number of outputs"""
        if self.output_vars:
            return len(self.output_vars)
        elif self._h_sym is not None:
            return self._h_sym.shape[0]
        else:
            return self.nx

    @property
    def nq(self) -> int:
        """Number of generalized coordinates (for higher-order systems)"""
        return self.nx // self.order if self.order > 1 else self.nx

    @property
    def x_equilibrium(self) -> torch.Tensor:
        """Equilibrium state (override in subclass if needed)"""
        return torch.zeros(self.nx)

    @property
    def u_equilibrium(self) -> torch.Tensor:
        """Equilibrium control (override in subclass if needed)"""
        return torch.zeros(self.nu)

    def substitute_parameters(
        self, expr: Union[sp.Expr, sp.Matrix]
    ) -> Union[sp.Expr, sp.Matrix]:
        """
        Substitute numerical parameter values into symbolic expression

        Args:
            expr: SymPy expression or matrix

        Returns:
            Expression with parameters substituted
        """
        return expr.subs(self.parameters)

    def _cache_jacobians(self):
        """Cache symbolic Jacobians for improved performance"""
        if self._f_sym is not None and self._A_sym_cached is None:
            self._A_sym_cached = self._f_sym.jacobian(self.state_vars)
            self._B_sym_cached = self._f_sym.jacobian(self.control_vars)

        if self._h_sym is not None and self._C_sym_cached is None:
            self._C_sym_cached = self._h_sym.jacobian(self.state_vars)

    def linearized_dynamics_symbolic(
        self, x_eq: Optional[sp.Matrix] = None, u_eq: Optional[sp.Matrix] = None
    ) -> Tuple[sp.Matrix, sp.Matrix]:
        """
        Compute symbolic linearization A = df/dx, B = df/du

        For second-order systems, constructs the full state-space linearization
        from the acceleration dynamics.

        Args:
            x_eq: Equilibrium state (zeros if None)
            u_eq: Equilibrium control (zeros if None)

        Returns:
            (A, B): Linearized state and control matrices
        """
        if x_eq is None:
            x_eq = sp.Matrix([0] * self.nx)
        if u_eq is None:
            u_eq = sp.Matrix([0] * self.nu)

        # Use cached Jacobians if available
        if self._A_sym_cached is None:
            self._cache_jacobians()

        if self.order == 1:
            # First-order system: straightforward Jacobian
            A_sym = self._A_sym_cached
            B_sym = self._B_sym_cached
        elif self.order == 2:
            # Second-order system: x = [q, qdot], qddot = f(x, u)
            # Need to construct full state-space form:
            # d/dt [q]    = [      0       I ]  [q]    + [  0  ] u
            #      [qdot]   [df/dq   df/dqdot]  [qdot]   [df/du]

            nq = self.nq

            # Compute Jacobians of acceleration w.r.t. q and qdot
            A_accel = self._f_sym.jacobian(self.state_vars)  # (nq, nx)
            B_accel = self._B_sym_cached  # (nq, nu)

            # Construct full state-space matrices
            A_sym = sp.zeros(self.nx, self.nx)
            A_sym[:nq, nq:] = sp.eye(nq)  # dq/dt = qdot
            A_sym[nq:, :] = A_accel  # dqdot/dt = f(q, qdot, u)

            B_sym = sp.zeros(self.nx, self.nu)
            B_sym[nq:, :] = B_accel  # Control affects acceleration
        else:
            # Higher-order systems
            # x = [q, q', q'', ..., q^(n-1)], q^(n) = f(x, u)
            # State-space form has similar structure
            nq = self.nq
            order = self.order

            A_highest = self._f_sym.jacobian(
                self.state_vars
            )  # Jacobian of highest derivative
            B_highest = self._B_sym_cached

            A_sym = sp.zeros(self.nx, self.nx)
            # Each derivative becomes the next one
            for i in range(order - 1):
                A_sym[i * nq : (i + 1) * nq, (i + 1) * nq : (i + 2) * nq] = sp.eye(nq)
            # Highest derivative
            A_sym[(order - 1) * nq :, :] = A_highest

            B_sym = sp.zeros(self.nx, self.nu)
            B_sym[(order - 1) * nq :, :] = B_highest

        # Substitute equilibrium point
        subs_dict = dict(
            zip(self.state_vars + self.control_vars, list(x_eq) + list(u_eq))
        )
        A = A_sym.subs(subs_dict)
        B = B_sym.subs(subs_dict)

        # Substitute parameters
        A = self.substitute_parameters(A)
        B = self.substitute_parameters(B)

        return A, B

    def linearized_observation_symbolic(
        self, x_eq: Optional[sp.Matrix] = None
    ) -> sp.Matrix:
        """
        Compute symbolic linearization C = dh/dx

        Args:
            x_eq: Equilibrium state (zeros if None)

        Returns:
            C: Linearized output matrix
        """
        if self._h_sym is None:
            return sp.eye(self.nx)

        if x_eq is None:
            x_eq = sp.Matrix([0] * self.nx)

        # Use cached Jacobian if available
        if self._C_sym_cached is None:
            self._cache_jacobians()

        subs_dict = dict(zip(self.state_vars, list(x_eq)))
        C = self._C_sym_cached.subs(subs_dict)
        C = self.substitute_parameters(C)

        return C

    def generate_numpy_function(self) -> Callable:
        """
        Generate lambdified NumPy function for dynamics

        Returns:
            Callable function compatible with NumPy
        """
        f_with_params = self.substitute_parameters(self._f_sym)
        all_vars = self.state_vars + self.control_vars
        self._f_numpy = sp.lambdify(all_vars, f_with_params, modules="numpy")
        return self._f_numpy

    def generate_torch_function(self) -> Callable:
        """
        Generate PyTorch-compatible function for dynamics using code generation

        This method generates Python code as a string and executes it to create
        a function that uses PyTorch operations. This approach avoids issues with
        SymPy's lambdify and PyTorch tensor operations.

        Returns:
            Callable function compatible with PyTorch tensors
        """
        f_with_params = self.substitute_parameters(self._f_sym)
        f_with_params = sp.simplify(f_with_params)

        all_vars = self.state_vars + self.control_vars

        # Generate function signature
        func_code_lines = [
            "def dynamics_func(" + ", ".join([str(v) for v in all_vars]) + "):",
            "    import torch",
        ]

        # Generate code for each output component
        results = []
        for i, expr in enumerate(f_with_params):
            code = pycode(expr)
            # Replace module prefixes with torch
            code = code.replace("numpy.", "torch.")
            code = code.replace("math.", "torch.")

            var_name = f"result_{i}"
            func_code_lines.append(f"    {var_name} = {code}")
            results.append(var_name)

        # Return tuple of results
        func_code_lines.append(f"    return ({', '.join(results)},)")
        func_code = "\n".join(func_code_lines)

        # Execute the generated code
        namespace = {"torch": torch}
        exec(func_code, namespace)
        base_func = namespace["dynamics_func"]

        # Wrap to ensure proper tensor handling
        def wrapped_func(*args):
            result = base_func(*args)

            if isinstance(result, (list, tuple)):
                return torch.stack(list(result), dim=-1)
            elif isinstance(result, torch.Tensor):
                if len(result.shape) == 1:
                    return result.unsqueeze(-1)
                return result
            else:
                return torch.tensor([result]).unsqueeze(0)

        self._f_torch = wrapped_func
        return self._f_torch

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Evaluate continuous-time dynamics: compute state derivative dx/dt = f(x, u)
    
        **CRITICAL DISTINCTION**: This method returns the DERIVATIVE (rate of change)
        of the state, NOT the next state value. This is fundamentally different from
        discrete-time systems.

        Args:
            x: State tensor (batch_size, nx) or (nx,)
            u: Control tensor (batch_size, nu) or (nu,)

        Returns:
            State derivative tensor (same shape as input)

        Raises:
            ValueError: If input dimensions don't match system dimensions
        """

        start_time = time.time()

        # Input validation - handle edge cases
        if len(x.shape) == 0 or len(u.shape) == 0:
            raise ValueError("Input tensors must be at least 1D")

        # Check dimensions only if tensors are at least 1D
        if len(x.shape) >= 1 and x.shape[-1] != self.nx:
            raise ValueError(f"Expected state dimension {self.nx}, got {x.shape[-1]}")
        if len(u.shape) >= 1 and u.shape[-1] != self.nu:
            raise ValueError(f"Expected control dimension {self.nu}, got {u.shape[-1]}")

        if self._f_torch is None:
            self.generate_torch_function()

        # Handle batched vs single evaluation
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Prepare arguments
        x_list = [x[:, i] for i in range(self.nx)]
        u_list = [u[:, i] for i in range(self.nu)]
        all_args = x_list + u_list

        # Call generated function
        result = self._f_torch(*all_args)

        if squeeze_output:
            result = result.squeeze(0)

        # Update performance stats
        self._perf_stats["forward_calls"] += 1
        self._perf_stats["forward_time"] += time.time() - start_time

        return result

    def linearized_dynamics(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Numerical evaluation of linearized dynamics at point (x, u)

        Args:
            x: State tensor
            u: Control tensor

        Returns:
            (A, B): Linearized dynamics matrices as PyTorch tensors
        """
        start_time = time.time()

        # Handle batched input
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        # Allocate output tensors
        A_batch = torch.zeros(batch_size, self.nx, self.nx, dtype=dtype, device=device)
        B_batch = torch.zeros(batch_size, self.nx, self.nu, dtype=dtype, device=device)

        # Evaluate for each sample
        for i in range(batch_size):
            # Convert to numpy - handle both 1D and potential 0D cases
            x_i = x[i] if batch_size > 1 else x.squeeze(0)
            u_i = u[i] if batch_size > 1 else u.squeeze(0)

            x_np = x_i.detach().cpu().numpy()
            u_np = u_i.detach().cpu().numpy()

            # Ensure arrays are at least 1D for SymPy Matrix
            x_np = np.atleast_1d(x_np)
            u_np = np.atleast_1d(u_np)

            A_sym, B_sym = self.linearized_dynamics_symbolic(
                sp.Matrix(x_np), sp.Matrix(u_np)
            )
            A_batch[i] = torch.tensor(
                np.array(A_sym, dtype=np.float64), dtype=dtype, device=device
            )
            B_batch[i] = torch.tensor(
                np.array(B_sym, dtype=np.float64), dtype=dtype, device=device
            )

        if squeeze_output:
            A_batch = A_batch.squeeze(0)
            B_batch = B_batch.squeeze(0)

        # Update performance stats
        self._perf_stats["linearization_calls"] += 1
        self._perf_stats["linearization_time"] += time.time() - start_time

        return A_batch, B_batch

    def linearized_observation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Numerical evaluation of output linearization C = dh/dx

        Args:
            x: State tensor

        Returns:
            C: Linearized observation matrix as PyTorch tensor
        """

        # Handle batched input
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        C_batch = torch.zeros(batch_size, self.ny, self.nx, dtype=dtype, device=device)

        for i in range(batch_size):
            # Handle indexing properly
            x_i = x[i] if batch_size > 1 else x.squeeze(0)
            x_np = x_i.detach().cpu().numpy()

            # Ensure at least 1D
            x_np = np.atleast_1d(x_np)

            C_sym = self.linearized_observation_symbolic(sp.Matrix(x_np))
            C_batch[i] = torch.tensor(
                np.array(C_sym, dtype=np.float64), dtype=dtype, device=device
            )

        if squeeze_output:
            C_batch = C_batch.squeeze(0)

        return C_batch

    def h(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate output equation: y = h(x)

        Args:
            x: State tensor

        Returns:
            Output tensor
        """
        if self._h_sym is None:
            return x

        # Generate torch function for h if not cached
        if self._h_torch is None:
            h_with_params = self.substitute_parameters(self._h_sym)
            # Use ONLY custom namespace (don't add 'torch' as fallback to avoid conflicts)
            self._h_torch = sp.lambdify(
                self.state_vars, h_with_params, modules=[SYMPY_TO_TORCH]
            )

        # Handle batched input
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        x_list = [x[:, i] for i in range(self.nx)]
        result = self._h_torch(*x_list)

        # Handle various return types from lambdify
        def flatten_result(r):
            """Recursively flatten nested lists/tuples to get tensors"""
            if isinstance(r, torch.Tensor):
                return [r]
            elif isinstance(r, (list, tuple)):
                flat = []
                for item in r:
                    flat.extend(flatten_result(item))
                return flat
            else:
                # Scalar - convert to tensor
                return [torch.as_tensor(r)]

        if isinstance(result, (list, tuple)):
            flat_tensors = flatten_result(result)
            result = torch.stack(flat_tensors, dim=-1)
        elif not isinstance(result, torch.Tensor):
            result = torch.as_tensor(result).unsqueeze(-1)

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def print_equations(self, simplify: bool = True):
        """
        Print symbolic equations in human-readable format

        Args:
            simplify: Whether to simplify expressions before printing
        """
        print("=" * 70)
        print(f"{self.__class__.__name__}")
        print("=" * 70)
        print(f"State Variables: {self.state_vars}")
        print(f"Control Variables: {self.control_vars}")
        print(f"System Order: {self.order}")
        print(f"Dimensions: nx={self.nx}, nu={self.nu}, ny={self.ny}")

        print("\nDynamics: dx/dt = f(x, u)")
        for i, (var, expr) in enumerate(zip(self.state_vars, self._f_sym)):
            expr_sub = self.substitute_parameters(expr)
            if simplify:
                expr_sub = sp.simplify(expr_sub)
            print(f"  d{var}/dt = {expr_sub}")

        if self._h_sym is not None:
            print("\nOutput: y = h(x)")
            for i, expr in enumerate(self._h_sym):
                expr_sub = self.substitute_parameters(expr)
                if simplify:
                    expr_sub = sp.simplify(expr_sub)
                print(f"  y[{i}] = {expr_sub}")

        print("=" * 70)

    def check_equilibrium(
        self, x_eq: torch.Tensor, u_eq: torch.Tensor, tol: float = 1e-6
    ) -> Tuple[bool, float]:
        """
        Check if (x_eq, u_eq) is an equilibrium point

        Args:
            x_eq: Candidate equilibrium state
            u_eq: Candidate equilibrium control
            tol: Tolerance for considering derivative as zero

        Returns:
            (is_equilibrium, max_derivative): Boolean and max derivative magnitude
        """
        with torch.no_grad():
            dx = self.forward(
                x_eq.unsqueeze(0) if len(x_eq.shape) == 1 else x_eq,
                u_eq.unsqueeze(0) if len(u_eq.shape) == 1 else u_eq,
            )
            max_deriv = torch.abs(dx).max().item()
            is_eq = max_deriv < tol
        return is_eq, max_deriv

    def eigenvalues_at_equilibrium(self) -> np.ndarray:
        """
        Compute eigenvalues of linearization at equilibrium

        Returns:
            Eigenvalues as complex numpy array
        """
        x_eq = self.x_equilibrium.unsqueeze(0)
        u_eq = self.u_equilibrium.unsqueeze(0)
        A, _ = self.linearized_dynamics(x_eq, u_eq)
        A_np = A.squeeze().detach().cpu().numpy()
        eigenvalues = np.linalg.eigvals(A_np)
        return eigenvalues

    def is_stable_equilibrium(self, discrete_time: bool = False) -> bool:
        """
        Check if equilibrium is stable based on eigenvalues

        Args:
            discrete_time: If True, check |λ| < 1; if False, check Re(λ) < 0

        Returns:
            True if equilibrium is stable
        """
        eigs = self.eigenvalues_at_equilibrium()
        if discrete_time:
            return bool(np.all(np.abs(eigs) < 1.0))
        else:
            return bool(np.all(np.real(eigs) < 0.0))

    def clone(self):
        """Create a deep copy of the system"""
        return copy.deepcopy(self)

    def to_device(self, device: Union[str, torch.device]):
        """
        Move system to specified device

        Args:
            device: Target device ('cpu', 'cuda', or torch.device)

        Returns:
            Self for chaining
        """
        if isinstance(device, str):
            device = torch.device(device)

        # Move equilibrium points
        if hasattr(self, "_x_eq_cached"):
            self._x_eq_cached = self._x_eq_cached.to(device)
        if hasattr(self, "_u_eq_cached"):
            self._u_eq_cached = self._u_eq_cached.to(device)

        return self

    def verify_jacobians(
        self, x: torch.Tensor, u: torch.Tensor, tol: float = 1e-3
    ) -> Dict[str, Union[bool, float]]:
        """
        Verify symbolic Jacobians against numerical finite differences
    
        Checks:
        - A_match: Does ∂f/∂x from SymPy match autograd?
        - B_match: Does ∂f/∂u from SymPy match autograd?
        
        Use for:
        - Debugging symbolic derivations after system modifications
        - Ensuring code generation correctness
        - Validating against hardcoded implementations

        Args:
            x: State at which to verify (can be 1D or 2D)
            u: Control at which to verify (can be 1D or 2D)
            tol: Tolerance for considering Jacobians equal

        Returns:
            Dict with 'A_match', 'B_match' booleans and error magnitudes
        """
        # Ensure proper 2D shape (batch_size=1, dim)
        x_2d = x.reshape(1, -1) if len(x.shape) <= 1 else x
        u_2d = u.reshape(1, -1) if len(u.shape) <= 1 else u

        # Clone for autograd - keep 2D shape
        x_grad = x_2d.clone().requires_grad_(True)
        u_grad = u_2d.clone().requires_grad_(True)

        # Compute symbolic Jacobians
        A_sym, B_sym = self.linearized_dynamics(x_2d.detach(), u_2d.detach())

        # Ensure 3D shape for batch processing
        if len(A_sym.shape) == 2:
            A_sym = A_sym.unsqueeze(0)
            B_sym = B_sym.unsqueeze(0)

        # Compute numerical Jacobians via autograd
        fx = self.forward(x_grad, u_grad)  # fx shape: (1, n_outputs)

        # - First-order: n_outputs = nx (all state derivatives)
        # - Second-order: n_outputs = nq (only accelerations)
        # - Higher-order: n_outputs = nq (highest derivative only)
        if self.order == 1:
            n_outputs = self.nx
        else:
            n_outputs = self.nq

        # For higher-order systems, the Jacobians A and B are of full state-space form
        # but forward() only returns the highest derivative. We need to verify only
        # the relevant part of the Jacobians.
        A_num = torch.zeros_like(A_sym)
        B_num = torch.zeros_like(B_sym)

        if self.order == 1:
            # First-order: forward() returns dx/dt, verify full A and B
            for i in range(n_outputs):
                if fx[0, i].requires_grad:
                    grad_x = torch.autograd.grad(
                        fx[0, i], x_grad, retain_graph=True, create_graph=False
                    )[0]
                    grad_u = torch.autograd.grad(
                        fx[0, i], u_grad, retain_graph=True, create_graph=False
                    )[0]
                    A_num[0, i] = grad_x[0]  # grad_x shape: (1, nx)
                    B_num[0, i] = grad_u[0]  # grad_u shape: (1, nu)
        else:
            # Higher-order: forward() returns highest derivative only
            # The full state-space A matrix has structure:
            # For second-order: A = [[0, I], [A_accel]]
            # We verify only the A_accel part (rows nq:nx)

            for i in range(n_outputs):
                if fx[0, i].requires_grad:
                    grad_x = torch.autograd.grad(
                        fx[0, i], x_grad, retain_graph=True, create_graph=False
                    )[0]
                    grad_u = torch.autograd.grad(
                        fx[0, i], u_grad, retain_graph=True, create_graph=False
                    )[0]

                    # Place in the acceleration rows of the full state-space matrix
                    row_idx = (self.order - 1) * self.nq + i
                    A_num[0, row_idx] = grad_x[0]
                    B_num[0, row_idx] = grad_u[0]

            # For the derivative relationships (upper rows), we verify analytically
            # These should be identity blocks: dq/dt = qdot, etc.
            # The symbolic linearization already includes these, so we just copy them
            for i in range((self.order - 1) * self.nq):
                A_num[0, i] = A_sym[0, i]
                B_num[0, i] = B_sym[0, i]

        A_error = (A_sym - A_num).abs().max().item()
        B_error = (B_sym - B_num).abs().max().item()
        A_match = A_error < tol
        B_match = B_error < tol

        return {
            "A_match": bool(A_match),
            "B_match": bool(B_match),
            "A_error": float(A_error),
            "B_error": float(B_error),
        }

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics

        Returns:
            Dict with timing and call count statistics
        """
        return {
            **self._perf_stats,
            "avg_forward_time": self._perf_stats["forward_time"]
            / max(1, self._perf_stats["forward_calls"]),
            "avg_linearization_time": self._perf_stats["linearization_time"]
            / max(1, self._perf_stats["linearization_calls"]),
        }

    def reset_performance_stats(self):
        """Reset performance counters"""
        for key in self._perf_stats:
            self._perf_stats[key] = 0.0 if "time" in key else 0

    def check_numerical_stability(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> Dict[str, Union[bool, float]]:
        """
        Check for numerical issues (NaN, Inf, extreme values)

        Args:
            x: State to check (any shape)
            u: Control to check (any shape)

        Returns:
            Dict with stability indicators
        """
        # Ensure proper shape
        x_2d = x.reshape(1, -1) if len(x.shape) <= 1 else x
        u_2d = u.reshape(1, -1) if len(u.shape) <= 1 else u

        with torch.no_grad():
            dx = self.forward(x_2d, u_2d)
            return {
                "has_nan": bool(torch.isnan(dx).any().item()),
                "has_inf": bool(torch.isinf(dx).any().item()),
                "max_derivative": float(dx.abs().max().item()),
                "is_stable": bool(not (torch.isnan(dx).any() or torch.isinf(dx).any())),
            }

    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"{self.__class__.__name__}("
            f"nx={self.nx}, nu={self.nu}, ny={self.ny}, order={self.order})"
        )

    def __str__(self) -> str:
        """Human-readable string representation"""
        return (
            f"{self.__class__.__name__}(nx={self.nx}, nu={self.nu}, order={self.order})"
        )

    def save_config(self, filename: str):
        """
        Save system configuration to file

        Args:
            filename: Path to save configuration (supports .json, .yaml, .pt)
        """

        config = {
            "class_name": self.__class__.__name__,
            "parameters": {str(k): float(v) for k, v in self.parameters.items()},
            "order": self.order,
            "nx": self.nx,
            "nu": self.nu,
            "ny": self.ny,
        }

        if filename.endswith(".json"):
            with open(filename, "w") as f:
                json.dump(config, f, indent=2)
        elif filename.endswith(".pt"):
            torch.save(config, filename)
        else:
            raise ValueError(f"Unsupported file format: {filename}. Use .json or .pt")

        print(f"Configuration saved to {filename}")

    def get_config_dict(self) -> Dict:
        """
        Get configuration as dictionary

        Returns:
            Dict with system configuration
        """
        return {
            "class_name": self.__class__.__name__,
            "parameters": {str(k): float(v) for k, v in self.parameters.items()},
            "order": self.order,
            "nx": self.nx,
            "nu": self.nu,
            "ny": self.ny,
        }

    def lqr_control(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        x_eq: Optional[torch.Tensor] = None,
        u_eq: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute LQR control gain for continuous-time system.

        **IMPORTANT**: This method linearizes the nonlinear system around the
        equilibrium point and computes the optimal gain for the linearized system.
        The resulting controller is:
        - Globally optimal for linear systems
        - Locally optimal near equilibrium for nonlinear systems
        - Performance degrades as state moves away from equilibrium

        Theory:
        ------
        Solves the continuous-time algebraic Riccati equation (CARE):
            A^T S + S A - S B R^{-1} B^T S + Q = 0

        The optimal gain is:
            K = -R^{-1} B^T S

        Control law:
            u(t) = K @ (x(t) - x_eq) + u_eq

        Cost function minimized (for linearized system):
            J = ∫[0,∞] [(x-x_eq)^T Q (x-x_eq) + (u-u_eq)^T R (u-u_eq)] dt

        Args:
            Q: State cost matrix (nx, nx). Must be positive semi-definite.
            Larger values penalize state deviations more heavily.
            R: Control cost matrix (nu, nu) or scalar for single input.
            Must be positive definite. Larger values penalize control effort.
            x_eq: Equilibrium state (uses self.x_equilibrium if None)
            u_eq: Equilibrium control (uses self.u_equilibrium if None)

        Returns:
            K: Control gain matrix (nu, nx). Control law is u = K @ (x - x_eq) + u_eq
            S: Solution to continuous-time Riccati equation (nx, nx)

        Raises:
            ValueError: If matrix dimensions are incompatible
            LinAlgError: If Riccati equation has no stabilizing solution

        Example:
            >>> # Design LQR for pendulum
            >>> pendulum = SymbolicPendulum(m=0.15, l=0.5, beta=0.1, g=9.81)
            >>> Q = np.diag([10.0, 1.0])  # Penalize angle more than velocity
            >>> R = np.array([[0.1]])      # Small control cost
            >>> K, S = pendulum.lqr_control(Q, R)
            >>>
            >>> # Apply control in simulation
            >>> controller = lambda x: K @ (x - pendulum.x_equilibrium) + pendulum.u_equilibrium

        Notes:
            - The linearization is computed at (x_eq, u_eq) using symbolic differentiation
            - For second-order systems, the full state-space linearization is used
            - The method assumes (x_eq, u_eq) is a valid equilibrium point
            - Stability is only guaranteed in a neighborhood of the equilibrium

        See Also:
            kalman_gain: Design optimal observer
            lqg_control: Combined controller and observer design
            linearized_dynamics: View the linearization used
        """
        if x_eq is None:
            x_eq = self.x_equilibrium
        if u_eq is None:
            u_eq = self.u_equilibrium

        # Ensure proper shape
        if len(x_eq.shape) == 1:
            x_eq = x_eq.unsqueeze(0)
        if len(u_eq.shape) == 1:
            u_eq = u_eq.unsqueeze(0)

        # Get linearized dynamics at equilibrium
        A, B = self.linearized_dynamics(x_eq, u_eq)
        A = A.squeeze().detach().cpu().numpy()
        B = B.squeeze().detach().cpu().numpy()

        # Ensure B is 2D (nx, nu)
        if B.ndim == 1:
            B = B.reshape(-1, 1)

        # Ensure R is 2D
        if isinstance(R, (int, float)):
            R = np.array([[R]])
        elif R.ndim == 1:
            R = np.diag(R)

        # Validate dimensions
        nx, nu = B.shape
        if A.shape != (nx, nx):
            raise ValueError(f"A must be ({nx}, {nx}), got {A.shape}")
        if Q.shape != (nx, nx):
            raise ValueError(f"Q must be ({nx}, {nx}), got {Q.shape}")
        if R.shape != (nu, nu):
            raise ValueError(f"R must be ({nu}, {nu}), got {R.shape}")

        # Solve continuous-time algebraic Riccati equation
        S = scipy.linalg.solve_continuous_are(A, B, Q, R)

        # Compute optimal gain
        K = -np.linalg.solve(R, B.T @ S)

        return K, S

    def kalman_gain(
        self,
        Q_process: Optional[np.ndarray] = None,
        R_measurement: Optional[np.ndarray] = None,
        x_eq: Optional[torch.Tensor] = None,
        u_eq: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Compute Kalman filter gain for continuous-time system.

        **IMPORTANT**: This method linearizes the nonlinear system around the
        equilibrium point and computes the optimal observer gain for the linearized
        system. The resulting observer is:
        - Globally optimal for linear systems with Gaussian noise
        - Locally optimal near equilibrium for nonlinear systems
        - Performance degrades as state moves away from equilibrium

        Theory:
        ------
        Solves the continuous-time dual Riccati equation:
            A P + P A^T - P C^T R^{-1} C P + Q = 0

        The optimal gain is:
            L = P C^T R^{-1}

        Observer dynamics:
            d x̂/dt = f(x̂, u) + L(y - h(x̂))

        For linearized system:
            d x̂/dt = A x̂ + B u + L(y - C x̂)

        Args:
            Q_process: Process noise covariance (nx, nx). Must be positive
                    semi-definite. Represents uncertainty in dynamics.
                    Default: 0.001 * I
            R_measurement: Measurement noise covariance (ny, ny) or scalar.
                        Must be positive definite. Represents sensor noise.
                        Default: 0.001 * I
            x_eq: Equilibrium state for linearization (uses self.x_equilibrium if None)
            u_eq: Equilibrium control for linearization (uses self.u_equilibrium if None)

        Returns:
            L: Kalman gain matrix (nx, ny). Observer correction term is L @ innovation

        Raises:
            ValueError: If matrix dimensions are incompatible
            LinAlgError: If dual Riccati equation has no stabilizing solution

        Example:
            >>> # Design Kalman filter for pendulum (measuring only angle)
            >>> pendulum = SymbolicPendulum(m=0.15, l=0.5, beta=0.1, g=9.81)
            >>> Q_process = np.diag([0.001, 0.01])      # Process noise
            >>> R_measurement = np.array([[0.1]])        # Measurement noise
            >>> L = pendulum.kalman_gain(Q_process, R_measurement)
            >>>
            >>> # Use in observer
            >>> observer = LinearObserver(pendulum, L)
            >>> observer.update(u, y_measured, dt=0.01)
            >>> x_estimate = observer.x_hat

        Notes:
            - The linearization is computed at (x_eq, u_eq) using symbolic differentiation
            - Q_process represents model uncertainty and unmodeled disturbances
            - R_measurement represents sensor noise characteristics
            - Larger Q_process → trust measurements more (higher gain)
            - Larger R_measurement → trust model more (lower gain)
            - The observer is guaranteed stable for the linearized system

        See Also:
            lqr_control: Design optimal controller
            lqg_control: Combined controller and observer design
            ExtendedKalmanFilter: Nonlinear state estimation
        """
        if Q_process is None:
            Q_process = np.eye(self.nx) * 1e-3
        if R_measurement is None:
            R_measurement = np.eye(self.ny) * 1e-3
        if x_eq is None:
            x_eq = self.x_equilibrium
        if u_eq is None:
            u_eq = self.u_equilibrium

        # Ensure proper shape
        if len(x_eq.shape) == 1:
            x_eq = x_eq.unsqueeze(0)

        # Get linearized dynamics
        A, _ = self.linearized_dynamics(
            x_eq, u_eq if len(u_eq.shape) > 1 else u_eq.unsqueeze(0)
        )
        A = A.squeeze().detach().cpu().numpy()

        C = self.linearized_observation(x_eq)
        C = C.squeeze().detach().cpu().numpy()

        # Ensure C is 2D (ny, nx)
        if C.ndim == 1:
            C = C.reshape(1, -1)

        # Ensure R_measurement is 2D
        if isinstance(R_measurement, (int, float)):
            R_measurement = np.array([[R_measurement]])
        elif R_measurement.ndim == 1:
            R_measurement = np.diag(R_measurement)

        # Validate dimensions
        nx = A.shape[0]
        ny = C.shape[0]

        if A.shape != (nx, nx):
            raise ValueError(f"A must be square, got {A.shape}")
        if C.shape[1] != nx:
            raise ValueError(f"C must have {nx} columns, got {C.shape}")
        if Q_process.shape != (nx, nx):
            raise ValueError(f"Q_process must be ({nx}, {nx}), got {Q_process.shape}")
        if R_measurement.shape != (ny, ny):
            raise ValueError(
                f"R_measurement must be ({ny}, {ny}), got {R_measurement.shape}"
            )

        # Solve continuous-time algebraic Riccati equation (dual problem)
        P = scipy.linalg.solve_continuous_are(A.T, C.T, Q_process, R_measurement)

        # Compute Kalman gain
        L = P @ C.T @ np.linalg.inv(R_measurement)

        return L

    def lqg_control(
        self,
        Q_lqr: np.ndarray,
        R_lqr: np.ndarray,
        Q_process: Optional[np.ndarray] = None,
        R_measurement: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute LQG controller (combined LQR controller + Kalman filter).

        **IMPORTANT**: This method designs an output feedback controller by combining:
        1. LQR controller for the linearized dynamics
        2. Kalman filter for the linearized observations

        The separation principle guarantees that for LINEAR systems:
        - Designing K and L separately is optimal
        - The closed-loop stability equals the product of controller/observer poles

        For NONLINEAR systems:
        - This provides a locally optimal solution near equilibrium
        - The separation principle does NOT hold globally
        - Performance degrades away from the equilibrium point
        - Consider adaptive or gain-scheduled approaches for large operating regions

        Theory:
        ------
        Output feedback control law:
            d x̂/dt = f(x̂, u) + L(y - h(x̂))    [Observer]
            u = K @ (x̂ - x_eq) + u_eq         [Controller based on estimate]

        For linearized system:
            Closed-loop poles = {eig(A + BK)} ∪ {eig(A - LC)}

        Args:
            Q_lqr: State cost for LQR (nx, nx)
            R_lqr: Control cost for LQR (nu, nu) or scalar
            Q_process: Process noise covariance (nx, nx). Default: 0.001 * I
            R_measurement: Measurement noise covariance (ny, ny) or scalar. Default: 0.001 * I

        Returns:
            K: LQR control gain (nu, nx)
            L: Kalman observer gain (nx, ny)

        Example:
            >>> # Design LQG controller for pendulum with noisy measurements
            >>> pendulum = SymbolicPendulum(m=0.15, l=0.5, beta=0.1, g=9.81)
            >>>
            >>> # Controller costs
            >>> Q_lqr = np.diag([10.0, 1.0])
            >>> R_lqr = np.array([[0.1]])
            >>>
            >>> # Noise covariances
            >>> Q_process = np.diag([0.001, 0.01])
            >>> R_measurement = np.array([[0.1]])
            >>>
            >>> K, L = pendulum.lqg_control(Q_lqr, R_lqr, Q_process, R_measurement)
            >>>
            >>> # Simulate with observer-based control
            >>> controller = LinearController(K, pendulum.x_equilibrium, pendulum.u_equilibrium)
            >>> observer = LinearObserver(pendulum, L)
            >>>
            >>> for t in range(steps):
            >>>     y = measure(x_true)  # Noisy measurement
            >>>     observer.update(u, y, dt)
            >>>     u = controller(observer.x_hat)
            >>>     x_true = step_dynamics(x_true, u, dt)

        Notes:
            - Separation principle: K and L can be designed independently (for linear systems)
            - The controller never sees the true state, only the estimate x̂
            - Closed-loop has 2*nx states: [x, x̂] (true state and estimate)
            - For nonlinear systems, consider EKF for the observer instead

        See Also:
            lqr_control: Controller design only
            kalman_gain: Observer design only
            lqg_closed_loop_matrix: Analyze closed-loop stability
            ExtendedKalmanFilter: Nonlinear observer alternative
        """
        K, _ = self.lqr_control(Q_lqr, R_lqr)
        L = self.kalman_gain(Q_process, R_measurement)
        return K, L

    def lqg_closed_loop_matrix(self, K: np.ndarray, L: np.ndarray) -> np.ndarray:
        """
        Compute closed-loop system matrix for LQG control.

        Returns the linearized dynamics of the augmented system [x, x̂] where:
        - x is the true state
        - x̂ is the observer's state estimate

        Theory:
        ------
        Closed-loop dynamics (linearized):
            d/dt [x  ]  = [A + BK    -BK   ] [x  ]
                 [x̂ ]   = [LC      A+BK-LC ] [x̂ ]

        Or equivalently in terms of state and estimation error e = x - x̂:
            d/dt [x]  = [A + BK   -BK ] [x]
                 [e]    [0      A - LC] [e]

        Eigenvalues:
            eig(A_cl) = {eig(A + BK)} ∪ {eig(A - LC)}

        This shows the separation principle: closed-loop poles are the union
        of controller poles and observer poles (for linear systems).

        Args:
            K: LQR control gain (nu, nx) from lqr_control()
            L: Kalman filter gain (nx, ny) from kalman_gain()

        Returns:
            A_cl: Closed-loop system matrix (2*nx, 2*nx)
                **State ordering**: [x[0], ..., x[nx-1], x̂[0], ..., x̂[nx-1]]
                    First nx elements: true state
                    Last nx elements:  estimate

        Example:
            >>> # Design LQG and analyze stability
            >>> K, L = system.lqg_control(Q_lqr, R_lqr, Q_process, R_measurement)
            >>> A_cl = system.lqg_closed_loop_matrix(K, L)
            >>>
            >>> # Check stability
            >>> eigenvalues = np.linalg.eigvals(A_cl)
            >>> is_stable = np.all(np.real(eigenvalues) < 0)
            >>> print(f"Closed-loop stable: {is_stable}")
            >>>
            >>> # Compare with open-loop
            >>> A, B = system.linearized_dynamics(x_eq, u_eq)
            >>> open_loop_eigs = np.linalg.eigvals(A)
            >>> print(f"Open-loop poles: {open_loop_eigs}")
            >>> print(f"Closed-loop poles: {eigenvalues}")

        Notes:
            - All eigenvalues should have negative real parts for stability
            - The matrix has block structure showing separation principle
            - Small entries (< 1e-6) are zeroed for numerical cleanliness
            - This is the linearized closed-loop; nonlinear behavior may differ

        See Also:
            lqg_control: Design the gains K and L
            eigenvalues_at_equilibrium: Open-loop eigenvalues
            is_stable_equilibrium: Check open-loop stability
        """
        x_eq = self.x_equilibrium.unsqueeze(0)
        u_eq = self.u_equilibrium.unsqueeze(0)

        A, B = self.linearized_dynamics(x_eq, u_eq)
        A = A.squeeze().detach().cpu().numpy()
        B = B.squeeze().detach().cpu().numpy()

        # Ensure B is 2D
        if B.ndim == 1:
            B = B.reshape(-1, 1)

        C = self.linearized_observation(x_eq).squeeze().detach().cpu().numpy()

        # Ensure C is 2D
        if C.ndim == 1:
            C = C.reshape(1, -1)

        # Ensure K is 2D (nu, nx)
        if K.ndim == 1:
            K = K.reshape(1, -1)

        # Ensure L is 2D (nx, ny)
        if L.ndim == 1:
            L = L.reshape(-1, 1)

        # Closed-loop system: [x, x̂]
        # dx/dt = Ax + B K x̂
        # dx̂/dt = A x̂ + B K x̂ + L(Cx - C x̂) = (A + B K - L C) x̂ + L C x
        A_cl = np.vstack(
            [
                np.hstack([A + B @ K, -B @ K]),  # dx/dt
                np.hstack([L @ C, A + B @ K - L @ C]),  # dx̂/dt
            ]
        )

        # Clean up near-zero entries
        A_cl[np.abs(A_cl) <= 1e-6] = 0

        return A_cl


class GenericDiscreteTimeSystem(nn.Module):
    """
    Generic discrete-time system for arbitrary order continuous systems.

    Automatically handles first-order, second-order, and higher-order systems
    using various numerical integration methods.

    Attributes:
        continuous_time_system: The underlying continuous-time system
        dt: Integration time step
        order: System order (inherited from continuous system)
        integration_method: Method for integrating derivatives
        position_integration: Method for integrating positions (order > 1)
    """

    def __init__(
        self,
        continuous_time_system: SymbolicDynamicalSystem,
        dt: float,
        integration_method: IntegrationMethod = IntegrationMethod.ExplicitEuler,
        position_integration: Optional[IntegrationMethod] = None,
    ):
        """
        Initialize discrete-time system wrapper

        Args:
            continuous_time_system: Symbolic dynamical system
            dt: Time step for discretization
            integration_method: Method for velocity/derivative integration
            position_integration: Method for position integration (order > 1)
        """
        super().__init__()

        # Validate continuous system
        if (
            not hasattr(continuous_time_system, "_initialized")
            or not continuous_time_system._initialized
        ):
            continuous_time_system._validate_system()

        self.continuous_time_system = continuous_time_system
        self.nx = continuous_time_system.nx
        self.nu = continuous_time_system.nu
        self.dt = float(dt)
        self.order = continuous_time_system.order
        self.integration_method = integration_method
        self.position_integration = position_integration or integration_method
        self.Ix = torch.eye(self.nx)

        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """

        Compute next state: x[k+1] = discrete_dynamics(x, u[k])
    
        **CRITICAL DISTINCTION**: This method returns the NEXT STATE x[k+1], NOT
        the derivative dx/dt.

        Args:
            x: Current state
            u: Control input

        Returns:
            Next state after one time step (same shape as input)
        """
        if self.order == 1:
            return self._integrate_first_order(x, u)
        elif self.order == 2:
            return self._integrate_second_order(x, u)
        else:
            return self._integrate_arbitrary_order(x, u)

    def __call__(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Make the system callable like a function"""
        return self.forward(x, u)

    def simulate(
        self,
        x0: torch.Tensor,
        controller: Optional[Union[torch.Tensor, Callable, torch.nn.Module]] = None,
        horizon: Optional[int] = None,
        return_controls: bool = False,
        return_all: bool = True,
        observer: Optional[Callable] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Simulate trajectory from initial state.

        Args:
            x0: Initial state (nx,) or (batch, nx)
            controller:
                - torch.Tensor: Control sequence (T, nu) - horizon inferred from T
                - Callable/nn.Module: Controller π(x) - horizon MUST be specified
                - None: Zero control - horizon MUST be specified
            horizon: Number of timesteps (required for controller functions)
            return_controls: If True, return (trajectory, controls)
            return_all: If True, return all states; if False, only final state
            observer: Optional observer x̂ = obs(x) for output feedback

        Returns:
            If return_controls=False:
                Trajectory: (T+1, nx) or (batch, T+1, nx) if return_all=True
                        (nx,) or (batch, nx) if return_all=False
            If return_controls=True:
                (Trajectory, Controls): Both same batch format

        Examples:
            >>> # 1. Pre-computed control sequence
            >>> u_seq = torch.zeros(100, 1)
            >>> traj = system.simulate(x0, controller=u_seq)

            >>> # 2. Neural network controller
            >>> controller_nn = NeuralNetworkController(...)
            >>> traj = system.simulate(x0, controller=controller_nn, horizon=100)

            >>> # 3. Lambda function controller
            >>> lqr_controller = lambda x: K @ (x - x_eq).T
            >>> traj = system.simulate(x0, controller=lqr_controller, horizon=100)

            >>> # 4. Output feedback with observer
            >>> traj = system.simulate(x0, controller=controller_nn,
            ...                        observer=observer_nn, horizon=100)

            >>> # 5. Return both trajectory and controls
            >>> traj, controls = system.simulate(x0, controller=controller_nn,
            ...                                  horizon=100, return_controls=True)
        """

        # Determine type
        is_control_sequence = isinstance(controller, torch.Tensor)

        if is_control_sequence:
            # Can infer T from sequence
            u_sequence = controller
            if len(u_sequence.shape) == 2:
                u_sequence = u_sequence.unsqueeze(0)
            T = u_sequence.shape[1]

            if horizon is not None and horizon != T:
                warnings.warn(
                    f"horizon={horizon} specified but control sequence has length {T}. "
                    f"Using sequence length T={T}."
                )

        else:
            # Controller function or None - MUST have horizon
            if horizon is None:
                raise ValueError(
                    "horizon must be specified when controller is a function or None.\n"
                    "Usage:\n"
                    "  - Control sequence: simulate(x0, controller=u_seq)  # horizon inferred\n"
                    "  - Controller func:  simulate(x0, controller=π, horizon=100)\n"
                    "  - Zero control:     simulate(x0, controller=None, horizon=100)"
                )
            T = horizon

            if controller is None:
                controller_func = lambda x: torch.zeros(
                    x.shape[0], self.nu, device=x.device
                )
            else:
                controller_func = controller

        # Handle dimensionality
        if len(x0.shape) == 1:
            x0 = x0.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        batch_size = x0.shape[0]

        # Determine if controller is a sequence or a function
        is_control_sequence = isinstance(controller, torch.Tensor)
        is_controller_function = callable(controller) or isinstance(
            controller, torch.nn.Module
        )

        if controller is None:
            # Zero control
            if horizon is None:
                raise ValueError("horizon must be specified when controller is None")
            T = horizon
            controller_func = lambda x: torch.zeros(
                x.shape[0], self.nu, device=x.device, dtype=x.dtype
            )
            is_controller_function = True
        elif is_control_sequence:
            # Pre-computed control sequence
            u_sequence = controller

            # Handle batch dimensions
            if len(u_sequence.shape) == 2:
                u_sequence = u_sequence.unsqueeze(0)

            # Expand to match batch size if needed
            if u_sequence.shape[0] == 1 and batch_size > 1:
                u_sequence = u_sequence.expand(batch_size, -1, -1)

            if u_sequence.shape[0] != batch_size:
                raise ValueError(
                    f"Control sequence batch size {u_sequence.shape[0]} "
                    f"doesn't match state batch size {batch_size}"
                )

            T = u_sequence.shape[1]

        elif is_controller_function:
            # Controller is a function or neural network
            if horizon is None:
                raise ValueError(
                    "horizon must be specified when controller is a callable"
                )
            T = horizon
            controller_func = controller
        else:
            raise TypeError(
                f"controller must be torch.Tensor, callable, nn.Module, or None. "
                f"Got {type(controller)}"
            )

        # Initialize storage
        if return_all:
            trajectory = [x0]
        if return_controls:
            controls = []

        x = x0

        # Simulation loop
        for t in range(T):
            if is_control_sequence:
                # Use pre-computed control
                u = u_sequence[:, t, :]
            else:
                # Compute control from current state
                if observer is not None:
                    # Output feedback: u = π(x̂) where x̂ = obs(x)
                    with torch.no_grad():
                        x_hat = observer(x)
                    u = controller_func(x_hat)
                else:
                    # State feedback: u = π(x)
                    u = controller_func(x)

                # Ensure proper shape
                if len(u.shape) == 1:
                    u = u.unsqueeze(0)
                elif len(u.shape) == 3:
                    u = u.squeeze(1)

            # Store control if requested
            if return_controls:
                controls.append(u)

            # Step forward
            x = self.forward(x, u)

            if return_all:
                trajectory.append(x)

        # Format outputs
        if return_all:
            result = torch.stack(trajectory, dim=1)  # (batch, T+1, nx)
            if squeeze_batch:
                result = result.squeeze(0)
        else:
            result = x.squeeze(0) if squeeze_batch else x

        if return_controls:
            controls_tensor = torch.stack(controls, dim=1)  # (batch, T, nu)
            if squeeze_batch:
                controls_tensor = controls_tensor.squeeze(0)
            return result, controls_tensor
        else:
            return result

    def _integrate_first_order(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Integrate first-order system: dx/dt = f(x, u)"""

        # Handle 1D vs 2D input
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        xdot = self.continuous_time_system.forward(x, u)

        if self.integration_method == IntegrationMethod.ExplicitEuler:
            x_next = x + xdot * self.dt
        elif self.integration_method == IntegrationMethod.MidPoint:
            k1 = xdot
            x_mid = x + 0.5 * self.dt * k1
            k2 = self.continuous_time_system.forward(x_mid, u)
            x_next = x + self.dt * k2
        elif self.integration_method == IntegrationMethod.RK4:
            k1 = xdot
            k2 = self.continuous_time_system.forward(x + 0.5 * self.dt * k1, u)
            k3 = self.continuous_time_system.forward(x + 0.5 * self.dt * k2, u)
            k4 = self.continuous_time_system.forward(x + self.dt * k3, u)
            x_next = x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise NotImplementedError(
                f"Integration method {self.integration_method} not implemented"
            )

        return x_next

    def _integrate_second_order(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Integrate second-order system: x = [q, qdot], qddot = f(x, u)
        """

        # Handle 1D vs 2D input
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        nq = self.continuous_time_system.nq
        q = x[:, :nq]
        qdot = x[:, nq:]

        # Compute acceleration at current state
        qddot = self.continuous_time_system.forward(x, u)

        # Validate acceleration shape
        if qddot.shape[1] != nq:
            if qddot.shape[1] == 1 and nq == 1:
                pass  # Correct
            else:
                raise ValueError(f"Expected qddot shape (*, {nq}), got {qddot.shape}")

        if self.integration_method == IntegrationMethod.ExplicitEuler:
            qdot_next = qdot + qddot * self.dt

        elif self.integration_method == IntegrationMethod.MidPoint:
            qdot_mid = qdot + 0.5 * self.dt * qddot
            x_mid = torch.cat([q, qdot_mid], dim=1)
            qddot_mid = self.continuous_time_system.forward(x_mid, u)
            qdot_next = qdot + self.dt * qddot_mid

        elif self.integration_method == IntegrationMethod.RK4:
            k1_vel = qddot
            qdot_stage2 = qdot + 0.5 * self.dt * k1_vel
            x_stage2 = torch.cat([q, qdot_stage2], dim=1)
            k2_vel = self.continuous_time_system.forward(x_stage2, u)
            qdot_stage3 = qdot + 0.5 * self.dt * k2_vel
            x_stage3 = torch.cat([q, qdot_stage3], dim=1)
            k3_vel = self.continuous_time_system.forward(x_stage3, u)
            qdot_stage4 = qdot + self.dt * k3_vel
            x_stage4 = torch.cat([q, qdot_stage4], dim=1)
            k4_vel = self.continuous_time_system.forward(x_stage4, u)

            # Combine stages
            qdot_next = qdot + (self.dt / 6.0) * (
                k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel
            )

        else:
            raise NotImplementedError(
                f"Integration method {self.integration_method} not implemented for 2nd order"
            )

        if self.position_integration == IntegrationMethod.ExplicitEuler:
            q_next = q + qdot * self.dt

        elif self.position_integration == IntegrationMethod.MidPoint:
            q_next = q + (qdot_next + qdot) / 2 * self.dt

        elif self.position_integration == IntegrationMethod.RK4:

            if self.integration_method == IntegrationMethod.RK4:
                k1_pos = qdot
                k2_pos = qdot_stage2
                k3_pos = qdot_stage3
                k4_pos = qdot_next  # Final velocity

                q_next = q + (self.dt / 6.0) * (
                    k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos
                )
            else:
                q_next = q + (qdot_next + qdot) / 2 * self.dt

        else:
            raise NotImplementedError(
                f"Position integration {self.position_integration} not implemented"
            )

        result = torch.cat([q_next, qdot_next], dim=1)

        # Squeeze back to 1D if input was 1D
        if squeeze_output:
            result = result.squeeze(0)

        return result

    def _integrate_arbitrary_order(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate arbitrary order system: x = [q, q', ..., q^(n-1)], q^(n) = f(x, u)
        """

        # Handle 1D vs 2D input
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        order = self.order
        nq = self.nx // order

        # Split state into derivative levels
        derivatives = [x[:, i * nq : (i + 1) * nq] for i in range(order)]
        highest_deriv = self.continuous_time_system.forward(x, u)

        derivatives_next = []

        if self.integration_method == IntegrationMethod.ExplicitEuler:
            for i in range(order - 1):
                derivatives_next.append(derivatives[i] + derivatives[i + 1] * self.dt)
            derivatives_next.append(derivatives[-1] + highest_deriv * self.dt)

        elif self.integration_method == IntegrationMethod.MidPoint:
            for i in range(order - 1):
                derivatives_next.append(derivatives[i] + self.dt * derivatives[i + 1])

            q_highest_mid = derivatives[-1] + 0.5 * self.dt * highest_deriv
            x_mid = torch.cat(derivatives[:-1] + [q_highest_mid], dim=1)
            highest_deriv_mid = self.continuous_time_system.forward(x_mid, u)
            derivatives_next.append(derivatives[-1] + self.dt * highest_deriv_mid)

        elif self.integration_method == IntegrationMethod.RK4:
            # Stage 1
            k1_derivs = derivatives[1:] + [highest_deriv]

            # Stage 2
            x_stage2 = [
                derivatives[i] + 0.5 * self.dt * k1_derivs[i] for i in range(order)
            ]
            x_mid_2 = torch.cat(x_stage2, dim=1)
            highest_deriv_2 = self.continuous_time_system.forward(x_mid_2, u)
            k2_derivs = x_stage2[1:] + [highest_deriv_2]

            # Stage 3
            x_stage3 = [
                derivatives[i] + 0.5 * self.dt * k2_derivs[i] for i in range(order)
            ]
            x_mid_3 = torch.cat(x_stage3, dim=1)
            highest_deriv_3 = self.continuous_time_system.forward(x_mid_3, u)
            k3_derivs = x_stage3[1:] + [highest_deriv_3]

            # Stage 4
            x_stage4 = [derivatives[i] + self.dt * k3_derivs[i] for i in range(order)]
            x_end = torch.cat(x_stage4, dim=1)
            highest_deriv_4 = self.continuous_time_system.forward(x_end, u)
            k4_derivs = x_stage4[1:] + [highest_deriv_4]

            # Combine
            for i in range(order):
                weighted = (
                    k1_derivs[i] + 2 * k2_derivs[i] + 2 * k3_derivs[i] + k4_derivs[i]
                ) / 6.0
                derivatives_next.append(derivatives[i] + self.dt * weighted)

        else:
            raise NotImplementedError(
                f"Integration method {self.integration_method} not implemented for order {order}"
            )

        result = torch.cat(derivatives_next, dim=1)

        # Squeeze back to 1D if input was 1D
        if squeeze_output:
            result = result.squeeze(0)

        return result

    def linearized_dynamics(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute linearized discrete dynamics using Euler approximation

        Returns:
            (Ad, Bd): Discrete-time linearized dynamics
        """
        Ac, Bc = self.continuous_time_system.linearized_dynamics(x, u)
        Ad = self.dt * Ac + self.Ix.to(x.device)
        Bd = self.dt * Bc
        return Ad, Bd

    def linearized_observation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute linearized observation matrix C = dh/dx

        For discrete-time systems, the observation is the same as continuous-time
        since h(x) doesn't depend on the discretization. The observation is with respect to
        state x and not time t.

        Args:
            x: State tensor (batch, nx) or (nx,)

        Returns:
            C: Observation Jacobian (batch, ny, nx) or (ny, nx)
        """
        return self.continuous_time_system.linearized_observation(x)

    def h(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate output equation: y = h(x)

        For discrete-time systems, the observation is the same as continuous-time
        since h(x) doesn't depend on the discretization. The observation is with respect to
        state x and not time t.

        Args:
            x: State tensor

        Returns:
            Output tensor
        """
        return self.continuous_time_system.h(x)

    @property
    def x_equilibrium(self) -> torch.Tensor:
        return self.continuous_time_system.x_equilibrium

    @property
    def u_equilibrium(self) -> torch.Tensor:
        return self.continuous_time_system.u_equilibrium

    def __repr__(self) -> str:
        return (
            f"GenericDiscreteTimeSystem({self.continuous_time_system.__class__.__name__}, "
            f"dt={self.dt}, method={self.integration_method.name})"
        )

    def __str__(self) -> str:
        """Human-readable string representation"""
        return (
            f"Discrete {self.continuous_time_system.__class__.__name__} "
            f"(dt={self.dt:.4f}, {self.integration_method.name})"
        )

    def dlqr_control(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        x_eq: Optional[torch.Tensor] = None,
        u_eq: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute discrete-time LQR control gain.

        **IMPORTANT**: This method linearizes the nonlinear discrete-time system
        around the equilibrium point and computes the optimal gain for the linearized
        system. The resulting controller is:
        - Globally optimal for linear discrete-time systems
        - Locally optimal near equilibrium for nonlinear systems
        - Performance degrades as state moves away from equilibrium

        Theory:
        ------
        Solves the discrete-time algebraic Riccati equation (DARE):
            S = Q + A^T S A - A^T S B (R + B^T S B)^{-1} B^T S A

        The optimal gain is:
            K = -(R + B^T S B)^{-1} B^T S A

        Control law:
            u[k] = K @ (x[k] - x_eq) + u_eq

        Cost function minimized (for linearized system):
            J = Σ[k=0,∞] [(x[k]-x_eq)^T Q (x[k]-x_eq) + (u[k]-u_eq)^T R (u[k]-u_eq)]

        Args:
            Q: State cost matrix (nx, nx). Must be positive semi-definite.
            R: Control cost matrix (nu, nu) or scalar. Must be positive definite.
            x_eq: Equilibrium state (uses self.x_equilibrium if None)
            u_eq: Equilibrium control (uses self.u_equilibrium if None)

        Returns:
            K: Discrete control gain matrix (nu, nx)
            S: Solution to discrete-time Riccati equation (nx, nx)

        Raises:
            ValueError: If matrix dimensions are incompatible
            LinAlgError: If DARE has no stabilizing solution

        Example:
            >>> # Create discrete-time system
            >>> pendulum_ct = SymbolicPendulum(m=0.15, l=0.5, beta=0.1, g=9.81)
            >>> pendulum_dt = GenericDiscreteTimeSystem(pendulum_ct, dt=0.01)
            >>>
            >>> # Design discrete LQR
            >>> Q = np.diag([10.0, 1.0])
            >>> R = np.array([[0.1]])
            >>> K, S = pendulum_dt.dlqr_control(Q, R)
            >>>
            >>> # Simulate
            >>> controller = lambda x: K @ (x - pendulum_dt.x_equilibrium) + pendulum_dt.u_equilibrium
            >>> trajectory = pendulum_dt.simulate(x0, controller, horizon=1000)

        Notes:
            - Linearization uses the discretization method specified in __init__
            - For second-order systems, the full discrete state-space form is used
            - Closed-loop eigenvalues should satisfy |λ| < 1 for stability
            - Discrete LQR often performs better than discretized continuous LQR

        See Also:
            lqr_control: Continuous-time version
            discrete_kalman_gain: Discrete observer design
            dlqg_control: Combined controller and observer
        """
        if x_eq is None:
            x_eq = self.x_equilibrium
        if u_eq is None:
            u_eq = self.u_equilibrium

        # Ensure proper shape
        if len(x_eq.shape) == 1:
            x_eq = x_eq.unsqueeze(0)
        if len(u_eq.shape) == 1:
            u_eq = u_eq.unsqueeze(0)

        # Get discrete linearized dynamics at equilibrium
        Ad, Bd = self.linearized_dynamics(x_eq, u_eq)
        Ad = Ad.squeeze().detach().cpu().numpy()
        Bd = Bd.squeeze().detach().cpu().numpy()

        # Ensure Bd is 2D
        if Bd.ndim == 1:
            Bd = Bd.reshape(-1, 1)

        # Ensure R is 2D
        if isinstance(R, (int, float)):
            R = np.array([[R]])
        elif R.ndim == 1:
            R = np.diag(R)

        # Solve discrete-time algebraic Riccati equation
        S = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)

        # Compute optimal gain
        K = -np.linalg.solve(R + Bd.T @ S @ Bd, Bd.T @ S @ Ad)

        return K, S

    def discrete_kalman_gain(
        self,
        Q_process: Optional[np.ndarray] = None,
        R_measurement: Optional[np.ndarray] = None,
        x_eq: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Compute discrete-time Kalman filter gain.

        **IMPORTANT**: This method linearizes the nonlinear discrete-time system
        and computes the optimal observer gain for the linearized system. For
        nonlinear systems, this provides local optimality near equilibrium.

        Theory:
        ------
        Solves the discrete-time dual Riccati equation:
            P = Q + A P A^T - A P C^T (C P C^T + R)^{-1} C P A^T

        The optimal gain is:
            L = P C^T (C P C^T + R)^{-1}

        Observer update:
            x̂[k|k-1] = f(x̂[k-1|k-1], u[k-1])    [Predict]
            x̂[k|k] = x̂[k|k-1] + L(y[k] - h(x̂[k|k-1]))    [Update]

        For linearized system:
            x̂[k+1|k] = A x̂[k|k] + B u[k]
            x̂[k|k] = x̂[k|k-1] + L(y[k] - C x̂[k|k-1])

        Args:
            Q_process: Process noise covariance (nx, nx). Default: 0.001 * I
            R_measurement: Measurement noise covariance (ny, ny) or scalar. Default: 0.001 * I
            x_eq: Equilibrium state for linearization (uses self.x_equilibrium if None)

        Returns:
            L: Kalman gain matrix (nx, ny)

        Example:
            >>> # Create discrete system
            >>> pendulum_ct = SymbolicPendulum(m=0.15, l=0.5, beta=0.1, g=9.81)
            >>> pendulum_dt = GenericDiscreteTimeSystem(pendulum_ct, dt=0.01)
            >>>
            >>> # Design Kalman filter
            >>> Q_process = np.diag([0.001, 0.01])
            >>> R_measurement = np.array([[0.1]])
            >>> L = pendulum_dt.discrete_kalman_gain(Q_process, R_measurement)
            >>>
            >>> # Use in simulation
            >>> observer = LinearObserver(pendulum_dt, L)
            >>> for k in range(steps):
            >>>     observer.update(u[k], y_measured[k], dt=pendulum_dt.dt)
            >>>     x_estimate = observer.x_hat

        Notes:
            - This is the steady-state Kalman gain (infinite horizon)
            - For time-varying gain, implement a Kalman filter loop
            - The gain balances prediction uncertainty and measurement noise
            - Linearization matches the integration method used for dynamics

        See Also:
            dlqr_control: Discrete controller design
            dlqg_control: Combined controller and observer
            ExtendedKalmanFilter: Nonlinear filtering
        """
        if Q_process is None:
            Q_process = np.eye(self.nx) * 1e-3
        if R_measurement is None:
            R_measurement = np.eye(self.continuous_time_system.ny) * 1e-3
        if x_eq is None:
            x_eq = self.x_equilibrium

        # Ensure proper shape
        if len(x_eq.shape) == 1:
            x_eq = x_eq.unsqueeze(0)

        # Get discrete linearized dynamics
        u_eq = self.u_equilibrium
        if len(u_eq.shape) == 1:
            u_eq = u_eq.unsqueeze(0)

        Ad, _ = self.linearized_dynamics(x_eq, u_eq)
        Ad = Ad.squeeze().detach().cpu().numpy()

        C = self.continuous_time_system.linearized_observation(x_eq)
        C = C.squeeze().detach().cpu().numpy()

        # Ensure C is 2D
        if C.ndim == 1:
            C = C.reshape(1, -1)

        # Ensure R_measurement is 2D
        if isinstance(R_measurement, (int, float)):
            R_measurement = np.array([[R_measurement]])
        elif R_measurement.ndim == 1:
            R_measurement = np.diag(R_measurement)

        # Solve discrete-time algebraic Riccati equation (dual problem)
        P = scipy.linalg.solve_discrete_are(Ad.T, C.T, Q_process, R_measurement)

        # Compute Kalman gain
        L = P @ C.T @ np.linalg.inv(C @ P @ C.T + R_measurement)

        return L

    def dlqg_control(
        self,
        Q_lqr: np.ndarray,
        R_lqr: np.ndarray,
        Q_process: Optional[np.ndarray] = None,
        R_measurement: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute discrete-time LQG controller (LQR + Kalman filter).

        **IMPORTANT**: Designs output feedback control for the linearized discrete-time
        system. The separation principle applies to linear systems but not globally
        to nonlinear systems.

        Theory:
        ------
        Discrete-time LQG control:
            x̂[k|k-1] = f(x̂[k-1|k-1], u[k-1])           [Predict]
            x̂[k|k] = x̂[k|k-1] + L(y[k] - h(x̂[k|k-1]))  [Correct]
            u[k] = K @ (x̂[k|k] - x_eq) + u_eq          [Control]

        Closed-loop eigenvalues (for linear system):
            eig(A_cl) = {eig(A + BK)} ∪ {eig(A - LC)}

        Args:
            Q_lqr: State cost for LQR (nx, nx)
            R_lqr: Control cost for LQR (nu, nu) or scalar
            Q_process: Process noise covariance (nx, nx). Default: 0.001 * I
            R_measurement: Measurement noise covariance (ny, ny) or scalar. Default: 0.001 * I

        Returns:
            K: Discrete LQR control gain (nu, nx)
            L: Discrete Kalman gain (nx, ny)

        Example:
            >>> # Create discrete system
            >>> quad_ct = SymbolicQuadrotor2D()
            >>> quad_dt = GenericDiscreteTimeSystem(quad_ct, dt=0.01,
            ...                                     integration_method=IntegrationMethod.RK4)
            >>>
            >>> # Design LQG controller
            >>> Q_lqr = np.diag([10, 10, 5, 1, 1, 1])  # Position > velocity
            >>> R_lqr = np.eye(2) * 0.1
            >>> Q_process = np.eye(6) * 0.01
            >>> R_measurement = np.eye(3) * 0.1  # Measure [x, y, theta]
            >>>
            >>> K, L = quad_dt.dlqg_control(Q_lqr, R_lqr, Q_process, R_measurement)
            >>>
            >>> # Simulate with observer feedback
            >>> controller = LinearController(K, quad_dt.x_equilibrium, quad_dt.u_equilibrium)
            >>> observer = LinearObserver(quad_dt, L)
            >>>
            >>> x = x0
            >>> for k in range(1000):
            >>>     y = quad_dt.h(x) + measurement_noise()
            >>>     observer.update(u, y, dt=quad_dt.dt)
            >>>     u = controller(observer.x_hat)
            >>>     x = quad_dt(x, u)

        Notes:
            - Separation principle: design K and L independently (for linear systems)
            - Discrete-time implementation more natural for digital control
            - Observer and controller run at the same rate (dt)
            - For different rates, use multirate control techniques

        See Also:
            dlqr_control: Controller only
            discrete_kalman_gain: Observer only
            dlqg_closed_loop_matrix: Closed-loop analysis
        """
        K, _ = self.dlqr_control(Q_lqr, R_lqr)
        L = self.discrete_kalman_gain(Q_process, R_measurement)
        return K, L

    def dlqg_closed_loop_matrix(self, K: np.ndarray, L: np.ndarray) -> np.ndarray:
        """
        Compute closed-loop discrete system matrix for LQG control.

        Returns the linearized dynamics of the augmented discrete-time system [x, x̂].

        Theory:
        ------
        Closed-loop discrete dynamics:
            [x[k+1]]  = [A + BK       -BK    ] [x[k]]
            [x̂[k+1]]  = [LC           A+BK-LC] [x̂[k]]

        Eigenvalues (for linear systems):
            eig(A_cl) = {eig(A + BK)} ∪ {eig(A - LC)}

        Stability condition:
            All eigenvalues must satisfy |λ| < 1

        Args:
            K: Discrete LQR gain (nu, nx) from dlqr_control()
            L: Discrete Kalman gain (nx, ny) from discrete_kalman_gain()

        Returns:
            A_cl: Closed-loop discrete system matrix (2*nx, 2*nx)

        Example:
            >>> # Design and analyze discrete LQG
            >>> K, L = system.dlqg_control(Q_lqr, R_lqr, Q_process, R_measurement)
            >>> A_cl = system.dlqg_closed_loop_matrix(K, L)
            >>>
            >>> # Check discrete stability
            >>> eigenvalues = np.linalg.eigvals(A_cl)
            >>> is_stable = np.all(np.abs(eigenvalues) < 1.0)
            >>> print(f"Closed-loop stable: {is_stable}")
            >>> print(f"Spectral radius: {np.max(np.abs(eigenvalues)):.4f}")
            >>>
            >>> # Visualize eigenvalues on unit circle
            >>> import matplotlib.pyplot as plt
            >>> theta = np.linspace(0, 2*np.pi, 100)
            >>> plt.plot(np.cos(theta), np.sin(theta), 'k--', label='Unit circle')
            >>> plt.plot(eigenvalues.real, eigenvalues.imag, 'rx', label='Poles')
            >>> plt.axis('equal')
            >>> plt.legend()
            >>> plt.show()

        Notes:
            - Eigenvalues inside unit circle → stable
            - Eigenvalues on unit circle → marginally stable
            - Eigenvalues outside unit circle → unstable
            - Small entries (< 1e-6) are zeroed for cleanliness

        See Also:
            dlqg_control: Design the gains
            output_feedback_lyapunov: Lyapunov stability analysis
        """
        x_eq = self.x_equilibrium.unsqueeze(0)
        u_eq = self.u_equilibrium.unsqueeze(0)

        Ad, Bd = self.linearized_dynamics(x_eq, u_eq)
        Ad = Ad.squeeze().detach().cpu().numpy()
        Bd = Bd.squeeze().detach().cpu().numpy()

        # Ensure Bd is 2D (nx, nu)
        if Bd.ndim == 1:
            Bd = Bd.reshape(-1, 1)

        C = self.continuous_time_system.linearized_observation(x_eq)
        C = C.squeeze().detach().cpu().numpy()

        # Ensure C is 2D (ny, nx)
        if C.ndim == 1:
            C = C.reshape(1, -1)

        # Ensure K is 2D (nu, nx)
        if K.ndim == 1:
            K = K.reshape(1, -1)

        # Ensure L is 2D (nx, ny)
        if L.ndim == 1:
            L = L.reshape(-1, 1)

        # Closed-loop discrete system: [x[k], x̂[k]]
        # x[k+1] = Ad @ x[k] + Bd @ K @ x̂[k]
        # x̂[k+1] = Ad @ x̂[k] + Bd @ K @ x̂[k] + L @ (C @ x[k] - C @ x̂[k])
        #         = (Ad + Bd @ K - L @ C) @ x̂[k] + L @ C @ x[k]
        A_cl = np.vstack(
            [
                np.hstack([Ad + Bd @ K, -Bd @ K]),  # x[k+1]
                np.hstack([L @ C, Ad + Bd @ K - L @ C]),  # x̂[k+1]
            ]
        )

        # Clean up near-zero entries
        A_cl[np.abs(A_cl) <= 1e-6] = 0

        return A_cl

    def output_feedback_lyapunov(self, K: np.ndarray, L: np.ndarray) -> np.ndarray:
        """
        Solve discrete-time Lyapunov equation for output feedback system.

        **IMPORTANT**: This solves the Lyapunov equation for the LINEARIZED
        closed-loop system around equilibrium. The resulting Lyapunov function
        V(z) = z^T S z proves:
        - Global asymptotic stability for LINEAR systems
        - LOCAL asymptotic stability for NONLINEAR systems (near equilibrium only)

        Finds positive definite matrix S satisfying:
            A_cl^T S A_cl - S + I = 0

        where A_cl is the LINEARIZED closed-loop system matrix.

        Args:
            K: Control gain (nu, nx)
            L: Observer gain (nx, ny)

        Returns:
            S: Solution to discrete Lyapunov equation (2*nx, 2*nx)
            Positive definite if linearized closed-loop is stable

        Example:
            >>> K, L = system.dlqg_control(Q_lqr, R_lqr, Q_process, R_measurement)
            >>> S = system.output_feedback_lyapunov(K, L)
            >>>
            >>> # Verify LOCAL stability
            >>> eigenvalues = np.linalg.eigvals(S)
            >>> is_positive_definite = np.all(eigenvalues > 0)
            >>> print(f"Linearized system locally stable: {is_positive_definite}")
            >>>
            >>> # WARNING: This doesn't estimate region of attraction!
            >>> # For nonlinear system, stability only guaranteed near equilibrium
            >>> def lyapunov_function(z):
            >>>     '''V(z) proves local stability, not global'''
            >>>     return z.T @ S @ z
            >>>
            >>> # To estimate region: sample many initial conditions
            >>> # and check which ones converge (Monte Carlo approach)

        Notes:
            - Positive definite S → linearized closed-loop is asymptotically stable
            - For nonlinear systems: only proves local stability
            - The linearization is computed at (x_eq, u_eq)
            - Does NOT provide region of attraction estimate

        See Also:
            dlqg_closed_loop_matrix: Get the linearized closed-loop matrix A_cl
            dlqg_control: Design the gains
        """

        A_cl = self.dlqg_closed_loop_matrix(K, L)
        S = control.dlyap(A_cl, np.eye(2 * self.nx))

        return S

    def print_info(
        self, include_equations: bool = True, include_linearization: bool = True
    ):
        """
        Print comprehensive information about the discrete-time system

        Args:
            include_equations: Whether to print symbolic equations
            include_linearization: Whether to print linearization at equilibrium
        """
        print("=" * 70)
        print(f"Discrete-Time System: {self.continuous_time_system.__class__.__name__}")
        print("=" * 70)

        # Basic info
        print(f"\nDiscretization:")
        print(f"  Time step (dt):        {self.dt}")
        print(f"  Integration method:    {self.integration_method.name}")
        if self.order > 1:
            print(f"  Position integration:  {self.position_integration.name}")

        print(f"\nDimensions:")
        print(f"  State dimension (nx):    {self.nx}")
        print(f"  Control dimension (nu):  {self.nu}")
        print(f"  Output dimension (ny):   {self.continuous_time_system.ny}")
        print(f"  System order:            {self.order}")
        if self.order > 1:
            print(f"  Generalized coords (nq): {self.continuous_time_system.nq}")

        print(f"\nEquilibrium:")
        x_eq = self.x_equilibrium.detach().cpu().numpy()
        u_eq = self.u_equilibrium.detach().cpu().numpy()
        print(f"  x_eq = {x_eq}")
        print(f"  u_eq = {u_eq}")

        # Symbolic equations from continuous system
        if include_equations:
            print("\n" + "-" * 70)
            print("Continuous-Time Dynamics (before discretization):")
            print("-" * 70)
            self.continuous_time_system.print_equations(simplify=True)

        # Linearization at equilibrium
        if include_linearization:
            print("\n" + "-" * 70)
            print("Linearization at Equilibrium:")
            print("-" * 70)

            # Continuous-time linearization
            Ac, Bc = self.continuous_time_system.linearized_dynamics(
                self.x_equilibrium.unsqueeze(0), self.u_equilibrium.unsqueeze(0)
            )
            Ac_np = Ac.squeeze().detach().cpu().numpy()
            Bc_np = Bc.squeeze().detach().cpu().numpy()

            print("Continuous-time (Ac, Bc):")
            print(f"  Ac =\n{Ac_np}")
            print(f"  Bc =\n{Bc_np}")

            # Discrete-time linearization
            Ad, Bd = self.linearized_dynamics(
                self.x_equilibrium.unsqueeze(0), self.u_equilibrium.unsqueeze(0)
            )
            Ad_np = Ad.squeeze().detach().cpu().numpy()
            Bd_np = Bd.squeeze().detach().cpu().numpy()

            print(f"\nDiscrete-time (Ad, Bd) with dt={self.dt}:")
            print(f"  Ad =\n{Ad_np}")
            print(f"  Bd =\n{Bd_np}")

            # Eigenvalues
            eigs_c = np.linalg.eigvals(Ac_np)
            eigs_d = np.linalg.eigvals(Ad_np)

            print(f"\nEigenvalues:")
            print(f"  Continuous: {eigs_c}")
            print(f"  Discrete:   {eigs_d}")
            print(f"\nStability:")
            print(f"  Continuous stable? {np.all(np.real(eigs_c) < 0)}")
            print(f"  Discrete stable?   {np.all(np.abs(eigs_d) < 1)}")

            # Observation matrix
            C = self.linearized_observation(self.x_equilibrium.unsqueeze(0))
            C_np = C.squeeze().detach().cpu().numpy()
            print(f"\nObservation matrix C:")
            print(f"  C =\n{C_np}")

        print("=" * 70)

    def summary(self) -> str:
        """
        Get a brief summary string

        Returns:
            Summary string with key system info
        """
        ct_stable = self.continuous_time_system.is_stable_equilibrium(
            discrete_time=False
        )

        # Check discrete stability
        eigs_d = np.linalg.eigvals(
            self.linearized_dynamics(
                self.x_equilibrium.unsqueeze(0), self.u_equilibrium.unsqueeze(0)
            )[0]
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        dt_stable = bool(np.all(np.abs(eigs_d) < 1))

        summary_str = (
            f"{self.continuous_time_system.__class__.__name__} "
            f"(nx={self.nx}, nu={self.nu}, ny={self.continuous_time_system.ny}, "
            f"order={self.order}, dt={self.dt:.4f}, {self.integration_method.name})\n"
            f"  Continuous stable: {ct_stable}, Discrete stable: {dt_stable}"
        )
        return summary_str

    def plot_trajectory(
        self,
        trajectory: torch.Tensor,
        state_names: Optional[List[str]] = None,
        control_sequence: Optional[torch.Tensor] = None,
        title: Optional[str] = None,
        save_html: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot trajectory using Plotly (interactive visualization)

        Args:
            trajectory: State trajectory (T, nx) or (batch, T, nx)
            state_names: Names for state variables (uses x0, x1, ... if None)
            control_sequence: Optional control inputs (T, nu) or (batch, T, nu)
            title: Plot title
            save_html: If provided, save interactive plot to this HTML file
            show: If True, display the plot
        """

        # Handle batched trajectories
        if len(trajectory.shape) == 3:
            batch_size = trajectory.shape[0]
            print(f"Plotting {batch_size} trajectories...")
        else:
            trajectory = trajectory.unsqueeze(0)
            batch_size = 1
            if control_sequence is not None:
                control_sequence = control_sequence.unsqueeze(0)

        # Convert to numpy
        traj_np = trajectory.detach().cpu().numpy()

        # Determine subplot layout
        has_control = control_sequence is not None
        num_plots = self.nx + (self.nu if has_control else 0)

        # Create subplots
        if num_plots <= 3:
            rows, cols = 1, num_plots
        elif num_plots <= 6:
            rows, cols = 2, (num_plots + 1) // 2
        else:
            rows = (num_plots + 2) // 3
            cols = 3

        # State names
        if state_names is None:
            state_names = [f"x{i}" for i in range(self.nx)]

        subplot_titles = state_names.copy()
        if has_control:
            control_names = [f"u{i}" for i in range(self.nu)]
            subplot_titles.extend(control_names)

        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

        # Time axis
        T = traj_np.shape[1]
        time_steps = np.arange(T) * self.dt

        # Plot states
        for i in range(self.nx):
            row = i // cols + 1
            col = i % cols + 1

            for b in range(batch_size):
                fig.add_trace(
                    go.Scatter(
                        x=time_steps,
                        y=traj_np[b, :, i],
                        mode="lines",
                        name=f"{state_names[i]}"
                        + (f" (batch {b})" if batch_size > 1 else ""),
                        showlegend=(batch_size > 1 or self.nx > 1),
                        line=dict(width=2),
                    ),
                    row=row,
                    col=col,
                )

            fig.update_xaxes(title_text="Time (s)", row=row, col=col)
            fig.update_yaxes(title_text=state_names[i], row=row, col=col)

        # Plot controls
        if has_control:
            control_np = control_sequence.detach().cpu().numpy()
            control_time = np.arange(control_np.shape[1]) * self.dt

            for i in range(self.nu):
                plot_idx = self.nx + i
                row = plot_idx // cols + 1
                col = plot_idx % cols + 1

                for b in range(batch_size):
                    fig.add_trace(
                        go.Scatter(
                            x=control_time,
                            y=control_np[b, :, i],
                            mode="lines",
                            name=f"u{i}" + (f" (batch {b})" if batch_size > 1 else ""),
                            showlegend=(batch_size > 1 or self.nu > 1),
                            line=dict(width=2, dash="dash"),
                        ),
                        row=row,
                        col=col,
                    )

                fig.update_xaxes(title_text="Time (s)", row=row, col=col)
                fig.update_yaxes(title_text=f"u{i}", row=row, col=col)

        # Update layout
        if title is None:
            title = f"{self.continuous_time_system.__class__.__name__} Trajectory"

        fig.update_layout(
            title=title, height=300 * rows, showlegend=True, hovermode="x unified"
        )

        # Save if requested
        if save_html:
            fig.write_html(save_html)
            print(f"Interactive plot saved to {save_html}")

        # Show if requested
        if show:
            fig.show()

        return fig

    def plot_phase_portrait_2d(
        self,
        trajectory: torch.Tensor,
        state_indices: Tuple[int, int] = (0, 1),
        state_names: Optional[Tuple[str, str]] = None,
        title: Optional[str] = None,
        save_html: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot 2D phase portrait

        Args:
            trajectory: State trajectory (T, nx) or (batch, T, nx)
            state_indices: Which two states to plot (default: first two)
            state_names: Names for the two states
            title: Plot title
            save_html: If provided, save to this HTML file
            show: If True, display the plot
        """

        # Handle batched trajectories
        if len(trajectory.shape) == 3:
            batch_size = trajectory.shape[0]
        else:
            trajectory = trajectory.unsqueeze(0)
            batch_size = 1

        traj_np = trajectory.detach().cpu().numpy()

        idx0, idx1 = state_indices
        if state_names is None:
            state_names = (f"x{idx0}", f"x{idx1}")

        fig = go.Figure()

        # Plot trajectories
        for b in range(batch_size):
            fig.add_trace(
                go.Scatter(
                    x=traj_np[b, :, idx0],
                    y=traj_np[b, :, idx1],
                    mode="lines+markers",
                    name=f"Trajectory {b}" if batch_size > 1 else "Trajectory",
                    line=dict(width=2),
                    marker=dict(size=4),
                )
            )

            # Mark start and end points
            fig.add_trace(
                go.Scatter(
                    x=[traj_np[b, 0, idx0]],
                    y=[traj_np[b, 0, idx1]],
                    mode="markers",
                    name="Start" if b == 0 else None,
                    marker=dict(size=12, color="green", symbol="star"),
                    showlegend=(b == 0),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[traj_np[b, -1, idx0]],
                    y=[traj_np[b, -1, idx1]],
                    mode="markers",
                    name="End" if b == 0 else None,
                    marker=dict(size=12, color="red", symbol="x"),
                    showlegend=(b == 0),
                )
            )

        # Mark equilibrium if it's 2D
        if self.nx >= 2:
            x_eq = self.continuous_time_system.x_equilibrium.detach().cpu().numpy()
            fig.add_trace(
                go.Scatter(
                    x=[x_eq[idx0]],
                    y=[x_eq[idx1]],
                    mode="markers",
                    name="Equilibrium",
                    marker=dict(size=15, color="black", symbol="diamond"),
                )
            )

        if title is None:
            title = f"{self.continuous_time_system.__class__.__name__} Phase Portrait"

        fig.update_layout(
            title=title,
            xaxis_title=state_names[0],
            yaxis_title=state_names[1],
            hovermode="closest",
            width=800,
            height=600,
        )

        if save_html:
            fig.write_html(save_html)
            print(f"Phase portrait saved to {save_html}")

        if show:
            fig.show()

        return fig


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter (EKF) for nonlinear state estimation.

    The EKF estimates the state of a nonlinear system by:
    1. Propagating state through NONLINEAR dynamics
    2. Re-linearizing at the current estimate each time step
    3. Using the linearization to propagate uncertainty
    4. Updating based on measurements

    Theory:
    ------
    **Predict Step**:
        x̂[k|k-1] = f(x̂[k-1|k-1], u[k-1])               [Nonlinear dynamics]
        A[k] = ∂f/∂x |_{x̂[k-1|k-1], u[k-1]}            [Linearization at current estimate]
        P[k|k-1] = A[k] P[k-1|k-1] A[k]^T + Q          [Covariance propagation]

    **Update Step**:
        ŷ[k|k-1] = h(x̂[k|k-1])                         [Nonlinear observation]
        C[k] = ∂h/∂x |_{x̂[k|k-1]}                      [Observation Jacobian]
        S[k] = C[k] P[k|k-1] C[k]^T + R                [Innovation covariance]
        K[k] = P[k|k-1] C[k]^T S[k]^{-1}               [Kalman gain - varies with time!]
        x̂[k|k] = x̂[k|k-1] + K[k](y[k] - ŷ[k|k-1])      [State update]
        P[k|k] = (I - K[k]C[k]) P[k|k-1]               [Covariance update]

    Attributes:
        system: SymbolicDynamicalSystem or GenericDiscreteTimeSystem
        Q: Process noise covariance (nx, nx)
        R: Measurement noise covariance (ny, ny)
        x_hat: Current state estimate (nx,)
        P: Current covariance estimate (nx, nx)
        is_discrete: Whether system is discrete or continuous

    Example:
        >>> # Create EKF for pendulum
        >>> pendulum = SymbolicPendulum(m=0.15, l=0.5, beta=0.1, g=9.81)
        >>> Q_process = np.diag([0.001, 0.01])
        >>> R_measurement = np.array([[0.1]])
        >>>
        >>> ekf = ExtendedKalmanFilter(pendulum, Q_process, R_measurement)
        >>>
        >>> # Initialize at origin
        >>> ekf.reset(x0=torch.tensor([0.1, 0.0]))
        >>>
        >>> # Estimation loop
        >>> for t in range(num_steps):
        >>>     # Predict
        >>>     ekf.predict(u[t], dt=0.01)
        >>>
        >>>     # Get noisy measurement
        >>>     y_measured = measure_angle(x_true[t]) + np.random.randn() * 0.1
        >>>
        >>>     # Update
        >>>     ekf.update(torch.tensor([y_measured]))
        >>>
        >>>     # Get estimate
        >>>     x_estimate = ekf.x_hat
        >>>     uncertainty = ekf.P
        >>>
        >>> # EKF can track large swings
        >>> # Unlike constant-gain observer which is only valid near equilibrium

    Notes:
        - Process noise Q represents model uncertainty and disturbances
        - Measurement noise R represents sensor characteristics
        - Larger Q → trust measurements more (higher gain)
        - Larger R → trust model more (lower gain)
        - Covariance P tracks estimate uncertainty
        - Can be used with nonlinear controllers (MPC, feedback linearization)

    See Also:
        kalman_gain: Constant-gain observer for linear systems
        LinearObserver: Linear observer with constant gain
        discrete_kalman_gain: Discrete-time constant-gain design
    """

    def __init__(self, system, Q_process: np.ndarray, R_measurement: np.ndarray):
        """
        Initialize Extended Kalman Filter.

        Args:
            system: SymbolicDynamicalSystem or GenericDiscreteTimeSystem
                   Must have forward(), h(), linearized_dynamics(), and
                   linearized_observation() methods
            Q_process: Process noise covariance (nx, nx). Represents model
                      uncertainty and unmodeled disturbances.
            R_measurement: Measurement noise covariance (ny, ny). Represents
                          sensor noise characteristics.

        Example:
            >>> system = SymbolicQuadrotor2D()
            >>> Q = np.eye(6) * 0.01  # Low process noise
            >>> R = np.eye(3) * 0.1   # Moderate measurement noise
            >>> ekf = ExtendedKalmanFilter(system, Q, R)
        """
        self.system = system
        self.Q = Q_process
        self.R = R_measurement

        # State estimate and covariance
        self.x_hat = system.x_equilibrium.clone()
        self.P = torch.eye(system.nx) * 0.1

        self.is_discrete = hasattr(system, "continuous_time_system")

    def predict(self, u: torch.Tensor, dt: Optional[float] = None):
        """
        EKF prediction step: propagate state estimate and covariance.

        Args:
            u: Control input (nu,)
            dt: Time step (required for continuous-time systems, ignored for discrete)

        Example:
            >>> ekf.predict(u=torch.tensor([1.0]), dt=0.01)
            >>> print(f"Predicted state: {ekf.x_hat}")
            >>> print(f"Predicted covariance: {ekf.P}")

        Notes:
            - Must call predict() before update() in each cycle
            - For discrete systems, dt is ignored
            - For continuous systems, uses Euler integration
            - Covariance grows during prediction (adds Q)
        """
        if self.is_discrete:
            # Discrete system: x̂[k+1|k] = f(x̂[k|k], u[k])
            with torch.no_grad():
                self.x_hat = self.system(self.x_hat, u)

            # Propagate covariance: P[k+1|k] = A P[k|k] A^T + Q
            A, _ = self.system.linearized_dynamics(
                self.x_hat.unsqueeze(0), u.unsqueeze(0)
            )
            A = A.squeeze()
        else:
            # Continuous system: integrate forward
            if dt is None:
                raise ValueError("dt required for continuous systems")

            with torch.no_grad():
                dx = self.system.forward(self.x_hat, u)
                self.x_hat = self.x_hat + dx * dt

            A, _ = self.system.linearized_dynamics(
                self.x_hat.unsqueeze(0), u.unsqueeze(0)
            )
            A = A.squeeze()
            A = torch.eye(self.system.nx) + A * dt  # Euler discretization

        Q_tensor = torch.tensor(self.Q, dtype=self.P.dtype, device=self.P.device)
        self.P = A @ self.P @ A.T + Q_tensor

    def update(self, y_measurement: torch.Tensor):
        """
        EKF update step: correct estimate using measurement.

        Args:
            y_measurement: Measurement vector (ny,). Should match the
                          output dimension of h(x).

        Example:
            >>> # After prediction
            >>> y_measured = torch.tensor([0.15, 2.1, 0.05])  # Noisy measurement
            >>> ekf.update(y_measured)
            >>> print(f"Updated state: {ekf.x_hat}")
            >>> print(f"Updated covariance: {ekf.P}")
            >>> print(f"Uncertainty reduced: {np.trace(ekf.P)}")

        Notes:
            - Covariance shrinks during update (information gained)
            - Large innovation → either bad estimate or bad measurement
            - Gain K[k] adapts based on current uncertainty P
            - Must call predict() before update()
        """
        # Ensure y_measurement is 1D
        if len(y_measurement.shape) == 0:
            y_measurement = y_measurement.unsqueeze(0)

        # Predicted measurement
        with torch.no_grad():
            if self.is_discrete:
                y_pred = self.system.continuous_time_system.h(
                    self.x_hat.unsqueeze(0)
                ).squeeze()
            else:
                y_pred = self.system.h(self.x_hat.unsqueeze(0)).squeeze()

        # Ensure y_pred is 1D
        if len(y_pred.shape) == 0:
            y_pred = y_pred.unsqueeze(0)

        # Measurement residual (innovation)
        innovation = y_measurement - y_pred
        if len(innovation.shape) == 0:
            innovation = innovation.unsqueeze(0)

        # Get measurement Jacobian
        if self.is_discrete:
            C = self.system.continuous_time_system.linearized_observation(
                self.x_hat.unsqueeze(0)
            ).squeeze()
        else:
            C = self.system.linearized_observation(self.x_hat.unsqueeze(0)).squeeze()

        # Ensure C is 2D (ny, nx)
        if len(C.shape) == 1:
            C = C.unsqueeze(0)  # (ny, nx)

        # Innovation covariance: S = C P C^T + R
        R_tensor = torch.tensor(self.R, dtype=self.P.dtype, device=self.P.device)
        S = C @ self.P @ C.mT + R_tensor  # Use .mT for matrix transpose

        # Ensure S is 2D
        if len(S.shape) == 0:
            S = S.unsqueeze(0).unsqueeze(0)
        elif len(S.shape) == 1:
            S = S.unsqueeze(0)

        # Kalman gain: K = P C^T S^{-1}
        Kt = self.P @ C.mT @ torch.inverse(S)  # (nx, ny)

        # Update state estimate: x̂ = x̂ + K * innovation
        correction = (Kt @ innovation.unsqueeze(-1)).squeeze(-1)
        self.x_hat = self.x_hat + correction

        # Update covariance: P = (I - K C) P
        nx = (
            self.system.nx
            if not self.is_discrete
            else self.system.continuous_time_system.nx
        )
        I = torch.eye(nx, device=self.P.device, dtype=self.P.dtype)
        self.P = (I - Kt @ C) @ self.P

    def reset(
        self, x0: Optional[torch.Tensor] = None, P0: Optional[torch.Tensor] = None
    ):
        """
        Reset filter to initial state and covariance.

        Useful for:
        - Starting a new estimation sequence
        - Recovering from filter divergence
        - Testing different initial conditions

        Args:
            x0: Initial state estimate (nx,). Uses equilibrium if None.
            P0: Initial covariance (nx, nx). Uses 0.1*I if None.

        Example:
            >>> # Reset to known initial condition
            >>> ekf.reset(x0=torch.tensor([0.1, 0.0]),
            ...           P0=torch.eye(2) * 0.01)  # Low initial uncertainty
            >>>
            >>> # Reset to equilibrium with high uncertainty
            >>> ekf.reset()  # Uses default x_equilibrium and 0.1*I

        Notes:
            - Called automatically in __init__()
            - P0 represents initial uncertainty about x0
            - Larger P0 → less confident in initial estimate
            - After reset, start with predict() then update()
        """
        if x0 is not None:
            self.x_hat = x0.clone()
        else:
            if self.is_discrete:
                self.x_hat = self.system.x_equilibrium.clone()
            else:
                self.x_hat = self.system.x_equilibrium.clone()

        if P0 is not None:
            self.P = P0.clone()
        else:
            nx = (
                self.system.nx
                if not self.is_discrete
                else self.system.continuous_time_system.nx
            )
            self.P = torch.eye(nx) * 0.1


class LinearController:
    """
    Linear state feedback controller with equilibrium offset.

    Implements the control law:
        u(x) = K @ (x - x_eq) + u_eq

    where:
    - K is the control gain matrix
    - x_eq is the equilibrium/reference state
    - u_eq is the equilibrium/feedforward control

    Valid Region:
    ------------
    - **Linear systems**: Globally valid
    - **Nonlinear systems**: Valid near equilibrium where linearization holds
    - Performance degrades as ||x - x_eq|| increases

    Attributes:
        K: Control gain matrix (nu, nx)
        x_eq: Equilibrium/reference state (nx,)
        u_eq: Equilibrium/feedforward control (nu,)

    Example - LQR Control:
        >>> # Design LQR controller
        >>> system = SymbolicPendulum(m=1.0, l=0.5)
        >>> Q = np.diag([10.0, 1.0])
        >>> R = np.array([[0.1]])
        >>> K, S = system.lqr_control(Q, R)
        >>>
        >>> # Create controller
        >>> controller = LinearController(K, system.x_equilibrium, system.u_equilibrium)
        >>>
        >>> # Use in simulation
        >>> x = torch.tensor([0.1, 0.0])  # Small deviation from equilibrium
        >>> u = controller(x)
        >>> print(f"Control: {u}")

    Example - Tracking Reference:
        >>> # Track different equilibrium
        >>> x_ref = torch.tensor([0.5, 0.0])  # Desired position
        >>> u_ref = compute_feedforward(x_ref)
        >>>
        >>> # Controller drives x → x_ref
        >>> tracking_controller = LinearController(K, x_ref, u_ref)
        >>>
        >>> # In control loop
        >>> for t in range(steps):
        >>>     u = tracking_controller(x[t])
        >>>     x[t+1] = system(x[t], u)

    Example - Output Feedback (with Observer):
        >>> # Design LQG
        >>> K, L = system.lqg_control(Q_lqr, R_lqr, Q_process, R_measurement)
        >>>
        >>> # Controller uses state estimate, not true state
        >>> controller = LinearController(K, system.x_equilibrium, system.u_equilibrium)
        >>> observer = LinearObserver(system, L)
        >>>
        >>> for t in range(steps):
        >>>     y = measure(x_true[t])
        >>>     observer.update(u, y, dt)
        >>>     u = controller(observer.x_hat)  # Use estimate!
        >>>     x_true[t+1] = system(x_true[t], u)

    Example - Gain Scheduling:
        >>> # Different gains for different regions
        >>> K_upright = system.lqr_control(Q1, R)[0]  # Near upright
        >>> K_hanging = system.lqr_control(Q2, R)[0]  # Near hanging
        >>>
        >>> controller_upright = LinearController(K_upright, x_eq_up, u_eq_up)
        >>> controller_hanging = LinearController(K_hanging, x_eq_down, u_eq_down)
        >>>
        >>> # Switch based on state
        >>> def adaptive_control(x):
        >>>     if abs(x[0]) < np.pi/4:  # Near upright
        >>>         return controller_upright(x)
        >>>     else:
        >>>         return controller_hanging(x)

    Notes:
        - The feedforward term u_eq ensures u=u_eq at x=x_eq
        - For tracking time-varying references, update x_eq and u_eq online
        - Can be used with state estimation (observer-based control)
        - Handles both 1D and batched inputs automatically

    See Also:
        lqr_control: Design optimal K matrix
        LinearObserver: State estimation for output feedback
        lqg_control: Combined controller and observer design
    """

    def __init__(self, K: np.ndarray, x_eq: torch.Tensor, u_eq: torch.Tensor):
        """
        Initialize linear state feedback controller.

        Args:
            K: Control gain matrix (nu, nx).
            x_eq: Equilibrium/reference state (nx,)
            u_eq: Equilibrium/feedforward control (nu,)

        Example:
            >>> K = np.array([[-12.5, -5.0]])  # SISO system
            >>> x_eq = torch.zeros(2)
            >>> u_eq = torch.zeros(1)
            >>> controller = LinearController(K, x_eq, u_eq)
        """
        self.K = torch.tensor(K, dtype=torch.float32)
        self.x_eq = x_eq
        self.u_eq = u_eq

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute control input: u = K @ (x - x_eq) + u_eq

        Args:
            x: Current state (nx,) or (batch, nx)

        Returns:
            u: Control input (nu,) or (batch, nu)

        Example:
            >>> x = torch.tensor([0.1, 0.05])
            >>> u = controller(x)
            >>> print(f"Control: {u}")
            >>>
            >>> # Batched computation
            >>> x_batch = torch.randn(100, 2)  # 100 states
            >>> u_batch = controller(x_batch)  # 100 controls

        Notes:
            - Automatically handles 1D or 2D inputs
            - Returns same batch structure as input
            - All operations differentiable (can be used in learning)
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        u = self.u_eq + (self.K @ (x - self.x_eq).T).T

        if squeeze:
            u = u.squeeze(0)

        return u

    def to(self, device):
        """
        Move controller to specified device (CPU/GPU).

        Args:
            device: torch.device or string ('cpu', 'cuda')

        Returns:
            Self for chaining
        """
        self.K = self.K.to(device)
        self.x_eq = self.x_eq.to(device)
        self.u_eq = self.u_eq.to(device)
        return self


class LinearObserver:
    """
    Linear state observer with constant gain.

    Implements the observer:
        d x̂/dt = f(x̂, u) + L(y - h(x̂))                         [Continuous-time]
        x̂[k+1] = f(x̂[k], u[k]) + L(y[k+1] - h(f(x̂[k], u[k])))  [Discrete-time]

    where:
    - x̂ is the state estimate
    - L is the observer gain matrix
    - y is the measurement
    - h(x̂) is the predicted measurement

    Attributes:
        system: SymbolicDynamicalSystem or GenericDiscreteTimeSystem
        L: Observer gain matrix (nx, ny)
        x_hat: Current state estimate (nx,)

    Example - Continuous-Time Observer:
        >>> # Design Kalman gain
        >>> system = SymbolicPendulum(m=1.0, l=0.5)
        >>> Q_process = np.diag([0.001, 0.01])
        >>> R_measurement = np.array([[0.1]])
        >>> L = system.kalman_gain(Q_process, R_measurement)
        >>>
        >>> # Create observer
        >>> observer = LinearObserver(system, L)
        >>> observer.reset(x0=torch.zeros(2))
        >>>
        >>> # Observer loop
        >>> dt = 0.01
        >>> for t in range(num_steps):
        >>>     observer.update(u[t], y_measured[t], dt)
        >>>     x_estimate = observer.x_hat

    Example - Discrete-Time Observer:
        >>> # Create discrete system
        >>> system_ct = SymbolicQuadrotor2D()
        >>> system_dt = GenericDiscreteTimeSystem(system_ct, dt=0.01)
        >>>
        >>> # Design discrete Kalman gain
        >>> L = system_dt.discrete_kalman_gain(Q_process, R_measurement)
        >>>
        >>> # Create observer
        >>> observer = LinearObserver(system_dt, L)
        >>>
        >>> # Observer loop (no dt needed for discrete)
        >>> for k in range(num_steps):
        >>>     observer.update(u[k], y[k], dt=None)  # dt ignored for discrete
        >>>     x_estimate = observer.x_hat

    Example - Output Feedback Control:
        >>> # Design LQG controller
        >>> K, L = system.lqg_control(Q_lqr, R_lqr, Q_process, R_measurement)
        >>>
        >>> # Create controller and observer
        >>> controller = LinearController(K, system.x_equilibrium, system.u_equilibrium)
        >>> observer = LinearObserver(system, L)
        >>>
        >>> # Closed-loop with output feedback
        >>> x_true = x0
        >>> observer.reset(x0=system.x_equilibrium)  # Start from equilibrium guess
        >>>
        >>> for t in range(num_steps):
        >>>     # Measure (with noise)
        >>>     y = system.h(x_true) + torch.randn(system.ny) * 0.1
        >>>
        >>>     # Update observer
        >>>     observer.update(u, y, dt)
        >>>
        >>>     # Compute control based on estimate
        >>>     u = controller(observer.x_hat)
        >>>
        >>>     # Update true system
        >>>     x_true = system(x_true, u) if discrete else integrate(x_true, u, dt)

    Notes:
        - Gain L is constant and designed at equilibrium
        - For nonlinear systems, only accurate near equilibrium
        - No covariance tracking
        - Can be used with both continuous and discrete systems
        - Lower computational cost than EKF

    See Also:
        kalman_gain: Design L for continuous systems
        discrete_kalman_gain: Design L for discrete systems
        ExtendedKalmanFilter: Adaptive nonlinear observer
        LinearController: State feedback controller
        lqg_control: Combined controller and observer design
    """

    def __init__(self, system, L: np.ndarray):
        """
        Initialize linear observer with constant gain.

        Args:
            system: SymbolicDynamicalSystem or GenericDiscreteTimeSystem
            L: Observer gain matrix (nx, ny). Typically from kalman_gain()
               or discrete_kalman_gain().

        Example:
            >>> L = system.kalman_gain(Q_process, R_measurement)
            >>> observer = LinearObserver(system, L)
        """
        self.system = system
        self.L = torch.tensor(L, dtype=torch.float32)
        self.x_hat = system.x_equilibrium.clone()

    def update(self, u: torch.Tensor, y: torch.Tensor, dt: float):
        """
        Update observer state estimate.

        Continuous-time:
            d x̂/dt = f(x̂, u) + L(y - h(x̂))
            x̂_new ≈ x̂_old + dt * [f(x̂, u) + L(y - h(x̂))]

        Discrete-time:
            x̂_pred = f(x̂, u)
            x̂_new = x̂_pred + L(y - h(x̂_pred))

        Args:
            u: Control input (nu,)
            y: Measurement (ny,)
            dt: Time step (used for continuous systems, ignored for discrete)

        Example:
            >>> u = torch.tensor([1.0])
            >>> y_measured = torch.tensor([0.15])  # Noisy angle measurement
            >>> observer.update(u, y_measured, dt=0.01)
            >>> print(f"Estimate: {observer.x_hat}")

        Notes:
            - For continuous systems: uses Euler integration
            - For discrete systems: dt parameter is ignored
            - Innovation = y - h(x̂_pred) is the measurement residual
            - Large innovation suggests either bad estimate or bad measurement
            - Gain L determines how much to trust innovation vs model
        """
        # Predict
        with torch.no_grad():
            if hasattr(self.system, "continuous_time_system"):
                # Discrete system
                x_pred = self.system(self.x_hat.unsqueeze(0), u.unsqueeze(0)).squeeze(0)
                y_pred = self.system.continuous_time_system.h(
                    x_pred.unsqueeze(0)
                ).squeeze(0)
            else:
                # Continuous system
                dx = self.system.forward(
                    self.x_hat.unsqueeze(0), u.unsqueeze(0)
                ).squeeze(0)
                x_pred = self.x_hat + dx * dt
                y_pred = self.system.h(x_pred.unsqueeze(0)).squeeze(0)

            # Correct
            innovation = y - y_pred
            self.x_hat = x_pred + (self.L @ innovation.unsqueeze(-1)).squeeze(-1)

    def reset(self, x0: Optional[torch.Tensor] = None):
        """
        Reset observer to initial state estimate.

        Args:
            x0: Initial state estimate (nx,). Uses equilibrium if None.

        Example:
            >>> # Reset to known initial condition
            >>> observer.reset(x0=torch.tensor([0.1, 0.0]))
            >>>
            >>> # Reset to equilibrium
            >>> observer.reset()  # Uses system.x_equilibrium

        Notes:
            - Called automatically in __init__()
            - Unlike EKF, no covariance to reset
            - Start observer from best available estimate
        """
        if x0 is not None:
            self.x_hat = x0.clone()
        else:
            self.x_hat = self.system.x_equilibrium.clone()

    def to(self, device):
        """
        Move observer to specified device (CPU/GPU).

        Args:
            device: torch.device or string ('cpu', 'cuda')

        Returns:
            Self for chaining
        """
        self.L = self.L.to(device)
        self.x_hat = self.x_hat.to(device)
        return self
