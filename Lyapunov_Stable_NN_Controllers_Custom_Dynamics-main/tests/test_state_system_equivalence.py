"""Test that all symbolic systems match their hardcoded equivalents"""
import torch
import neural_lyapunov_training.dynamical_system as ds
import neural_lyapunov_training.symbolic_dynamics as sd
import neural_lyapunov_training.symbolic_systems as ss
from neural_lyapunov_training.pendulum import PendulumDynamics
from neural_lyapunov_training.quadrotor2d import Quadrotor2DDynamics

def test_pendulum():
    """Test pendulum symbolic vs hardcoded"""
    print("\n" + "="*70)
    print("TESTING PENDULUM")
    print("="*70)
    
    # Hardcoded
    pend_hard = PendulumDynamics(m=0.15, l=0.5, beta=0.1, g=9.81)
    dyn_hard = ds.SecondOrderDiscreteTimeSystem(
        pend_hard, dt=0.01,
        position_integration=ds.IntegrationMethod.ExplicitEuler,
        velocity_integration=ds.IntegrationMethod.ExplicitEuler,
    )
    
    # Symbolic
    pend_sym = ss.SymbolicPendulum2ndOrder(m=0.15, l=0.5, beta=0.1, g=9.81)
    dyn_sym = ds.SecondOrderDiscreteTimeSystem(
        pend_sym, dt=0.01,
        position_integration=ds.IntegrationMethod.ExplicitEuler,
        velocity_integration=ds.IntegrationMethod.ExplicitEuler,
    )
    
    # Test
    x0 = torch.tensor([[0.5, 0.0]])
    u = torch.zeros(1, 1)
    
    # Single step
    x1_hard = dyn_hard.forward(x0, u)
    x1_sym = dyn_sym.forward(x0, u)
    diff_single = (x1_hard - x1_sym).abs().max().item()
    
    # Multi-step
    x_hard, x_sym = x0.clone(), x0.clone()
    for _ in range(100):
        x_hard = dyn_hard.forward(x_hard, u)
        x_sym = dyn_sym.forward(x_sym, u)
    diff_multi = (x_hard - x_sym).abs().max().item()
    
    print(f"Single step difference: {diff_single:.2e}")
    print(f"100 steps difference:   {diff_multi:.2e}")
    
    if diff_multi < 1e-6:
        print("✅ PASS: Pendulum implementations match")
        return True
    else:
        print("❌ FAIL: Pendulum implementations differ")
        return False

def test_quadrotor():
    """Test quadrotor symbolic vs hardcoded"""
    print("\n" + "="*70)
    print("TESTING QUADROTOR 2D")
    print("="*70)
    
    # Hardcoded
    quad_hard = Quadrotor2DDynamics(
        length=0.25, mass=0.486, inertia=0.00383, g=9.81
    )
    dyn_hard = ds.SecondOrderDiscreteTimeSystem(
        quad_hard, dt=0.01,
        position_integration=ds.IntegrationMethod.ExplicitEuler,
        velocity_integration=ds.IntegrationMethod.ExplicitEuler,
    )
    
    # Symbolic
    quad_sym = ss.SymbolicQuadrotor2DState(
        length=0.25, mass=0.486, inertia=0.00383, gravity=9.81
    )
    dyn_sym = ds.SecondOrderDiscreteTimeSystem(
        quad_sym, dt=0.01,
        position_integration=ds.IntegrationMethod.ExplicitEuler,
        velocity_integration=ds.IntegrationMethod.ExplicitEuler,
    )
    
    # Test
    x0 = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
    u = quad_hard.u_equilibrium.unsqueeze(0)
    
    # Single step
    x1_hard = dyn_hard.forward(x0, u)
    x1_sym = dyn_sym.forward(x0, u)
    diff_single = (x1_hard - x1_sym).abs().max().item()
    
    # Multi-step
    x_hard, x_sym = x0.clone(), x0.clone()
    for _ in range(100):
        x_hard = dyn_hard.forward(x_hard, u)
        x_sym = dyn_sym.forward(x_sym, u)
    diff_multi = (x_hard - x_sym).abs().max().item()
    
    print(f"Single step difference: {diff_single:.2e}")
    print(f"100 steps difference:   {diff_multi:.2e}")
    
    if diff_multi < 1e-5:  # Slightly higher tolerance for 6D system
        print("✅ PASS: Quadrotor implementations match")
        return True
    else:
        print("❌ FAIL: Quadrotor implementations differ")
        print(f"   Hardcoded final: {x_hard.numpy()}")
        print(f"   Symbolic final:  {x_sym.numpy()}")
        return False

if __name__ == "__main__":
    results = []
    results.append(test_pendulum())
    results.append(test_quadrotor())
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if all(results):
        print("✅ ALL TESTS PASSED")
        exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        exit(1)