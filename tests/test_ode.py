"""Tests for ODE solvers."""

import pytest
import mlx.core as mx


class TestEulerSolver:
    """Tests for Euler ODE solver."""

    def test_linear_ode(self):
        """Test solving a simple linear ODE: dx/dt = 1."""
        from heartlib_mlx.ode.solver import euler_solve

        def velocity_fn(t, x, condition):
            return mx.ones_like(x)

        x0 = mx.zeros((1, 10))
        condition = mx.zeros((1, 10))

        x1 = euler_solve(
            velocity_fn=velocity_fn,
            x0=x0,
            condition=condition,
            t_start=0.0,
            t_end=1.0,
            num_steps=100,
        )

        # x(1) should be close to 1
        assert mx.allclose(x1, mx.ones_like(x1), atol=0.02)

    def test_cfg_application(self):
        """Test classifier-free guidance is applied correctly."""
        from heartlib_mlx.ode.solver import euler_solve

        def velocity_fn(t, x, condition):
            return condition  # Use condition as velocity

        x0 = mx.zeros((1, 10))
        cond = mx.ones((1, 10))
        uncond = mx.zeros((1, 10))

        # With CFG scale = 2, velocity = uncond + 2 * (cond - uncond) = 2 * cond
        x1 = euler_solve(
            velocity_fn=velocity_fn,
            x0=x0,
            condition=cond,
            t_start=0.0,
            t_end=1.0,
            num_steps=100,
            guidance_scale=2.0,
            uncond=uncond,
        )

        # x(1) should be close to 2
        assert mx.allclose(x1, mx.ones_like(x1) * 2, atol=0.05)


class TestMidpointSolver:
    """Tests for Midpoint ODE solver."""

    def test_quadratic_ode(self):
        """Test solving dx/dt = 2t (solution: x = t^2)."""
        from heartlib_mlx.ode.solver import midpoint_solve

        def velocity_fn(t, x, condition):
            return mx.ones_like(x) * float(2 * t[0])

        x0 = mx.zeros((1, 1))
        condition = mx.zeros((1, 1))

        x1 = midpoint_solve(
            velocity_fn=velocity_fn,
            x0=x0,
            condition=condition,
            t_start=0.0,
            t_end=1.0,
            num_steps=100,
        )

        # x(1) should be close to 1
        assert mx.allclose(x1, mx.ones_like(x1), atol=0.05)


class TestNeuralODE:
    """Tests for NeuralODE wrapper."""

    def test_sample(self):
        """Test sampling from a simple velocity model."""
        from heartlib_mlx.ode.neural_ode import NeuralODE
        import mlx.nn as nn

        # Simple linear velocity model
        class LinearVelocity(nn.Module):
            def __call__(self, t, x, condition):
                return condition - x

        model = LinearVelocity()
        ode = NeuralODE(model, solver="euler", num_steps=100)

        x0 = mx.zeros((1, 10))
        condition = mx.ones((1, 10))

        x1 = ode.sample(x0, condition)

        # Should approach condition (exponential decay towards target)
        assert mx.all(x1 > 0.5)


class TestFlowMatchingScheduler:
    """Tests for FlowMatchingScheduler."""

    def test_interpolate(self):
        """Test that interpolation works correctly."""
        from heartlib_mlx.ode.neural_ode import FlowMatchingScheduler

        scheduler = FlowMatchingScheduler()

        x0 = mx.zeros((2, 10))
        x1 = mx.ones((2, 10))

        # t=0 should give x0
        t = mx.array([0.0, 0.0])
        xt = scheduler.interpolate(x0, x1, t)
        assert mx.allclose(xt, x0, atol=1e-6)

        # t=1 should give x1
        t = mx.array([1.0, 1.0])
        xt = scheduler.interpolate(x0, x1, t)
        assert mx.allclose(xt, x1, atol=1e-6)

        # t=0.5 should give midpoint
        t = mx.array([0.5, 0.5])
        xt = scheduler.interpolate(x0, x1, t)
        assert mx.allclose(xt, mx.ones_like(xt) * 0.5, atol=1e-6)

    def test_velocity(self):
        """Test that target velocity is correct."""
        from heartlib_mlx.ode.neural_ode import FlowMatchingScheduler

        scheduler = FlowMatchingScheduler()

        x0 = mx.zeros((2, 10))
        x1 = mx.ones((2, 10))
        t = mx.array([0.5, 0.5])

        velocity = scheduler.get_velocity(x0, x1, t)

        # Velocity should be x1 - x0
        assert mx.allclose(velocity, x1 - x0, atol=1e-6)
