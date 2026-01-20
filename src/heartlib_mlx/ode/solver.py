"""ODE Solvers for Flow Matching in MLX."""

from typing import Callable, Optional

import mlx.core as mx


def euler_solve(
    velocity_fn: Callable[[float, mx.array, mx.array], mx.array],
    x0: mx.array,
    condition: mx.array,
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_steps: int = 10,
    guidance_scale: float = 1.0,
    uncond: Optional[mx.array] = None,
) -> mx.array:
    """Euler method ODE solver for flow matching.

    Integrates the ODE: dx/dt = v(t, x, condition)
    from t_start to t_end using fixed-step Euler method.

    Args:
        velocity_fn: Velocity function v(t, x, condition) -> velocity.
        x0: Initial state of shape (batch, ...).
        condition: Conditioning signal.
        t_start: Start time (typically 0).
        t_end: End time (typically 1).
        num_steps: Number of integration steps.
        guidance_scale: Classifier-free guidance scale.
        uncond: Unconditional embedding for CFG. If provided with
            guidance_scale > 1, applies CFG.

    Returns:
        Final state x(t_end) with same shape as x0.
    """
    dt = (t_end - t_start) / num_steps
    x = x0
    t = t_start

    for _ in range(num_steps):
        t_tensor = mx.array([t])

        if guidance_scale > 1.0 and uncond is not None:
            # Classifier-free guidance
            v_cond = velocity_fn(t_tensor, x, condition)
            v_uncond = velocity_fn(t_tensor, x, uncond)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v = velocity_fn(t_tensor, x, condition)

        # Euler step
        x = x + dt * v
        t = t + dt

    return x


def midpoint_solve(
    velocity_fn: Callable[[float, mx.array, mx.array], mx.array],
    x0: mx.array,
    condition: mx.array,
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_steps: int = 10,
    guidance_scale: float = 1.0,
    uncond: Optional[mx.array] = None,
) -> mx.array:
    """Midpoint method ODE solver for flow matching.

    Second-order method that evaluates the velocity at the midpoint
    of each step for improved accuracy.

    Args:
        velocity_fn: Velocity function v(t, x, condition) -> velocity.
        x0: Initial state of shape (batch, ...).
        condition: Conditioning signal.
        t_start: Start time (typically 0).
        t_end: End time (typically 1).
        num_steps: Number of integration steps.
        guidance_scale: Classifier-free guidance scale.
        uncond: Unconditional embedding for CFG.

    Returns:
        Final state x(t_end) with same shape as x0.
    """
    dt = (t_end - t_start) / num_steps
    x = x0
    t = t_start

    def get_velocity(t_val: float, x_val: mx.array) -> mx.array:
        t_tensor = mx.array([t_val])
        if guidance_scale > 1.0 and uncond is not None:
            v_cond = velocity_fn(t_tensor, x_val, condition)
            v_uncond = velocity_fn(t_tensor, x_val, uncond)
            return v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            return velocity_fn(t_tensor, x_val, condition)

    for _ in range(num_steps):
        # Evaluate at start
        v1 = get_velocity(t, x)

        # Midpoint estimate
        x_mid = x + 0.5 * dt * v1
        t_mid = t + 0.5 * dt

        # Evaluate at midpoint
        v_mid = get_velocity(t_mid, x_mid)

        # Full step with midpoint velocity
        x = x + dt * v_mid
        t = t + dt

    return x


def heun_solve(
    velocity_fn: Callable[[float, mx.array, mx.array], mx.array],
    x0: mx.array,
    condition: mx.array,
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_steps: int = 10,
    guidance_scale: float = 1.0,
    uncond: Optional[mx.array] = None,
) -> mx.array:
    """Heun's method (improved Euler) ODE solver.

    Second-order predictor-corrector method that averages velocities
    at the start and end of each step.

    Args:
        velocity_fn: Velocity function v(t, x, condition) -> velocity.
        x0: Initial state of shape (batch, ...).
        condition: Conditioning signal.
        t_start: Start time (typically 0).
        t_end: End time (typically 1).
        num_steps: Number of integration steps.
        guidance_scale: Classifier-free guidance scale.
        uncond: Unconditional embedding for CFG.

    Returns:
        Final state x(t_end) with same shape as x0.
    """
    dt = (t_end - t_start) / num_steps
    x = x0
    t = t_start

    def get_velocity(t_val: float, x_val: mx.array) -> mx.array:
        t_tensor = mx.array([t_val])
        if guidance_scale > 1.0 and uncond is not None:
            v_cond = velocity_fn(t_tensor, x_val, condition)
            v_uncond = velocity_fn(t_tensor, x_val, uncond)
            return v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            return velocity_fn(t_tensor, x_val, condition)

    for _ in range(num_steps):
        # Predictor (Euler step)
        v1 = get_velocity(t, x)
        x_pred = x + dt * v1

        # Corrector (evaluate at predicted point)
        t_next = t + dt
        v2 = get_velocity(t_next, x_pred)

        # Average velocities
        x = x + dt * 0.5 * (v1 + v2)
        t = t_next

    return x


def rk4_solve(
    velocity_fn: Callable[[float, mx.array, mx.array], mx.array],
    x0: mx.array,
    condition: mx.array,
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_steps: int = 10,
    guidance_scale: float = 1.0,
    uncond: Optional[mx.array] = None,
) -> mx.array:
    """Fourth-order Runge-Kutta ODE solver.

    Classic RK4 method for higher accuracy when needed.
    More expensive but better error characteristics.

    Args:
        velocity_fn: Velocity function v(t, x, condition) -> velocity.
        x0: Initial state of shape (batch, ...).
        condition: Conditioning signal.
        t_start: Start time (typically 0).
        t_end: End time (typically 1).
        num_steps: Number of integration steps.
        guidance_scale: Classifier-free guidance scale.
        uncond: Unconditional embedding for CFG.

    Returns:
        Final state x(t_end) with same shape as x0.
    """
    dt = (t_end - t_start) / num_steps
    x = x0
    t = t_start

    def get_velocity(t_val: float, x_val: mx.array) -> mx.array:
        t_tensor = mx.array([t_val])
        if guidance_scale > 1.0 and uncond is not None:
            v_cond = velocity_fn(t_tensor, x_val, condition)
            v_uncond = velocity_fn(t_tensor, x_val, uncond)
            return v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            return velocity_fn(t_tensor, x_val, condition)

    for _ in range(num_steps):
        k1 = get_velocity(t, x)
        k2 = get_velocity(t + 0.5 * dt, x + 0.5 * dt * k1)
        k3 = get_velocity(t + 0.5 * dt, x + 0.5 * dt * k2)
        k4 = get_velocity(t + dt, x + dt * k3)

        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + dt

    return x
