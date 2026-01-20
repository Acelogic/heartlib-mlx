"""Neural ODE wrapper for flow matching decoders."""

from typing import Callable, Optional, Literal

import mlx.core as mx
import mlx.nn as nn

from heartlib_mlx.ode.solver import euler_solve, midpoint_solve, heun_solve, rk4_solve


SolverType = Literal["euler", "midpoint", "heun", "rk4"]


class NeuralODE(nn.Module):
    """Neural ODE wrapper for flow matching models.

    Wraps a velocity network and provides methods for integration
    from noise to data (or vice versa).

    Args:
        velocity_model: The neural network that predicts velocities.
            Should accept (t, x, condition) and return velocity.
        solver: ODE solver to use. One of "euler", "midpoint", "heun", "rk4".
        num_steps: Default number of integration steps.
    """

    def __init__(
        self,
        velocity_model: nn.Module,
        solver: SolverType = "euler",
        num_steps: int = 10,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.solver = solver
        self.num_steps = num_steps

        self._solver_fn = {
            "euler": euler_solve,
            "midpoint": midpoint_solve,
            "heun": heun_solve,
            "rk4": rk4_solve,
        }[solver]

    def velocity(
        self,
        t: mx.array,
        x: mx.array,
        condition: mx.array,
    ) -> mx.array:
        """Compute velocity at given time and state.

        Args:
            t: Time value (scalar or batch).
            x: Current state.
            condition: Conditioning signal.

        Returns:
            Velocity tensor.
        """
        return self.velocity_model(t, x, condition)

    def sample(
        self,
        x0: mx.array,
        condition: mx.array,
        num_steps: Optional[int] = None,
        guidance_scale: float = 1.0,
        uncond: Optional[mx.array] = None,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> mx.array:
        """Sample from the flow model by integrating from noise to data.

        Args:
            x0: Initial noise sample.
            condition: Conditioning signal.
            num_steps: Number of integration steps (uses default if None).
            guidance_scale: Classifier-free guidance scale.
            uncond: Unconditional embedding for CFG.
            t_start: Start time (0 for noise).
            t_end: End time (1 for data).

        Returns:
            Generated sample.
        """
        steps = num_steps or self.num_steps

        return self._solver_fn(
            velocity_fn=self.velocity,
            x0=x0,
            condition=condition,
            t_start=t_start,
            t_end=t_end,
            num_steps=steps,
            guidance_scale=guidance_scale,
            uncond=uncond,
        )

    def encode(
        self,
        x1: mx.array,
        condition: mx.array,
        num_steps: Optional[int] = None,
    ) -> mx.array:
        """Encode data to noise by integrating backwards.

        Args:
            x1: Data sample.
            condition: Conditioning signal.
            num_steps: Number of integration steps.

        Returns:
            Latent noise.
        """
        steps = num_steps or self.num_steps

        # Integrate from t=1 (data) to t=0 (noise)
        # Need to negate velocity for reverse integration
        def neg_velocity(t, x, cond):
            return -self.velocity(t, x, cond)

        return self._solver_fn(
            velocity_fn=neg_velocity,
            x0=x1,
            condition=condition,
            t_start=1.0,
            t_end=0.0,
            num_steps=steps,
            guidance_scale=1.0,
            uncond=None,
        )


class FlowMatchingScheduler:
    """Scheduler for flow matching training and inference.

    Handles the interpolation between noise and data during training,
    and provides the target velocity for the loss.
    """

    def __init__(self, sigma_min: float = 1e-4):
        """Initialize scheduler.

        Args:
            sigma_min: Minimum noise level to prevent numerical issues.
        """
        self.sigma_min = sigma_min

    def interpolate(
        self,
        x0: mx.array,
        x1: mx.array,
        t: mx.array,
    ) -> mx.array:
        """Interpolate between noise (x0) and data (x1).

        Args:
            x0: Noise sample.
            x1: Data sample.
            t: Interpolation time in [0, 1].

        Returns:
            Interpolated sample.
        """
        # Ensure t has right shape for broadcasting
        t = t.reshape(-1, *([1] * (x0.ndim - 1)))

        # Linear interpolation with optional noise floor
        return (1 - t) * x0 + t * x1

    def get_velocity(
        self,
        x0: mx.array,
        x1: mx.array,
        t: mx.array,
    ) -> mx.array:
        """Get target velocity for flow matching loss.

        For linear interpolation, the velocity is simply (x1 - x0).

        Args:
            x0: Noise sample.
            x1: Data sample.
            t: Time (unused for linear interpolation, included for API).

        Returns:
            Target velocity.
        """
        return x1 - x0

    def sample_timesteps(
        self,
        batch_size: int,
        device: Optional[str] = None,
    ) -> mx.array:
        """Sample random timesteps for training.

        Args:
            batch_size: Number of timesteps to sample.
            device: Unused, for API compatibility.

        Returns:
            Random timesteps in [0, 1].
        """
        return mx.random.uniform(shape=(batch_size,))

    def get_loss(
        self,
        velocity_pred: mx.array,
        x0: mx.array,
        x1: mx.array,
        t: mx.array,
    ) -> mx.array:
        """Compute flow matching loss.

        Args:
            velocity_pred: Predicted velocity from the model.
            x0: Noise sample.
            x1: Data sample.
            t: Timestep.

        Returns:
            MSE loss between predicted and target velocity.
        """
        velocity_target = self.get_velocity(x0, x1, t)
        return mx.mean((velocity_pred - velocity_target) ** 2)
