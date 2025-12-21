"""
DDPM diffusion utilities.
Implements noise scheduling, forward diffusion, and sampling.
"""

import torch
import torch.nn.functional as F


class GaussianDiffusion:
    """
    Gaussian Diffusion process for DDPM.

    Uses a linear beta schedule by default, which works well for small images.
    """

    def __init__(
        self,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device="cpu",
    ):
        self.timesteps = timesteps
        self.device = device

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0).
        Add noise to x_0 to get x_t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][
            :, None, None, None
        ]

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, model, x_0, t, noise=None):
        """
        Compute training loss (predict the noise).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = model(x_t, t.float())

        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """
        Single reverse diffusion step: p(x_{t-1} | x_t).
        """
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][
            :, None, None, None
        ]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t][:, None, None, None]

        # Predict noise
        predicted_noise = model(x_t, t.float())

        # Compute mean
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t][:, None, None, None]
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, shape, seed=None):
        """
        Generate samples by running the full reverse diffusion process.

        Args:
            model: The trained denoising model
            shape: (batch_size, channels, height, width)
            seed: Random seed for reproducibility

        Returns:
            Generated images in [-1, 1] range
        """
        if seed is not None:
            torch.manual_seed(seed)

        batch_size = shape[0]
        device = self.device

        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Reverse diffusion
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch)

        return x

    @torch.no_grad()
    def sample_ddim(self, model, shape, seed=None, steps=50):
        """
        DDIM sampling for faster generation.

        Args:
            model: The trained denoising model
            shape: (batch_size, channels, height, width)
            seed: Random seed for reproducibility
            steps: Number of sampling steps (fewer = faster)

        Returns:
            Generated images in [-1, 1] range
        """
        if seed is not None:
            torch.manual_seed(seed)

        batch_size = shape[0]
        device = self.device

        # Create timestep schedule
        step_size = self.timesteps // steps
        timesteps = list(range(0, self.timesteps, step_size))
        timesteps = list(reversed(timesteps))

        # Start from pure noise
        x = torch.randn(shape, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict noise
            predicted_noise = model(x, t_batch.float())

            # Get alpha values
            alpha_cumprod = self.alphas_cumprod[t]
            alpha_cumprod_prev = (
                self.alphas_cumprod[timesteps[i + 1]]
                if i + 1 < len(timesteps)
                else torch.tensor(1.0, device=device)
            )

            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(
                alpha_cumprod
            )
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_cumprod_prev) * predicted_noise

            # DDIM step (deterministic)
            x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt

        return x
