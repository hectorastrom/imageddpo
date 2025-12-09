# @Time    : 2025-12-09 12:18
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : pipeline.py

import torch
import numpy as np
from trl import DefaultDDPOStableDiffusionPipeline
from trl.models.modeling_sd_base import (
    DDPOPipelineOutput,
    DDPOSchedulerOutput,
    _left_broadcast,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
)


# ==========================================
# 1. TRL/DDPO Helper Classes & Math
# ==========================================


def _get_variance(self, timestep, prev_timestep):
    """
    Compute the per-timestep variance used by the DDIM-style update.

    All tensors (alphas_cumprod, final_alpha_cumprod, timestep, prev_timestep)
    are placed on the same device to avoid device mismatch in gather.
    Handles both scalar and batched timesteps.

    This is near identical to the implementation in TRL; however, a custom implementation
    was needed as `alpha_cumprods` and `final_alpha_cumprod` were often moved
    .to(device) by accelerate, when they are expected to remain on CPU.
    """
    # Ensure we have tensors
    if not torch.is_tensor(timestep):
        timestep = torch.as_tensor(timestep)
    if not torch.is_tensor(prev_timestep):
        prev_timestep = torch.as_tensor(prev_timestep)

    # Use the scheduler's alpha device as the "home" device for the variance math
    device = self.alphas_cumprod.device
    timestep = timestep.to(device)
    prev_timestep = prev_timestep.to(device)

    if not torch.is_tensor(timestep):
        timestep = torch.tensor([timestep], device=device, dtype=torch.long)
    else:
        timestep = timestep.to(device).long().reshape(-1)

    if not torch.is_tensor(prev_timestep):
        prev_timestep = torch.tensor([prev_timestep], device=device, dtype=torch.long)
    else:
        prev_timestep = prev_timestep.to(device).long().reshape(-1)

    alphas_cumprod = self.alphas_cumprod.to(device)
    final_alpha_cumprod = self.final_alpha_cumprod.to(device)

    alpha_prod_t = torch.gather(alphas_cumprod, 0, timestep)

    prev_mask = prev_timestep >= 0
    prev_timestep_clipped = prev_timestep.clamp_min(0)

    alpha_prod_t_prev_raw = torch.gather(alphas_cumprod, 0, prev_timestep_clipped)
    alpha_prod_t_prev = torch.where(
        prev_mask, alpha_prod_t_prev_raw, final_alpha_cumprod
    )

    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance


def scheduler_step(
    self,
    model_output,
    timestep,
    sample,
    eta=0.0,
    use_clipped_model_output=False,
    generator=None,
    prev_sample=None,
):
    """
    Device-safe scheduler step used for:
    - sampling inside i2i_pipeline_step
    - training loss (via I2IDDPOStableDiffusionPipeline.scheduler_step)

    This version is robust to scalar timesteps (0-d tensors / ints) and
    batched timesteps (1-d tensors).
    """
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' "
            "after creating the scheduler"
        )

    device = sample.device

    # Normalize timestep to a tensor on the correct device
    if not torch.is_tensor(timestep):
        timestep = torch.as_tensor(timestep, device=device)
    else:
        timestep = timestep.to(device)

    prev_offset = self.config.num_train_timesteps // self.num_inference_steps

    # Normalize timestep to 1-D long tensor on the right device
    if not torch.is_tensor(timestep):
        timestep = torch.tensor([timestep], device=device, dtype=torch.long)
    else:
        timestep = timestep.to(device).long().reshape(-1)  # (N,)

    prev_timestep = timestep - prev_offset  # may be negative

    alphas_cumprod = self.alphas_cumprod.to(device)
    final_alpha_cumprod = self.final_alpha_cumprod.to(device)  # scalar

    # Mask which prev indices are valid
    prev_mask = prev_timestep >= 0  # (N,)
    prev_timestep_clipped = prev_timestep.clamp_min(0)  # (N,)

    # Gather current and previous alpha products
    alpha_prod_t = torch.gather(alphas_cumprod, 0, timestep)  # (N,)
    alpha_prod_t_prev_raw = torch.gather(
        alphas_cumprod, 0, prev_timestep_clipped
    )  # (N,)
    alpha_prod_t_prev = torch.where(
        prev_mask, alpha_prod_t_prev_raw, final_alpha_cumprod
    )

    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(device)

    beta_prod_t = 1 - alpha_prod_t

    if self.config.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5
        pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (
            sample - alpha_prod_t**0.5 * pred_original_sample
        ) / beta_prod_t**0.5
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (
            beta_prod_t**0.5
        ) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one "
            "of `epsilon`, `sample`, or `v_prediction`"
        )

    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # Compute variance
    variance = _get_variance(self, timestep, prev_timestep)

    # Branch for deterministic / zero-variance steps
    deterministic_step = (eta == 0.0) or torch.all(variance.abs() < 1e-20)

    if deterministic_step:
        # Standard DDIM deterministic update
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * pred_epsilon
        prev_sample_mean = (
            alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        )

        if prev_sample is None:
            prev_sample = prev_sample_mean

        # No stochasticity => no meaningful log-prob increment
        log_prob = torch.zeros(sample.shape[0], device=device, dtype=sample.dtype)
        return DDPOSchedulerOutput(prev_sample.type(sample.dtype), log_prob)

    # Stochastic branch (eta > 0 and variance > 0)
    variance = torch.clamp(variance, min=1e-20)
    std_dev_t = eta * variance**0.5
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(device)

    if use_clipped_model_output:
        pred_epsilon = (
            sample - alpha_prod_t**0.5 * pred_original_sample
        ) / beta_prod_t**0.5

    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * pred_epsilon
    prev_sample_mean = (
        alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
    )

    if prev_sample is None:
        variance_noise = torch.randn(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(
            torch.sqrt(2 * torch.as_tensor(np.pi, device=device, dtype=sample.dtype))
        )
    )
    # Normalize log-prob over spatial dimensions to keep magnitudes stable
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return DDPOSchedulerOutput(prev_sample.type(sample.dtype), log_prob)


# ==========================================
# 2. The Custom I2I Pipeline Logic
# ==========================================


@torch.no_grad()
def i2i_pipeline_step(
    self,
    prompt=None,
    height=None,
    width=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    negative_prompt=None,
    num_images_per_prompt=1,
    eta=0.0,
    generator=None,
    latents=None,
    prompt_embeds=None,
    negative_prompt_embeds=None,
    output_type="pil",
    return_dict=True,
    callback=None,
    callback_steps=1,
    cross_attention_kwargs=None,
    guidance_rescale=0.0,
    starting_step_ratio: float = 1.0,
):
    # 0. Defaults
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 2. Define batch size
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt

    # condition on classifier_free_guidance to avoid double generation in null
    # prompt case
    if not do_classifier_free_guidance:
        # completely avoid context if guidance_scale <= 1.0, so that e.g. "id"
        # used for per_prompt_stat_tracking isn't conditioned on at all
        if negative_prompt_embeds is None:
            raise ValueError(
                "negative_prompt_embeds required for unconditional generation"
            )
        prompt_embeds = negative_prompt_embeds.clone()
    else:
        # standard classifier-free guidance 2-inference pass
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

    # 4. Prepare timesteps (WITH I2I FIX)
    self.scheduler.set_timesteps(num_inference_steps, device=device)

    timesteps = self.scheduler.timesteps
    start_idx = int(len(timesteps) * (1 - starting_step_ratio))
    start_idx = max(0, min(start_idx, len(timesteps) - 1))

    timesteps = timesteps[start_idx:]

    # 5. Prepare latent variables
    if latents is None:
        raise ValueError(
            "I2I pipeline requires latents from an encoded image. "
            "Please pass latents derived from a VAE encode."
        )
    else:
        # For I2I we assume latents are already correctly scaled and noised
        latents = latents.to(device=device, dtype=prompt_embeds.dtype)

    # 6. Denoising loop
    all_latents = [latents]
    all_log_probs = []

    with self.progress_bar(total=len(timesteps)) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(
                    noise_pred,
                    noise_pred_text,
                    guidance_rescale=guidance_rescale,
                )

            # NOTE: this now safely handles scalar t (0-d tensor) inside scheduler_step
            scheduler_output = scheduler_step(
                self.scheduler, noise_pred, t, latents, eta
            )
            latents = scheduler_output.latents
            log_prob = scheduler_output.log_probs

            all_latents.append(latents)
            all_log_probs.append(log_prob)

            progress_bar.update()

    if output_type != "latent":
        image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]
        image = self.image_processor.postprocess(
            image,
            output_type=output_type,
            do_denormalize=[True] * image.shape[0],
        )
    else:
        image = latents

    return DDPOPipelineOutput(image, all_latents, all_log_probs)


class I2IDDPOStableDiffusionPipeline(DefaultDDPOStableDiffusionPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> DDPOPipelineOutput:
        return i2i_pipeline_step(self.sd_pipeline, *args, **kwargs)

    def scheduler_step(self, *args, **kwargs):
        """
        Override TRL's DefaultDDPOStableDiffusionPipeline.scheduler_step so that
        DDPOTrainer.calculate_loss uses the same device-safe scheduler_step.
        """
        return scheduler_step(self.sd_pipeline.scheduler, *args, **kwargs)

