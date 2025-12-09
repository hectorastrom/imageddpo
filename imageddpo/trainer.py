# @Time    : 2025-12-09 12:18
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : trainer.py

import torch
import wandb
from trl import DDPOTrainer


class ImageDDPOTrainer(DDPOTrainer):
    def __init__(self, *args, noise_strength=0.2, debug_hook=None, **kwargs):
        self.noise_strength = noise_strength
        self.debug_hook = debug_hook
        super().__init__(*args, **kwargs)

    def _generate_samples(self, iterations, batch_size):
        """
        Only overridden to:
        1. Handle Image Inputs
        2. Return UNPADDED trajectories (so shapes match the shortened training loop)
        3. Slice timesteps correctly (fixing your crash)
        """
        current_step = wandb.run.step if wandb.run is not None else 0
        if self.debug_hook is not None and self.accelerator.is_main_process:
            self.debug_hook(self.sd_pipeline, self.noise_strength, current_step)

        samples = []
        prompt_image_pairs = []

        self.sd_pipeline.unet.eval()
        self.sd_pipeline.vae.eval()

        sample_neg_prompt_embeds = self.neg_prompt_embed.repeat(batch_size, 1, 1)

        for _ in range(iterations):
            # 1. GENERATE INPUTS
            prompts, input_images, prompt_metadata = zip(
                *[self.prompt_fn() for _ in range(batch_size)]
            )
            input_images = torch.stack(input_images).to(
                self.accelerator.device, dtype=self.sd_pipeline.vae.dtype
            )

            # takes [0,1]->[-1,1] - i've confirmed images range from [0, 1].
            input_images = 2.0 * input_images - 1.0

            prompt_ids = self.sd_pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
            prompt_embeds = self.sd_pipeline.text_encoder(prompt_ids)[0]

            with self.autocast():
                # 2. ENCODE & NOISE
                init_latents = self.sd_pipeline.vae.encode(
                    input_images
                ).latent_dist.sample()
                init_latents = init_latents * self.sd_pipeline.vae.config.scaling_factor

                # Align the noise timestep with the first denoising timestep the pipeline will run.
                self.sd_pipeline.scheduler.set_timesteps(
                    self.config.sample_num_steps, device=self.accelerator.device
                )
                full_timesteps = (
                    self.sd_pipeline.scheduler.timesteps
                )  # shape [num_inference_steps]

                start_idx = int(len(full_timesteps) * (1 - self.noise_strength))
                start_idx = max(0, min(start_idx, len(full_timesteps) - 1))

                t_start = full_timesteps[
                    start_idx
                ]  # this is in scheduler timestep space
                timesteps = (
                    t_start.repeat(batch_size).to(self.accelerator.device).long()
                )

                noise = torch.randn_like(init_latents)
                noisy_latents = self.sd_pipeline.scheduler.add_noise(
                    init_latents, noise, timesteps
                )

                # 3. RUN PIPELINE
                sd_output = self.sd_pipeline(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=self.config.sample_num_steps,
                    guidance_scale=self.config.sample_guidance_scale,
                    eta=self.config.sample_eta,
                    output_type="pt",
                    latents=noisy_latents,
                    starting_step_ratio=self.noise_strength,
                )

                latents = torch.stack(sd_output.latents, dim=1)
                log_probs = torch.stack(sd_output.log_probs, dim=1)
                images = sd_output.images

            # 4. FIX TIMESTEPS SIZE
            # The scheduler holds all 50 timesteps. We only want the last N executed steps.
            actual_num_steps = log_probs.shape[1]
            full_timesteps = self.sd_pipeline.scheduler.timesteps.to(
                self.accelerator.device
            )
            timesteps = full_timesteps[-actual_num_steps:].repeat(batch_size, 1)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],
                    "next_latents": latents[:, 1:],
                    "log_probs": log_probs,
                    "negative_prompt_embeds": sample_neg_prompt_embeds,
                }
            )
            prompt_image_pairs.append([images, prompts, prompt_metadata])

        return samples, prompt_image_pairs

