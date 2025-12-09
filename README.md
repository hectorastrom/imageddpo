# imageddpo
**The equivalent of SDImg2ImgPipeline for DDPO**: modifying DDPOTrainer to support image inputs in addition to text prompts. 

All credit goes to the [DDPO Implementation from HuggingFace
TRL](https://huggingface.co/docs/trl/main/en/ddpo_trainer) (now deprecated) and
the [DDPO paper](https://arxiv.org/abs/2305.13301).

I also found [Dr. Tanishq Abraham's
blog](https://www.tanishq.ai/blog/posts/ddpo.html) to be incredibly helpful.

# Installation

## Install from source

```bash
git clone https://github.com/hectorastrom/imageddpo.git
cd imageddpo
pip install -e .
```

## Install as a package (after publishing to PyPI)

```bash
pip install imageddpo
```

# Example Usage

For a complete example, please refer to the [gaussian
glasses](https://github.com/hectorastrom/gaussian-glasses) repo and [website](https://hectorastrom.github.io/gaussian-glasses/).

There, you will see:
1. How a distributed training loop is set up in `rl/rl_trainer.py`
2. How a reward function can be defined in `rl/reward.py`
3. How ImageDDPOTrainer can be used for image decorruption or revealing
   camouflaged animals.

---

## Birds-Eye Mechanics of Trainer (DDPO and ImageDDPO)
1. **Initialize trainer** with pipeline, scheduler, reward, config.
2. **Sample trajectories** by running the diffusion process and recording actions + log-probs.
3. **Decode** final latents to images.
4. **Compute rewards** for each sample, only at the final x_0 state.
5. **Normalize advantages** globally or per-prompt.
6. **Compute PPO loss** using replayed log-probs + current policy.
7. **Update UNet via LoRA**, keeping scheduler + VAE fixed.
8. **Repeat** for many epochs (sample → PPO → update).

## Conceptual Changes from DDPO
1. Context `c = (text_prompt, input_image)`
1. Inital state is no longer pure noise, but some noisy version of `input_image`
1. Timestep is no longer from `t=1000 -> t=0`, but from `t=(1000 * noise_strength) -> t=0`
    - e.g. if `noise_strength=0.4` we're denoising from `t=400 -> t=0`

## Implementation Changes from DDPO
1. New pipeline subclass
   `Img2ImgDDPOStableDiffusionPipeline(DDPOStableDiffusionPipeline)`
    - Image encoder + partial forward noising to x_{s*t} + denoising back to
       x_0
    - Key hyperparam is `noise_strength := s` ranging from `[0, 1]`
1. New prompt function yielding `(init_images, text_prompts, metadata)` 
    - NOTE: order of prompts & images is backwards in my implementation
1. New reward function (depending on use case) accepts `(init_images, text_prompts,metadatas)`

### Hacks, Tips, and Patches to Make This Work
1. accelerate `save_state` monkey patch
    - Possible origin: version mismatch between accelerate & old TRL library
1. Separate debug_hook to log val images during training
1. Disabling CFG on null prompt (a common prompt when using images)
1. Incorporating some image hash / id to fix per_prompt stat tracking
1. To use LoRA, just enable `use_lora` in I2I pipeline (which inherits from
   `DefaultDDPOStableDiffusionPipeline`) which uses default settings on UNet
   - Namely, `r=4, lora_alpha=4, init_lora_weights="gaussian",
      target_modules=["to_k", "to_q", "to_v", "to_out.0"]`

## Todo
[x] Confirm each fundamental implementation change is implemented
[x] Check files for redundancy or error-prone rewriting (e.g. rewriting
denoising instead of using an Img2Img pipeline)
[x] Remove unclear or ambiguous sections - possible relics from stiching things
together
[ ] Point to example usage and blog in diffusion-lens repo

---

<details>
<summary><b>More Detailed Diffs from DDPOTrainer (courtesy of GPT5.1)</b></summary>


### High-level conceptual changes (DDPOTrainer → ImageDDPOTrainer)

1. **Treat the input image as part of the environment state**

   * `prompt_fn` was extended to return `(prompt, image, metadata)` instead of just text.
   * The image is encoded through the VAE to latents and scaled, so the *initial state* for RL is `x_0` derived from the image, not pure Gaussian noise.

2. **Switch from full text-to-image sampling to image-to    * Runs only a suffix of the diffusion schedule starting from a chosen noise
      level.ne` were introduced that:

     * Accept pre-computed latents from the VAE as input.
     * Run only a suffix of the diffus      * Run only a suffix of the diffusion schedule, starting from a chosen
        noise level instead of from pure noise.via noise_strength / starting_step_ratio**

   * `noise_strength` in the trainer determines:

     * Which timestep `t_start` you add noise at when constructing `x_t` from the encoded image.
     * What fraction of the scheduler timetable is actually executed via `starting_step_ratio`.
   * This couples “how corrupted the image is when the policy starts acting” with “how many denoising actions occur,” making the MDP horizon explicit and tunable.

4. **Generate trajectories (x_t, x_{t−1}) + log-probs instead of just final images**

   * The pipeline was modified to:

     * Store all latents `x_t` along the denoising path, and
     * Call a DDPO-compatible `scheduler_step` that returns both new latents and a per-step log-prob.
   * `_generate_samples` now returns:

     * `latents[:, :-1]`, `next_latents[:, 1:]` and aligned `timesteps`,
     * Matching the exact set of steps actually executed (no padding to the full schedule).

5. **Extend the sampling–reward interface to operate on images**

   * In addition to prompts, the trainer passes decoded images plus metadata out to the reward function.
   * `compute_rewards` now effectively receives “(generated image, original prompt, original image, metadata)” tuples so downstream vision models can score the *image-conditioned* generations.

6. **Keep DDPO’s RL machinery unchanged but re-wired to the image pipeline**

   * PPO-style pieces (advantages, clipping, KL implicitly via log-probs, etc.) remain as in `DDPOTrainer`.
   * What changed is *only* how samples are generated and structured: the optimizer still sees `(timesteps, latents, next_latents, log_probs, advantages)` but these now correspond to image-conditioned rollouts instead of purely text-conditioned ones.

7. **Support both “image + prompt” and “image-only” conditioning**

   * The pipeline handles:

     * Standard CFG when `guidance_scale > 1` (uncond + text embeddings).
     * Pure image-conditioning when `guidance_scale <= 1` by running only the unconditional embedding path (no extra CFG forward pass), so the policy can optimize purely w.r.t. the image.

8. **Adjust training step semantics to keep logging and epochs meaningful**

   * `ImageDDPOTrainer.step` was overridden to:

     * Log rewards at the time of sampling, and
     * Advance `global_step` by the number of collected samples so WandB x-axes reflect “data processed” rather than only “optimizer steps,” while still delegating actual weight updates to the original DDPO training loop.

---

### Small implementation nuances and patches that made it actually run

1. **Custom, device-safe `_get_variance` and `scheduler_step`**

   * The stock TRL `_get_variance` assumes `alphas_cumprod` lives on CPU and indexes with `timestep.cpu()`, which clashed with Accelerate moving the scheduler to CUDA.
   * A custom `_get_variance` and `scheduler_step` were implemented that:

     * Normalize timesteps to 1-D tensors on a single “home” device,
     * Move `alphas_cumprod` and `final_alpha_cumprod` onto that same device before `gather`, and
     * Compute the Gaussian log-prob in a way that tolerates scalar and batched timesteps.

2. **Deterministic vs stochastic scheduler steps**

   * The scheduler step explicitly branches:

     * Deterministic DDIM updates when `eta == 0` or variance is effectively zero (log-prob set to zero because there is no stochastic action),
     * Stochastic DDIM when `eta > 0`, where the Gaussian log-prob of `x_{t−1}` is computed.
   * This is important for DDPO, since only stochastic steps should contribute meaningful policy log-probs.

3. **Log-prob reduction uses mean over pixels, not sum**

   * As in TRL, per-step log-probs are averaged over all non-batch dimensions instead of summed.
   * This keeps magnitudes stable across image sizes and matches DDPO’s existing PPO hyperparameters, avoiding having to retune clip ranges and advantage scaling.

4. **Strict latent contract for I2I (no text-only fallback)**

   * Conceptually, `ImageDDPOTrainer` always expects image-derived latents.
   * The “if latents is None, sample pure noise” branch can be removed or turned into an explicit error, to prevent accidental text-only usage and enforce the image-conditioning contract.

5. **CFG and unconditional path details**

   * The “no CFG” path reuses the unconditional embedding (what was called `negative_prompt_embeds`) and runs the UNet only once per step, avoiding wasted compute when there is no textual guidance.
   * The CFG path matches DDPO / diffusers behavior and still supports `guidance_rescale`.

6. **Timesteps alignment with the truncated schedule**

   * Because only a truncated suffix of the scheduler is executed, timesteps must match the number of actual denoising steps.
   * The trainer reconstructs `timesteps` from `scheduler.timesteps` *after* the I2I call, ensuring that `timesteps.shape[1] == log_probs.shape[1]` and corresponds to the exact steps where actions were taken.

7. **Global step accounting for logging**

   * `global_step` is incremented by the number of samples collected per epoch (batch size × num batches × num processes) on top of inner training increments.
   * This is a logging-level tweak so reward curves and training metrics align in WandB, without changing the RL math itself.

</details>