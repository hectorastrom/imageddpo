# imageddpo
**The equivalent of SDImg2ImgPipeline for DDPO**: modifying DDPOTrainer to support image inputs in addition to text prompts. 

All credit goes to the [DDPO Implementation from HuggingFace
TRL](https://huggingface.co/docs/trl/main/en/ddpo_trainer) (now deprecated) and
the [DDPO paper](https://arxiv.org/abs/2305.13301).

I also found [Dr. Tanishq Abraham's blog](https://www.tanishq.ai/blog/posts/ddpo.html) to be incredibly helpful.

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
[ ] Confirm each fundamental implementation change is implemented
[ ] Check files for redundancy or error-prone rewriting (e.g. rewriting
denoising instead of using an Img2Img pipeline)
[ ] Remove unclear or ambiguous sections - possible relics from stiching things
together
