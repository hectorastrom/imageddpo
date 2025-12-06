# imageddpo
The equivalent of SDImg2ImgPipeline for DDPO: modifying DDPOTrainer to support
optimization around image inputs in addition to text prompts. 

All credit goes to the [DDPO Implementation from HuggingFace
TRL](https://huggingface.co/docs/trl/main/en/ddpo_trainer) (now deprecated) and
the [DDPO paper](https://arxiv.org/abs/2305.13301).

I also found [Dr. Tanishq Abraham's blog](https://www.tanishq.ai/blog/posts/ddpo.html) to be incredibly helpful.

## Mechanics of Trainer (DDPO and ImageDDPO)
1. **Initialize trainer** with pipeline, scheduler, reward, config.
2. **Sample trajectories** by running the diffusion process and recording actions + log-probs.
3. **Decode** final latents to images.
4. **Compute rewards** for each sample.
5. **Normalize advantages** globally or per-prompt.
6. **Compute PPO loss** using replayed log-probs + current policy.
7. **Update UNet via LoRA**, keeping scheduler + VAE fixed.
8. **Repeat** for many epochs (sample → PPO → update).

## Changes from DDPO
TODO:

### Tricks to make it work

## Todo