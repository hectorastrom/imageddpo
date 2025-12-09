"""
ImageDDPO: Image-to-Image DDPO Training

The equivalent of SDImg2ImgPipeline for DDPO: modifying DDPOTrainer to support
image inputs in addition to text prompts.
"""

from imageddpo.trainer import ImageDDPOTrainer
from imageddpo.pipeline import I2IDDPOStableDiffusionPipeline

__all__ = ["ImageDDPOTrainer", "I2IDDPOStableDiffusionPipeline"]
__version__ = "0.1.0"

