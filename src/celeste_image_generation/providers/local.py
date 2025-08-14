import io
from typing import Any, List

import torch
from celeste_core import ImageArtifact
from celeste_core.base.image_generator import BaseImageGenerator
from celeste_core.enums.capability import Capability
from celeste_core.enums.providers import Provider
from celeste_core.models.registry import supports
from diffusers import DiffusionPipeline


class LocalImageGenerator(BaseImageGenerator):
    """Local image generator using Hugging Face diffusers.

    Auto-detects CUDA > MPS > CPU.
    """

    def __init__(self, model: str = "stabilityai/sdxl-turbo", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model_name = model
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # FLUX models need bfloat16, others use float16/32
        if "flux" in self.model_name.lower():
            self.dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        else:
            self.dtype = torch.float16 if self.device != "cpu" else torch.float32

        # Load the pipeline with appropriate dtype
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            variant="fp16" if self.dtype == torch.float16 else None,
        )

        # For FLUX on CUDA, use CPU offloading to save memory
        if "flux" in self.model_name.lower() and self.device == "cuda":
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline = self.pipeline.to(self.device)
        if not supports(Provider.LOCAL, self.model_name, Capability.IMAGE_GENERATION):
            raise ValueError(
                f"Model '{self.model_name}' does not support IMAGE_GENERATION"
            )

    async def generate_image(self, prompt: str, **kwargs: Any) -> List[ImageArtifact]:
        # Support common 'n' alias for multiple images
        num_images = int(kwargs.pop("n", kwargs.pop("num_images", 1)))
        if num_images < 1:
            num_images = 1

        # Generate images
        with torch.no_grad():
            images = self.pipeline(
                prompt, num_images_per_prompt=num_images, **kwargs
            ).images

        # Convert to bytes
        return [
            ImageArtifact(
                data=(
                    img_bytes := io.BytesIO(),
                    img.save(img_bytes, format="PNG"),
                    img_bytes.getvalue(),
                )[2],
                metadata={"model": self.model_name, "device": self.device},
            )
            for img in images
        ]

    def __del__(self) -> None:
        """Cleanup resources when the generator is destroyed."""
        if hasattr(self, "pipeline"):
            # Clear CUDA cache if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()
            # Delete the pipeline to free memory
            del self.pipeline
