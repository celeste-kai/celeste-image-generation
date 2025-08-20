import io
from typing import Any, List

import torch
from celeste_core import ImageArtifact
from celeste_core.base.image_generator import BaseImageGenerator
from celeste_core.config.settings import settings
from diffusers import DiffusionPipeline


class LocalImageGenerator(BaseImageGenerator):
    """Local image generator using Hugging Face diffusers."""

    def __init__(self, model: str = "stabilityai/sdxl-turbo", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model_name = model

        # Detect device: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float32  # float32 is more stable on MPS
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        # Load pipeline with token from settings if available
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            token=settings.huggingface.access_token,
            use_safetensors=True,
        ).to(self.device)

        # Enable memory optimizations
        if self.device == "cuda":
            self.pipeline.enable_model_cpu_offload()
        elif self.device == "mps":
            # Recommended for Apple Silicon with < 64GB RAM
            self.pipeline.enable_attention_slicing()

    async def generate_image(self, prompt: str, **kwargs: Any) -> List[ImageArtifact]:
        num_images = kwargs.pop("n", kwargs.pop("num_images", 1))

        with torch.no_grad():
            images = self.pipeline(
                prompt, num_images_per_prompt=num_images, **kwargs
            ).images

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
