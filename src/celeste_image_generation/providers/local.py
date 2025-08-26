import io
from typing import Any, List, Optional

import torch
from celeste_core import ImageArtifact
from celeste_core.base.image_generator import BaseImageGenerator
from celeste_core.config.settings import settings
from celeste_core.enums.providers import Provider
from diffusers import DiffusionPipeline


class LocalImageGenerator(BaseImageGenerator):
    """Local image generator using Hugging Face diffusers."""

    def __init__(self, model: str = "stabilityai/sdxl-turbo", **kwargs: Any) -> None:
        super().__init__(model=model, provider=Provider.LOCAL, **kwargs)
        self.model = model
        self.pipeline: Optional[DiffusionPipeline] = None

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

    def _load_pipeline(self) -> None:
        """Lazy load the pipeline to save memory until first use."""
        if self.pipeline is None:
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model,
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
        self._load_pipeline()
        if self.pipeline is None:
            raise RuntimeError("Failed to load pipeline")
        with torch.no_grad():
            images = self.pipeline(prompt, **kwargs).images

        return [
            ImageArtifact(
                data=(
                    img_bytes := io.BytesIO(),
                    img.save(img_bytes, format="PNG"),
                    img_bytes.getvalue(),
                )[2],
                metadata={"model": self.model, "device": self.device, **kwargs},
            )
            for img in images
        ]
