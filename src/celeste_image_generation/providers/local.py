import io
from typing import Any, List

import torch
from diffusers import DiffusionPipeline

from celeste_image_generation.base import BaseImageGenerator
from celeste_image_generation.core.enums import LocalModel
from celeste_image_generation.core.types import GeneratedImage, ImagePrompt


class LocalImageGenerator(BaseImageGenerator):
    """Local image generator using Hugging Face diffusers. Auto-detects CUDA > MPS > CPU."""
    
    def __init__(self, model: str = LocalModel.SDXL_TURBO, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model_name = model.value if hasattr(model, 'value') else model
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        # FLUX models need bfloat16, others use float16/32
        if "flux" in self.model_name.lower():
            self.dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        else:
            self.dtype = torch.float16 if self.device != "cpu" else torch.float32
        
        # Load the pipeline with appropriate dtype
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_name, 
            torch_dtype=self.dtype,
            variant="fp16" if self.dtype == torch.float16 else None
        )
        
        # For FLUX on CUDA, use CPU offloading to save memory
        if "flux" in self.model_name.lower() and self.device == "cuda":
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline = self.pipeline.to(self.device)
    
    
    async def generate_image(self, prompt: ImagePrompt, **kwargs: Any) -> List[GeneratedImage]:
        # Generate images
        with torch.no_grad():
            images = self.pipeline(prompt.content, **kwargs).images

        # Convert to bytes
        return [
            GeneratedImage(
                image=(img_bytes := io.BytesIO(), img.save(img_bytes, format='PNG'), img_bytes.getvalue())[2],
                metadata={"model": self.model_name, "device": self.device}
            )
            for img in images
        ]

    def __del__(self):
        """Cleanup resources when the generator is destroyed."""
        if hasattr(self, 'pipeline'):
            # Clear CUDA cache if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()
            # Delete the pipeline to free memory
            del self.pipeline
