import io
from typing import Any, List

from celeste_core import ImageArtifact
from celeste_core.base.image_generator import BaseImageGenerator
from celeste_core.config.settings import settings
from celeste_core.enums.capability import Capability
from celeste_core.models.registry import supports
from huggingface_hub import InferenceClient


class HuggingFaceImageGenerator(BaseImageGenerator):
    """Hugging Face Inference API image generator."""

    def __init__(
        self, model: str = "black-forest-labs/FLUX.1-schnell", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.api_key = settings.huggingface.access_token
        if not self.api_key:
            raise ValueError(
                "Hugging Face API key not provided. "
                "Set HUGGINGFACE_TOKEN environment variable."
            )

        self.model_name = model

        # Initialize InferenceClient with automatic provider detection
        self.client = InferenceClient(token=self.api_key)
        if not supports(self.model_name, Capability.IMAGE_GENERATION):
            raise ValueError(
                f"Model '{self.model_name}' does not support IMAGE_GENERATION"
            )

    async def generate_image(self, prompt: str, **kwargs: Any) -> List[ImageArtifact]:
        """
        Generate images using Hugging Face Inference Client.
        """
        image = self.client.text_to_image(prompt, model=self.model_name, **kwargs)

        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        image_bytes = img_byte_arr.getvalue()

        return [
            ImageArtifact(
                data=image_bytes,
                metadata={"model": self.model_name, "provider": "huggingface"},
            )
        ]
