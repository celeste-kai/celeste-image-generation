import io
from typing import Any, List

from celeste_core import ImageArtifact
from celeste_core.base.image_generator import BaseImageGenerator
from celeste_core.config.settings import settings
from celeste_core.enums.capability import Capability
from celeste_core.enums.providers import Provider
from celeste_core.models.registry import supports
from huggingface_hub import AsyncInferenceClient


class HuggingFaceImageGenerator(BaseImageGenerator):
    def __init__(
        self, model: str = "black-forest-labs/FLUX.1-schnell", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        api_key = settings.huggingface.access_token
        # Non-raising: proceed with None token; upstream may handle anonymous
        self.model_name = model
        self.client = AsyncInferenceClient(token=api_key)
        # Non-raising validation; store support state for callers to inspect
        self.is_supported = supports(
            Provider.HUGGINGFACE, self.model_name, Capability.IMAGE_GENERATION
        )

    async def generate_image(self, prompt: str, **kwargs: Any) -> List[ImageArtifact]:
        # Produce a single image; callers control multiplicity via parallel calls
        # Strip non-HF args and honor caller-specified format when provided
        requested_format = kwargs.pop("output_format", None)
        img = await self.client.text_to_image(prompt, model=self.model_name, **kwargs)
        fmt = requested_format or getattr(img, "format", None) or "PNG"
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return [
            ImageArtifact(
                data=buf.getvalue(),
                metadata={
                    "model": self.model_name,
                    "provider": "huggingface",
                    "format": fmt,
                },
            )
        ]
