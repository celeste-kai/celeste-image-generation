import io
from typing import Any, List

from celeste_core import ImageArtifact
from celeste_core.base.image_generator import BaseImageGenerator
from celeste_core.config.settings import settings
from celeste_core.enums.providers import Provider
from huggingface_hub import AsyncInferenceClient


class HuggingFaceImageGenerator(BaseImageGenerator):
    def __init__(
        self, model: str = "black-forest-labs/FLUX.1-schnell", **kwargs: Any
    ) -> None:
        super().__init__(model=model, provider=Provider.HUGGINGFACE, **kwargs)
        api_key = settings.huggingface.access_token
        self.client = AsyncInferenceClient(token=api_key)

    async def generate_image(self, prompt: str, **kwargs: Any) -> List[ImageArtifact]:
        img = await self.client.text_to_image(prompt, model=self.model, **kwargs)
        fmt = getattr(img, "format", None) or "PNG"
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return [
            ImageArtifact(
                data=buf.getvalue(),
                metadata={
                    "model": self.model,
                    "provider": "huggingface",
                    "format": fmt,
                    **kwargs,
                },
            )
        ]
