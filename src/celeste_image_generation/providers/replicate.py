from typing import Any

import replicate
from celeste_core import ImageArtifact, Provider
from celeste_core.base.image_generator import BaseImageGenerator
from celeste_core.config.settings import settings


class ReplicateImageGenerator(BaseImageGenerator):
    """Replicate image generator for various models."""

    def __init__(self, model: str = "stability-ai/sdxl", **kwargs: Any) -> None:
        super().__init__(model=model, provider=Provider.REPLICATE, **kwargs)
        self.client = replicate.Client(api_token=settings.replicate.api_token)

    async def generate_image(self, prompt: str, **kwargs: Any) -> list[ImageArtifact]:
        """Generate images using Replicate's official SDK."""
        input_data = {"prompt": prompt, **kwargs}

        # Use client's async run method
        outputs = await self.client.async_run(self.model, input=input_data)

        images: list[ImageArtifact] = []

        # Handle both single output and list of outputs
        if not isinstance(outputs, list):
            outputs = [outputs]

        for output in outputs:
            # Read binary data from FileOutput object
            image_bytes = output.read()

            images.append(ImageArtifact(data=image_bytes, metadata={"model": self.model, **kwargs}))

        return images
