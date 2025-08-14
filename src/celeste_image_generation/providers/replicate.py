from typing import Any, List

import aiohttp
from celeste_core import ImageArtifact
from celeste_core.base.image_generator import BaseImageGenerator
from celeste_core.config.settings import settings


class ReplicateImageGenerator(BaseImageGenerator):
    """Replicate image generator for various models."""

    def __init__(self, model: str = "stability-ai/sdxl", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_key = settings.replicate.api_token
        self.model_name = model
        self.base_url = "https://api.replicate.com/v1"

    async def generate_image(self, prompt: str, **kwargs: Any) -> List[ImageArtifact]:
        """Generate images using Replicate's API."""
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

        input_data = {"prompt": prompt}
        for key in ["width", "height", "num_outputs", "guidance_scale", "seed"]:
            if key in kwargs:
                input_data[key] = kwargs[key]
        if "n" in kwargs:
            input_data["num_outputs"] = kwargs["n"]

        async with aiohttp.ClientSession() as session:
            # Create prediction and get result
            async with session.post(
                f"{self.base_url}/models/{self.model_name}/predictions",
                json={"input": input_data, "wait": True},  # Use wait parameter
                headers=headers,
            ) as response:
                result = await response.json()
                output = result.get("output", [])

                images = []
                for url in output:
                    async with session.get(url) as img_response:
                        images.append(
                            ImageArtifact(
                                data=await img_response.read(),
                                metadata={"model": self.model_name},
                            )
                        )
                return images
