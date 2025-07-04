import base64
from typing import Any, Dict, List, Tuple

import aiohttp

from celeste_image_generation.base import BaseImageGenerator
from celeste_image_generation.core.config import STABILITYAI_API_KEY
from celeste_image_generation.core.enums import StabilityModel
from celeste_image_generation.core.types import GeneratedImage, ImagePrompt


class StabilityAIImageGenerator(BaseImageGenerator):
    def __init__(self, model: str = StabilityModel.SDXL_1_0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_key = STABILITYAI_API_KEY
        self.model_name = model.value if hasattr(model, 'value') else model
        self.is_v2 = self.model_name in ["ultra", "core", "sd3.5-large", "sd3.5-large-turbo", "sd3.5-medium"]
        self.is_raw = self.model_name in ["core", "ultra"]

    def _format_request(self, prompt: ImagePrompt, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
        """Format the request based on API version."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "image/*" if self.is_raw else "application/json"
        }
        
        if self.is_v2:
            endpoint = f"https://api.stability.ai/v2beta/stable-image/generate/{'sd3' if self.model_name.startswith('sd3') else self.model_name}"
            data = aiohttp.FormData()
            data.add_field('none', '', filename='', content_type='application/octet-stream')
            data.add_field('prompt', prompt.content)
            data.add_field('output_format', kwargs.get("output_format", "webp" if self.is_raw else "png"))
            if self.model_name.startswith("sd3"):
                data.add_field('model', self.model_name)
            return endpoint, {"headers": headers, "data": data}
        else:
            endpoint = f"https://api.stability.ai/v1/generation/{self.model_name}/text-to-image"
            return endpoint, {"headers": headers, "json": {
                "text_prompts": [{"text": prompt.content}],
                "height": kwargs.get("height", 1024),
                "width": kwargs.get("width", 1024),
            }}

    def _parse_response(self, data: Any) -> List[GeneratedImage]:
        """Parse response based on API version and model type."""
        if self.is_v2:
            return [GeneratedImage(
                image=base64.b64decode(data["image"]),
                metadata={"model": self.model_name, "seed": data.get("seed")}
            )]
        else:
            return [
                GeneratedImage(
                    image=base64.b64decode(artifact["base64"]),
                    metadata={"model": self.model_name, "seed": artifact.get("seed")}
                )
                for artifact in data.get("artifacts", [])
            ]

    async def generate_image(self, prompt: ImagePrompt, **kwargs: Any) -> List[GeneratedImage]:
        endpoint, request_kwargs = self._format_request(prompt, **kwargs)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, **request_kwargs) as response:
                if response.status != 200:
                    error_text = await response.text()
                    if response.status == 403 and "content_moderation" in error_text:
                        raise Exception(
                            "Content blocked by Stability AI moderation. "
                            "Try rephrasing your prompt or using local models instead."
                        )
                    raise Exception(f"Stability AI API error ({response.status}): {error_text}")
                
                if self.is_raw:
                    return [GeneratedImage(image=await response.read(), metadata={"model": self.model_name})]
                
                return self._parse_response(await response.json())