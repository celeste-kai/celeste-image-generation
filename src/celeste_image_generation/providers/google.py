from typing import Any, List

from google import genai

from celeste_image_generation.base import BaseImageGenerator
from celeste_image_generation.core.config import GOOGLE_API_KEY
from celeste_image_generation.core.enums import GoogleModel
from celeste_image_generation.core.types import GeneratedImage, ImagePrompt


class GoogleImageGenerator(BaseImageGenerator):
    def __init__(self, model: str = GoogleModel.IMAGEN_3.value, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model_name = model

    async def generate_image(self, prompt: ImagePrompt, **kwargs: Any) -> List[GeneratedImage]:
        response = await self.client.aio.models.generate_images(
            model=self.model_name,
            prompt=prompt.content,
        )
        
        return [
            GeneratedImage(image=img.image.image_bytes, metadata={"model": self.model_name})
            for img in response.generated_images
        ]