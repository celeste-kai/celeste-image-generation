import base64
import io
from typing import Any, List, Optional
import aiohttp
from PIL import Image

from celeste_image_generation.base import BaseImageGenerator
from celeste_image_generation.core.config import HUGGINGFACE_TOKEN
from celeste_image_generation.core.enums import HuggingFaceModel
from celeste_image_generation.core.types import GeneratedImage, ImagePrompt


class HuggingFaceImageGenerator(BaseImageGenerator):
    """Hugging Face Inference API image generator."""
    
    def __init__(self, model: str = HuggingFaceModel.FLUX_SCHNELL.value, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_key = HUGGINGFACE_TOKEN
        if not self.api_key:
            raise ValueError("Hugging Face API key not provided. Set HUGGINGFACE_TOKEN environment variable.")
        
        self.model_name = model.value if hasattr(model, 'value') else model
        self.base_url = "https://api-inference.huggingface.co/models"
        
    async def generate_image(self, prompt: ImagePrompt, **kwargs: Any) -> List[GeneratedImage]:
        """
        Generate images using Hugging Face Inference API.
        
        Args:
            prompt: The image prompt
            **kwargs: Additional parameters supported by the model
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Build request data
        data = {
            "inputs": prompt.content,
        }
        
        # Add any additional parameters the model might support
        if kwargs:
            data["parameters"] = kwargs
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/{self.model_name}",
                headers=headers,
                json=data
            ) as response:
                if response.status == 503:
                    # Model is loading
                    error_data = await response.json()
                    estimated_time = error_data.get("estimated_time", 20)
                    raise Exception(f"Model is loading. Please try again in {estimated_time} seconds.")
                elif response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Hugging Face API error ({response.status}): {error_text}")
                
                # The API returns the image directly as bytes
                image_bytes = await response.read()
                
                # Verify it's a valid image
                try:
                    img = Image.open(io.BytesIO(image_bytes))
                    img.verify()
                except Exception as e:
                    raise Exception(f"Invalid image data received: {e}")
                
                return [GeneratedImage(
                    image=image_bytes,
                    metadata={
                        "model": self.model_name,
                        "provider": "huggingface"
                    }
                )]