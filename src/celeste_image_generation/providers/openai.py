from typing import Any, List
import base64
import aiohttp

from celeste_image_generation.base import BaseImageGenerator
from celeste_image_generation.core.config import OPENAI_API_KEY
from celeste_image_generation.core.enums import OpenAIModel
from celeste_image_generation.core.types import GeneratedImage, ImagePrompt


class OpenAIImageGenerator(BaseImageGenerator):
    """OpenAI image generator using DALL-E models."""
    
    def __init__(self, model: str = OpenAIModel.DALL_E_3.value, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_key = OPENAI_API_KEY
        self.model_name = model.value if hasattr(model, 'value') else model
        self.base_url = "https://api.openai.com/v1"
        
    async def generate_image(self, prompt: ImagePrompt, **kwargs: Any) -> List[GeneratedImage]:
        """
        Generate images using OpenAI's image generation API.
        """
        # Set defaults based on model
        if self.model_name == OpenAIModel.DALL_E_3.value:
            size = kwargs.get("size", "1024x1024")
            quality = kwargs.get("quality", "standard")
            style = kwargs.get("style", "vivid")
            n = 1  # DALL-E 3 only supports n=1
        else:  # DALL-E 2
            size = kwargs.get("size", "1024x1024")
            quality = None
            style = None
            n = kwargs.get("n", 1)
            
        response_format = kwargs.get("response_format", "b64_json")
        
        # Build request
        data = {
            "model": self.model_name,
            "prompt": prompt.content,
            "size": size,
            "n": n
        }
        
        # Only add response_format for DALL-E models (not gpt-image-1)
        if self.model_name in [OpenAIModel.DALL_E_2.value, OpenAIModel.DALL_E_3.value]:
            data["response_format"] = response_format
        
        # Add DALL-E 3 specific parameters
        if self.model_name == OpenAIModel.DALL_E_3.value:
            if quality:
                data["quality"] = quality
            if style:
                data["style"] = style
                
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/images/generations",
                json=data,
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error ({response.status}): {error_text}")
                
                result = await response.json()
                
                images = []
                for img_data in result["data"]:
                    # Handle different response formats
                    if "b64_json" in img_data:
                        image_bytes = base64.b64decode(img_data["b64_json"])
                    elif "url" in img_data:
                        # Fetch the image from the URL
                        async with session.get(img_data["url"]) as img_response:
                            image_bytes = await img_response.read()
                    else:
                        # For gpt-image-1 or other models that might have different response structure
                        # Try to find the image data in the response
                        if "image" in img_data:
                            # Could be base64 or URL
                            if img_data["image"].startswith("http"):
                                async with session.get(img_data["image"]) as img_response:
                                    image_bytes = await img_response.read()
                            else:
                                # Assume it's base64
                                image_bytes = base64.b64decode(img_data["image"])
                        else:
                            raise Exception(f"Unknown image data format in response: {img_data}")
                    
                    # Extract revised prompt if available (DALL-E 3 feature)
                    metadata = {
                        "model": self.model_name,
                        "size": size,
                    }
                    if "revised_prompt" in img_data:
                        metadata["revised_prompt"] = img_data["revised_prompt"]
                    if quality:
                        metadata["quality"] = quality
                    if style:
                        metadata["style"] = style
                        
                    images.append(GeneratedImage(
                        image=image_bytes,
                        metadata=metadata
                    ))
                    
                return images
