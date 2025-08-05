import io
from typing import Any, List
from huggingface_hub import InferenceClient

from celeste_image_generation.base import BaseImageGenerator
from celeste_image_generation.core.config import HUGGINGFACE_TOKEN
from celeste_image_generation.core.enums import HuggingFaceModel
from celeste_image_generation.core.types import GeneratedImage, ImagePrompt


class HuggingFaceImageGenerator(BaseImageGenerator):
    """Hugging Face Inference API image generator."""

    def __init__(
        self, model: str = HuggingFaceModel.FLUX_SCHNELL.value, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.api_key = HUGGINGFACE_TOKEN
        if not self.api_key:
            raise ValueError(
                "Hugging Face API key not provided. Set HUGGINGFACE_TOKEN environment variable."
            )

        self.model_name = model.value if hasattr(model, "value") else model

        # Initialize InferenceClient with automatic provider detection
        self.client = InferenceClient(token=self.api_key)

    async def generate_image(
        self, prompt: ImagePrompt, **kwargs: Any
    ) -> List[GeneratedImage]:
        """
        Generate images using Hugging Face Inference Client.
        """
        try:
            # Log the request details
            print(f"[HuggingFace] Requesting model: {self.model_name}")
            print(f"[HuggingFace] Prompt: {prompt.content}")

            # Use InferenceClient to generate image
            # This will automatically detect the correct provider (e.g., fal-ai for FLUX.1-Krea-dev)
            image = self.client.text_to_image(
                prompt.content, model=self.model_name, **kwargs
            )

            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            image_bytes = img_byte_arr.getvalue()

            return [
                GeneratedImage(
                    image=image_bytes,
                    metadata={"model": self.model_name, "provider": "huggingface"},
                )
            ]

        except Exception as e:
            # Provide detailed error information
            error_msg = str(e)
            if "503" in error_msg:
                raise Exception("Model is loading. Please try again in a few seconds.")
            elif "401" in error_msg:
                raise Exception(
                    f"Authentication error. Make sure your HUGGINGFACE_TOKEN has access to model '{self.model_name}'"
                )
            elif "404" in error_msg:
                raise Exception(
                    f"Model '{self.model_name}' not found or not available through Inference API"
                )
            else:
                raise Exception(
                    f"HuggingFace Inference error for model '{self.model_name}': {error_msg}"
                )
