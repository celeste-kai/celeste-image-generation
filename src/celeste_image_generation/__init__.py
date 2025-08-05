"""
Celeste Image Generation: A unified image generation interface for multiple providers.
"""

from typing import Any

from .base import BaseImageGenerator
from .core.enums import Provider
from .core.types import GeneratedImage

__version__ = "0.1.0"


def create_image_generator(provider: str, **kwargs: Any) -> "BaseImageGenerator":
    """
    Factory function to create an image generator instance based on the provider.

    Args:
        provider: The image generator provider to use (string or Provider enum).
        **kwargs: Additional arguments to pass to the image generator constructor.

    Returns:
        An instance of an image generator
    """
    # Convert Provider enum to string if needed
    if isinstance(provider, Provider):
        provider = provider.value

    # Lazy import mapping
    provider_mapping = {
        "google": ("providers.google", "GoogleImageGenerator"),
        "stabilityai": ("providers.stability_ai", "StabilityAIImageGenerator"),
        "local": ("providers.local", "LocalImageGenerator"),
        "openai": ("providers.openai", "OpenAIImageGenerator"),
        "huggingface": ("providers.huggingface", "HuggingFaceImageGenerator"),
        "luma": ("providers.luma", "LumaImageGenerator"),
        "xai": ("providers.xai", "XAIImageGenerator"),
    }

    if provider not in provider_mapping:
        raise ValueError(
            f"Unsupported provider: {provider}. Supported providers: {list(provider_mapping.keys())}"
        )

    module_path, class_name = provider_mapping[provider]
    module = __import__(
        f"celeste_image_generation.{module_path}", fromlist=[class_name]
    )
    generator_class = getattr(module, class_name)

    return generator_class(**kwargs)


__all__ = [
    "create_image_generator",
    "BaseImageGenerator",
    "Provider",
    "GeneratedImage",
    "__version__",
]
