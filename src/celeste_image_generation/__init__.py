"""
Celeste Image Generation: A unified image generation interface for multiple providers.
"""

from typing import Any

from .base import BaseImageGenerator
from .core.enums import Provider
from .core.types import GeneratedImage

__version__ = "0.1.0"

SUPPORTED_PROVIDERS = [
    "google",
    "stabilityai",
    "local",
    "openai",
    "huggingface",
    "luma",
    "xai",
]


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

    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")

    if provider == "google":
        from .providers.google import GoogleImageGenerator

        return GoogleImageGenerator(**kwargs)
    
    if provider == "stabilityai":
        from .providers.stability_ai import StabilityAIImageGenerator

        return StabilityAIImageGenerator(**kwargs)
    
    if provider == "local":
        from .providers.local import LocalImageGenerator

        return LocalImageGenerator(**kwargs)
    
    if provider == "openai":
        from .providers.openai import OpenAIImageGenerator

        return OpenAIImageGenerator(**kwargs)
    
    if provider == "huggingface":
        from .providers.huggingface import HuggingFaceImageGenerator

        return HuggingFaceImageGenerator(**kwargs)

    if provider == "luma":
        from .providers.luma import LumaImageGenerator

        return LumaImageGenerator(**kwargs)

    if provider == "xai":
        from .providers.xai import XAIImageGenerator

        return XAIImageGenerator(**kwargs)

    raise ValueError(f"Provider {provider} not implemented")


__all__ = [
    "create_image_generator",
    "BaseImageGenerator",
    "Provider",
    "GeneratedImage",
    "__version__",
]
