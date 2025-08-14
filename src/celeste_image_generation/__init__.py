"""
Celeste Image Generation: A unified image generation interface for multiple providers.
"""

from typing import Any

from celeste_core import ImageArtifact, Provider
from celeste_core.base.image_generator import BaseImageGenerator
from celeste_core.config.settings import settings

__version__ = "0.1.0"


SUPPORTED_PROVIDERS: set[Provider] = {
    Provider.GOOGLE,
    Provider.STABILITYAI,
    Provider.LOCAL,
    Provider.OPENAI,
    Provider.HUGGINGFACE,
    Provider.LUMA,
    Provider.XAI,
    Provider.REPLICATE,
}


def create_image_generator(
    provider: str | Provider, **kwargs: Any
) -> "BaseImageGenerator":
    """
    Factory function to create an image generator instance based on the provider.

    Args:
        provider: The image generator provider to use (string or Provider enum).
        **kwargs: Additional arguments to pass to the image generator constructor.

    Returns:
        An instance of an image generator
    """
    # Normalize to enum
    provider_enum: Provider = (
        provider if isinstance(provider, Provider) else Provider(provider)
    )

    # Mapping keyed by Provider enum
    provider_mapping = {
        Provider.GOOGLE: ("providers.google", "GoogleImageGenerator"),
        Provider.STABILITYAI: ("providers.stability_ai", "StabilityAIImageGenerator"),
        Provider.LOCAL: ("providers.local", "LocalImageGenerator"),
        Provider.OPENAI: ("providers.openai", "OpenAIImageGenerator"),
        Provider.HUGGINGFACE: ("providers.huggingface", "HuggingFaceImageGenerator"),
        Provider.LUMA: ("providers.luma", "LumaImageGenerator"),
        Provider.XAI: ("providers.xai", "XAIImageGenerator"),
        Provider.REPLICATE: ("providers.replicate", "ReplicateImageGenerator"),
    }

    if (
        provider_enum not in SUPPORTED_PROVIDERS
        or provider_enum not in provider_mapping
    ):
        supported = [p.value for p in provider_mapping.keys()]
        raise ValueError(
            f"Unsupported provider: {provider_enum.value}. "
            f"Supported providers: {supported}"
        )

    # Validate environment for the chosen provider
    settings.validate_for_provider(provider_enum.value)

    module_path, class_name = provider_mapping[provider_enum]
    module = __import__(
        f"celeste_image_generation.{module_path}", fromlist=[class_name]
    )
    generator_class = getattr(module, class_name)

    return generator_class(**kwargs)


__all__ = [
    "create_image_generator",
    "BaseImageGenerator",
    "Provider",
    "ImageArtifact",
    "__version__",
]
