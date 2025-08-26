from celeste_core.enums.capability import Capability
from celeste_core.enums.providers import Provider

# Capability for this domain package
CAPABILITY: Capability = Capability.IMAGE_GENERATION

# Provider wiring for image generation clients
PROVIDER_MAPPING: dict[Provider, tuple[str, str]] = {
    Provider.GOOGLE: ("providers.google", "GoogleImageGenerator"),
    Provider.STABILITYAI: ("providers.stability_ai", "StabilityAIImageGenerator"),
    Provider.LOCAL: ("providers.local", "LocalImageGenerator"),
    Provider.OPENAI: ("providers.openai", "OpenAIImageGenerator"),
    Provider.HUGGINGFACE: ("providers.huggingface", "HuggingFaceImageGenerator"),
    Provider.LUMA: ("providers.luma", "LumaImageGenerator"),
    Provider.XAI: ("providers.xai", "XAIImageGenerator"),
    Provider.REPLICATE: ("providers.replicate", "ReplicateImageGenerator"),
}

__all__ = ["CAPABILITY", "PROVIDER_MAPPING"]
