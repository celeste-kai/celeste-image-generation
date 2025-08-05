from enum import Enum


class Provider(Enum):
    OPENAI = "openai"
    STABILITYAI = "stabilityai"
    GOOGLE = "google"
    LOCAL = "local"
    HUGGINGFACE = "huggingface"
    LUMA = "luma"
    XAI = "xai"


class GoogleModel(Enum):
    """Google Imagen model enumeration."""

    # Imagen 4
    IMAGEN_4 = "imagen-4.0-generate-preview-06-06"
    IMAGEN_4_ULTRA = "imagen-4.0-ultra-generate-preview-06-06"

    # Imagen 3
    IMAGEN_3 = "imagen-3.0-generate-002"


class StabilityModel(Enum):
    """Stability AI model enumeration."""

    # New models (v2beta API)
    ULTRA = "ultra"  # Stable Image Ultra
    SD3_5_LARGE = "sd3.5-large"  # Stable Diffusion 3.5 Large
    SD3_5_LARGE_TURBO = "sd3.5-large-turbo"  # Stable Diffusion 3.5 Large Turbo
    SD3_5_MEDIUM = "sd3.5-medium"  # Stable Diffusion 3.5 Medium
    CORE = "core"  # Stable Image Core

    # Legacy models (v1 API)
    SDXL_1_0 = "stable-diffusion-xl-1024-v1-0"  # SDXL 1.0
    SD_1_6 = "stable-diffusion-v1-6"  # SD 1.6


# Stability AI model credit costs
STABILITY_CREDITS = {
    StabilityModel.ULTRA: 8.0,
    StabilityModel.SD3_5_LARGE: 6.5,
    StabilityModel.SD3_5_LARGE_TURBO: 4.0,
    StabilityModel.SD3_5_MEDIUM: 3.5,
    StabilityModel.CORE: 3.0,
    StabilityModel.SDXL_1_0: 0.9,
    StabilityModel.SD_1_6: 0.9,
}


class OpenAIModel(Enum):
    """OpenAI model enumeration."""

    # Current models
    DALL_E_3 = "dall-e-3"
    DALL_E_2 = "dall-e-2"

    # Future model (based on your findings)
    GPT_IMAGE_1 = "gpt-image-1"


class HuggingFaceModel(Enum):
    """Hugging Face Inference API model enumeration."""

    # FLUX models
    FLUX_SCHNELL = "black-forest-labs/FLUX.1-schnell"
    FLUX_DEV = "black-forest-labs/FLUX.1-dev"
    FLUX_KREA_DEV = "black-forest-labs/FLUX.1-Krea-dev"

    # Stable Diffusion XL
    SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"

    # Stable Diffusion 3
    SD3_MEDIUM = "stabilityai/stable-diffusion-3-medium-diffusers"

    # Qwen models
    QWEN_IMAGE = "Qwen/Qwen-Image"


class LumaModel(Enum):
    """Luma Labs Dream Machine model enumeration."""

    PHOTON_1 = "photon-1"  # Default model for image generation
    PHOTON_FLASH_1 = "photon-flash-1"  # Faster model for image generation


class XAIModel(Enum):
    """xAI image model enumeration."""

    GROK_2_IMAGE = "grok-2-image"


class LocalModel(Enum):
    """Local model enumeration for open source models."""

    # FLUX models
    FLUX_SCHNELL = "black-forest-labs/FLUX.1-schnell"
    FLUX_DEV = "black-forest-labs/FLUX.1-dev"

    # SDXL variants
    SDXL_TURBO = "stabilityai/sdxl-turbo"
    SDXL_LIGHTNING = "ByteDance/SDXL-Lightning"
    SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"

    # Other models
    SD_2_1 = "stabilityai/stable-diffusion-2-1"
