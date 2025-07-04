from abc import ABC, abstractmethod
from typing import Any, List

from celeste_image_generation.core.types import GeneratedImage

from celeste_image_generation.core.types import ImagePrompt


class BaseImageGenerator(ABC):
    """
    Abstract base class for all image generation clients.
    It defines the standard interface for interacting with different image providers.
    """

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the client, loading credentials from the environment.
        Provider-specific arguments can be passed via kwargs.
        """
        pass

    @abstractmethod
    async def generate_image(self, prompt: ImagePrompt, **kwargs: Any) -> List[GeneratedImage]:
        """
        Submits a request to start an image generation job.
        """
        pass

