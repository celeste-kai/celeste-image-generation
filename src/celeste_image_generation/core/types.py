from typing import Optional

from pydantic import BaseModel


class ImagePrompt(BaseModel):
    content: str


class GeneratedImage(BaseModel):
    """Represents a single generated image and its metadata"""

    image: bytes
    metadata: Optional[dict] = None
