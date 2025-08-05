"""Utility functions for image generation providers."""

import base64
import aiohttp


async def decode_image_response(
    image_data: dict, session: aiohttp.ClientSession
) -> bytes:
    """
    Decode image data from API response, handling both base64 and URL formats.

    Args:
        image_data: Dictionary containing either 'b64_json' or 'url' key
        session: Active aiohttp session for downloading images from URLs

    Returns:
        Image data as bytes

    Raises:
        ValueError: If neither b64_json nor url is found in image_data
        Exception: If image download fails
    """
    if "b64_json" in image_data:
        return base64.b64decode(image_data["b64_json"])
    elif "url" in image_data:
        async with session.get(image_data["url"]) as img_response:
            if img_response.status != 200:
                raise Exception(
                    f"Failed to download image from URL: {image_data['url']}"
                )
            return await img_response.read()
    else:
        raise ValueError("Image data must contain either 'b64_json' or 'url'")
