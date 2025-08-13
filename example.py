import asyncio
from typing import List, Optional, Union

import streamlit as st
from celeste_core import ImageArtifact, Provider, list_models
from celeste_core.enums.capability import Capability
from celeste_image_generation import create_image_generator
from dotenv import load_dotenv

load_dotenv()


def _get_image_displayable(artifact: ImageArtifact) -> Optional[Union[bytes, str]]:
    """Return bytes or path that Streamlit can display for an ImageArtifact."""
    if artifact.data:
        return artifact.data
    if artifact.path:
        return artifact.path
    return None


def display_image_grid(results: List[ImageArtifact]) -> None:
    """Display images in a responsive grid layout."""
    num_images = len(results)

    if num_images == 1:
        # Single large image
        img0 = _get_image_displayable(results[0])
        if img0 is not None:
            st.image(img0, caption="Generated Image", use_container_width=True)
        else:
            st.warning("No image content returned.")
        with st.expander("Details"):
            st.json(results[0].metadata or "No metadata available.")
    elif num_images <= 4:
        # For 2-4 images, use appropriate column layout
        cols_per_row = min(num_images, 2) if num_images <= 3 else 2
        rows = (num_images + cols_per_row - 1) // cols_per_row

        img_idx = 0
        for _row in range(rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                if img_idx < num_images:
                    with cols[col_idx]:
                        img_obj = _get_image_displayable(results[img_idx])
                        if img_obj is not None:
                            st.image(
                                img_obj,
                                caption=f"Image {img_idx + 1}",
                                use_container_width=True,
                            )
                        else:
                            st.warning(f"Image {img_idx + 1} had no content.")
                        with st.expander("Details"):
                            st.json(
                                results[img_idx].metadata or "No metadata available."
                            )
                    img_idx += 1
    else:
        # For more than 4 images, use 3-column grid
        for i in range(0, num_images, 3):
            cols = st.columns(3)
            for j in range(min(3, num_images - i)):
                with cols[j]:
                    img_idx = i + j
                    img_obj = _get_image_displayable(results[img_idx])
                    if img_obj is not None:
                        st.image(
                            img_obj,
                            caption=f"Image {img_idx + 1}",
                            use_container_width=True,
                        )
                    else:
                        st.warning(f"Image {img_idx + 1} had no content.")
                    with st.expander("Details"):
                        st.json(results[img_idx].metadata or "No metadata available.")


async def main() -> None:
    """
    Streamlit application for generating images using the library.
    """
    st.title("Celeste Image Generation")

    with st.sidebar:
        st.header("Configuration")
        # Derive providers that have IMAGE_GENERATION models in registry
        providers = sorted(
            {m.provider for m in list_models(capability=Capability.IMAGE_GENERATION)},
            key=lambda p: p.value,
        )
        display_providers = [p.value for p in providers]
        provider_name = st.selectbox("Provider", options=display_providers, index=0)
        provider = Provider(provider_name)

        # Load models from registry by provider and IMAGE_GENERATION capability
        models = list_models(provider=provider, capability=Capability.IMAGE_GENERATION)
        model_options = [m.id for m in models]
        display_names = [m.display_name or m.id for m in models]
        # Map display name to id for selection
        display_to_id = {display_names[i]: model_options[i] for i in range(len(models))}
        selected_display = st.selectbox("Model", options=display_names, index=0)
        selected_model = display_to_id[selected_display] if selected_display else None

        # Provider-specific options
        if provider == Provider.GOOGLE:
            # Allow selecting the number of images to generate
            num_images = st.number_input(
                "Number of Images",
                min_value=1,
                max_value=4,
                value=1,
                step=1,
            )
        elif provider == Provider.LUMA:
            # Aspect ratio selector
            aspect_ratio = st.selectbox(
                "Aspect Ratio",
                options=["16:9", "1:1", "3:4", "4:3", "9:16", "9:21", "21:9"],
                index=0,
            )

    prompt_text = st.text_area("Enter your image prompt:", height=150)

    if st.button("Generate Image") and prompt_text:
        if not selected_model:
            st.error("Please select a valid model in the sidebar.")
            return

        with st.spinner("Generating image..."):
            try:
                image_generator = create_image_generator(
                    provider.value, model=selected_model
                )
                # Prepare kwargs based on provider
                kwargs = {}
                if provider == Provider.GOOGLE:
                    kwargs["n"] = num_images
                elif provider == Provider.LUMA:
                    kwargs["aspect_ratio"] = aspect_ratio
                results = await image_generator.generate_image(prompt_text, **kwargs)
            except Exception as e:
                st.error(f"Error generating image: {str(e)}")
                # Show full error details in an expander
                with st.expander("Full error details"):
                    st.code(str(e))
                return

            if results:
                display_image_grid(results)
            else:
                st.warning("The model did not return any images.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "cannot run current event loop" in str(e):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
        else:
            raise
