import asyncio
import streamlit as st
from dotenv import load_dotenv

from celeste_image_generation import create_image_generator
from celeste_image_generation.core.enums import (
    Provider,
    GoogleModel,
    StabilityModel,
    LocalModel,
    OpenAIModel,
    HuggingFaceModel,
    LumaModel,
)
from celeste_image_generation.core.types import ImagePrompt

load_dotenv()


def display_image_grid(results):
    """Display images in a responsive grid layout."""
    num_images = len(results)

    if num_images == 1:
        # Single large image
        st.image(results[0].image, caption="Generated Image", use_container_width=True)
        with st.expander("Details"):
            st.json(results[0].metadata or "No metadata available.")
    elif num_images <= 4:
        # For 2-4 images, use appropriate column layout
        cols_per_row = min(num_images, 2) if num_images <= 3 else 2
        rows = (num_images + cols_per_row - 1) // cols_per_row

        img_idx = 0
        for row in range(rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                if img_idx < num_images:
                    with cols[col_idx]:
                        st.image(
                            results[img_idx].image,
                            caption=f"Image {img_idx+1}",
                            use_container_width=True,
                        )
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
                    st.image(
                        results[img_idx].image,
                        caption=f"Image {img_idx+1}",
                        use_container_width=True,
                    )
                    with st.expander("Details"):
                        st.json(results[img_idx].metadata or "No metadata available.")


async def main() -> None:
    """
    Streamlit application for generating images using the celeste-image-generation library.
    """
    st.title("Celeste Image Generation")

    with st.sidebar:
        st.header("Configuration")
        provider_options = [
            "google",
            "stabilityai",
            "local",
            "openai",
            "huggingface",
            "luma",
        ]
        provider_name = st.selectbox("Provider", options=provider_options, index=0)
        provider = Provider(provider_name)

        # Get the appropriate model enum based on provider
        model_enum_map = {
            Provider.GOOGLE: GoogleModel,
            Provider.STABILITYAI: StabilityModel,
            Provider.LOCAL: LocalModel,
            Provider.OPENAI: OpenAIModel,
            Provider.HUGGINGFACE: HuggingFaceModel,
            Provider.LUMA: LumaModel,
        }

        model_enum_class = model_enum_map.get(provider)

        if model_enum_class:
            model_options = list(model_enum_class)
            selected_model_enum = st.selectbox(
                "Model", options=model_options, format_func=lambda x: x.value, index=0
            )
            selected_model = selected_model_enum.value
        else:
            selected_model = None

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
                image_prompt = ImagePrompt(content=prompt_text)

                # Prepare kwargs based on provider
                kwargs = {}
                if provider == Provider.GOOGLE:
                    kwargs["n"] = num_images
                elif provider == Provider.LUMA:
                    kwargs["aspect_ratio"] = aspect_ratio

                results = await image_generator.generate_image(image_prompt, **kwargs)
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
