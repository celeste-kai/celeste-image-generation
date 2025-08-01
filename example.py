import asyncio
import streamlit as st
from dotenv import load_dotenv

from celeste_image_generation import create_image_generator
from celeste_image_generation.core.enums import Provider, GoogleModel, StabilityModel, LocalModel, OpenAIModel, HuggingFaceModel, LumaModel, STABILITY_CREDITS
from celeste_image_generation.core.types import ImagePrompt

load_dotenv()


async def main() -> None:
    """
    Streamlit application for generating images using the celeste-image-generation library.
    """
    st.title("Celeste Image Generation")

    with st.sidebar:
        st.header("Configuration")
        provider_options = ["google", "stabilityai", "local", "openai", "huggingface", "luma"]
        provider_name = st.selectbox("Provider", options=provider_options, index=0)
        provider = Provider(provider_name)

        if provider == Provider.GOOGLE:
            model_options = list(GoogleModel)
            selected_model_enum = st.selectbox(
                "Model",
                options=model_options,
                format_func=lambda x: x.value,
                index=0
            )
            selected_model = selected_model_enum.value

            # Allow selecting the number of images to generate
            num_images = st.number_input(
                "Number of Images",
                min_value=1,
                max_value=4,
                value=1,
                step=1,
            )
        elif provider == Provider.STABILITYAI:
            model_options = list(StabilityModel)
            selected_model_enum = st.selectbox(
                "Model",
                options=model_options,
                format_func=lambda x: x.value,
                index=0
            )
            selected_model = selected_model_enum.value
            
            # Display credit cost
            credits = STABILITY_CREDITS.get(selected_model_enum, "Unknown")
            st.info(f"💳 Credit cost: {credits} credits per image")
        elif provider == Provider.LOCAL:
            model_options = list(LocalModel)
            selected_model_enum = st.selectbox(
                "Model",
                options=model_options,
                format_func=lambda x: x.value,
                index=0
            )
            selected_model = selected_model_enum.value
            
            # Display model info
            model_info = {
                LocalModel.FLUX_SCHNELL: "🚀 4 steps, state-of-the-art quality (23.8 GB)",
                LocalModel.SDXL_TURBO: "⚡ 1-4 steps, very fast (~6.5 GB)",
                LocalModel.SDXL_LIGHTNING: "⚡ 1-4 steps, ByteDance optimized (6.94 GB)",
                LocalModel.SDXL_BASE: "🎨 50 steps, high quality (6.94 GB)",
                LocalModel.SD_2_1: "🎯 Classic Stable Diffusion 2.1 (5.21 GB)"
            }
            st.info(model_info.get(selected_model_enum, ""))
            st.warning("⚠️ Models will be downloaded on first use")
        elif provider == Provider.OPENAI:
            model_options = list(OpenAIModel)
            selected_model_enum = st.selectbox(
                "Model",
                options=model_options,
                format_func=lambda x: x.value,
                index=0
            )
            selected_model = selected_model_enum.value
            
            # Display model info and pricing
            if selected_model_enum == OpenAIModel.DALL_E_3:
                st.info("🎨 DALL-E 3: Latest model with advanced capabilities")
                st.write("**Pricing:**")
                st.write("- Standard 1024x1024: $0.04/image")
                st.write("- Standard 1024x1792/1792x1024: $0.08/image")
                st.write("- HD quality: 2x the standard price")
                
                # Add quality and style options
                quality = st.selectbox("Quality", options=["standard", "hd"])
                style = st.selectbox("Style", options=["vivid", "natural"])
                size = st.selectbox("Size", options=["1024x1024", "1024x1792", "1792x1024"])
            elif selected_model_enum == OpenAIModel.DALL_E_2:
                st.info("🎨 DALL-E 2: Previous generation model")
                st.write("**Pricing:** $0.02 per 1024x1024 image")
                size = "1024x1024"
                quality = None
                style = None
            elif selected_model_enum == OpenAIModel.GPT_IMAGE_1:
                st.info("🚀 GPT Image 1: Future model (not yet available)")
                size = "1024x1024"
                quality = None
                style = None
        elif provider == Provider.HUGGINGFACE:
            model_options = list(HuggingFaceModel)
            selected_model_enum = st.selectbox(
                "Model",
                options=model_options,
                format_func=lambda x: x.value,
                index=0
            )
            selected_model = selected_model_enum.value
            
            # Display model info
            st.info("🤗 Hugging Face Inference API")
            st.write("**Free tier**: 1000 requests/month")
            st.write("**Pro tier**: Higher rate limits")
            
            # Model-specific info
            model_descriptions = {
                HuggingFaceModel.FLUX_SCHNELL: "⚡ FLUX Schnell: Fast 4-step generation",
                HuggingFaceModel.FLUX_DEV: "🎨 FLUX Dev: High quality development model",
                HuggingFaceModel.SDXL_BASE: "🖼️ SDXL Base: Stable Diffusion XL 1.0",
                HuggingFaceModel.SDXL_TURBO: "🚀 SDXL Turbo: Fast 1-4 step generation",
                HuggingFaceModel.SD3_MEDIUM: "🆕 SD3 Medium: Latest Stable Diffusion 3",
                HuggingFaceModel.PLAYGROUND_V2_5: "🎪 Playground v2.5: Aesthetic focused",
                HuggingFaceModel.KANDINSKY_3: "🎨 Kandinsky 3: Russian text-to-image model",
                HuggingFaceModel.SDXL_LIGHTNING: "⚡ SDXL Lightning: ByteDance fast model",
            }
            if selected_model_enum in model_descriptions:
                st.info(model_descriptions[selected_model_enum])
            
            st.warning("⚠️ Model may need to load on first use (20-60s)")
        elif provider == Provider.LUMA:
            model_options = list(LumaModel)
            selected_model_enum = st.selectbox(
                "Model",
                options=model_options,
                format_func=lambda x: x.value,
                index=0
            )
            selected_model = selected_model_enum.value
            
            # Display model info
            st.info("🌟 Luma AI Dream Machine")
            st.write("**State-of-the-art image generation**")
            
            # Model descriptions
            if selected_model_enum == LumaModel.PHOTON_1:
                st.info("📸 Photon-1: High quality default model")
            elif selected_model_enum == LumaModel.PHOTON_FLASH_1:
                st.info("⚡ Photon Flash-1: Faster generation")
            
            # Aspect ratio selector
            aspect_ratio = st.selectbox(
                "Aspect Ratio",
                options=["16:9", "1:1", "3:4", "4:3", "9:16", "9:21", "21:9"],
                index=0
            )
            
            st.write("**Features:**")
            st.write("- Image references (up to 4)")
            st.write("- Style transfer")
            st.write("- Character consistency")
            st.write("- Image modification")
            
            # Optional reference inputs (commented out for simplicity in UI)
            # st.text_input("Image Reference URLs (comma-separated)", key="luma_image_ref")
            # st.text_input("Style Reference URL", key="luma_style_ref")
            # st.text_input("Character Reference URL", key="luma_char_ref")
            # st.text_input("Image to Modify URL", key="luma_modify_ref")
        else:
            selected_model = None

    prompt_text = st.text_area("Enter your image prompt:", height=150)

    if st.button("Generate Image") and prompt_text:
        if not selected_model:
            st.error("Please select a valid model in the sidebar.")
            return

        with st.spinner("Generating image..."):
            try:
                image_generator = create_image_generator(provider, model=selected_model)
                image_prompt = ImagePrompt(content=prompt_text)
                
                # Prepare kwargs based on provider
                kwargs = {}
                if provider == Provider.GOOGLE:
                    kwargs["n"] = num_images
                if provider == Provider.OPENAI and selected_model == OpenAIModel.DALL_E_3.value:
                    kwargs["size"] = size
                    kwargs["quality"] = quality
                    kwargs["style"] = style
                elif provider == Provider.LUMA:
                    kwargs["aspect_ratio"] = aspect_ratio
                
                results = await image_generator.generate_image(image_prompt, **kwargs)
            except Exception as e:
                st.error(f"Error generating image: {str(e)}")
                return

            if results:
                num_images = len(results)
                
                # Create responsive grid layout based on number of images
                if num_images == 1:
                    # Single large image
                    st.image(results[0].image, caption="Generated Image", use_container_width=True)
                    with st.expander("Details"):
                        st.json(results[0].metadata or "No metadata available.")
                
                elif num_images == 2:
                    # 2 images in a row
                    cols = st.columns(2)
                    for i, (col, result) in enumerate(zip(cols, results)):
                        with col:
                            st.image(result.image, caption=f"Image {i+1}", use_container_width=True)
                            with st.expander(f"Details"):
                                st.json(result.metadata or "No metadata available.")
                
                elif num_images == 3:
                    # 3 images in a row
                    cols = st.columns(3)
                    for i, (col, result) in enumerate(zip(cols, results)):
                        with col:
                            st.image(result.image, caption=f"Image {i+1}", use_container_width=True)
                            with st.expander(f"Details"):
                                st.json(result.metadata or "No metadata available.")
                
                elif num_images == 4:
                    # 2x2 grid
                    for row in range(2):
                        cols = st.columns(2)
                        for col_idx in range(2):
                            img_idx = row * 2 + col_idx
                            with cols[col_idx]:
                                st.image(results[img_idx].image, caption=f"Image {img_idx+1}", use_container_width=True)
                                with st.expander(f"Details"):
                                    st.json(results[img_idx].metadata or "No metadata available.")
                
                else:
                    # For more than 4 images, use 3-column grid
                    for i in range(0, num_images, 3):
                        cols = st.columns(3)
                        for j in range(min(3, num_images - i)):
                            with cols[j]:
                                img_idx = i + j
                                st.image(results[img_idx].image, caption=f"Image {img_idx+1}", use_container_width=True)
                                with st.expander(f"Details"):
                                    st.json(results[img_idx].metadata or "No metadata available.")
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
