import streamlit as st
from dotenv import load_dotenv

from celeste_video_generation import create_video_generator
from celeste_video_generation.core.enums import Provider, GoogleModel, LumaModel, LumaResolution, LumaDuration
from celeste_video_generation.core.types import VideoPrompt

load_dotenv()


def main() -> None:
    """
    Streamlit application for generating videos using the celeste-video-generation library.
    """
    st.title("Celeste Video Generation")

    with st.sidebar:
        st.header("Configuration")
        # Provider selection
        provider_options = ["google", "local", "luma"]
        provider_name = st.selectbox("Provider", provider_options, index=0)
        provider = Provider(provider_name)

        # Model selection based on provider
        if provider == Provider.GOOGLE:
            selected_model = st.selectbox(
                "Model",
                options=[model for model in GoogleModel],
                format_func=lambda x: x.value,
                index=0
            )
        elif provider == Provider.LOCAL:
            model_id = st.text_input(
                "Model ID", 
                value="ali-vilab/text-to-video-ms-1.7b",
                help="Hugging Face model ID for video generation"
            )
            num_inference_steps = st.slider("Inference Steps", 10, 50, 25)
            num_frames = st.slider("Number of Frames", 8, 32, 16)
        elif provider == Provider.LUMA:
            selected_model = st.selectbox(
                "Model",
                options=[model for model in LumaModel],
                format_func=lambda x: x.name,
                index=0
            )
            selected_resolution = st.selectbox(
                "Resolution",
                options=[res for res in LumaResolution],
                format_func=lambda x: x.value,
                index=1  # Default to 720p
            )
            selected_duration = st.selectbox(
                "Duration",
                options=[dur for dur in LumaDuration],
                format_func=lambda x: x.value,
                index=0  # Default to 5s
            )
            loop_video = st.checkbox("Loop", value=False)
            aspect_ratio = st.selectbox(
                "Aspect Ratio",
                options=["16:9", "9:16", "1:1", "4:3", "3:4"],
                index=0
            )
        else:
            selected_model = None

    prompt_text = st.text_area("Enter your video prompt:", height=150)

    if st.button("Generate Video") and prompt_text:
        with st.spinner("Generating video... This may take a few minutes."):
            try:
                # Create generator with appropriate parameters
                if provider == Provider.GOOGLE:
                    video_generator = create_video_generator(provider, model=selected_model.value)
                elif provider == Provider.LOCAL:
                    kwargs = {
                        "model_id": model_id,
                        "num_inference_steps": num_inference_steps,
                        "num_frames": num_frames
                    }
                    video_generator = create_video_generator(provider, **kwargs)
                elif provider == Provider.LUMA:
                    kwargs = {
                        "model": selected_model.value,
                        "resolution": selected_resolution.value,
                        "duration": selected_duration.value,
                        "loop": loop_video,
                        "aspect_ratio": aspect_ratio
                    }
                    video_generator = create_video_generator(provider, **kwargs)
                else:
                    st.error("Invalid provider selected")
                    return
                
                video_prompt = VideoPrompt(content=prompt_text)
                
                # For Luma, pass generation options as kwargs
                if provider == Provider.LUMA:
                    generation_kwargs = {
                        "resolution": selected_resolution.value,
                        "duration": selected_duration.value,
                        "loop": loop_video,
                        "aspect_ratio": aspect_ratio
                    }
                    results = video_generator.generate_video(video_prompt, **generation_kwargs)
                else:
                    results = video_generator.generate_video(video_prompt)

                if results:
                    for i, result in enumerate(results, 1):
                        st.video(result.video)
                        with st.expander(f"Details for Video {i}"):
                            st.json(result.metadata or {"message": "No metadata available"})
                else:
                    st.warning("The model did not return any videos.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)


if __name__ == "__main__":
    main()