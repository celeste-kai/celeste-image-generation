import asyncio

import streamlit as st
from celeste_core import Provider, list_models
from celeste_core.enums.capability import Capability

from celeste_image_generation import create_image_generator


async def main() -> None:
    st.set_page_config(page_title="Celeste Image Generation", page_icon="🎨", layout="wide")
    st.title("🎨 Celeste Image Generation")

    # Get providers that support image generation
    providers = sorted(
        {m.provider for m in list_models(capability=Capability.IMAGE_GENERATION)},
        key=lambda p: p.value,
    )

    with st.sidebar:
        st.header("⚙️ Configuration")
        provider = st.selectbox("Provider:", [p.value for p in providers], format_func=str.title)
        models = list_models(provider=Provider(provider), capability=Capability.IMAGE_GENERATION)
        model_names = [m.display_name or m.id for m in models]
        selected_idx = st.selectbox("Model:", range(len(models)), format_func=lambda i: model_names[i])
        model = models[selected_idx].id

    st.markdown(f"*Powered by {provider.title()}*")
    prompt = st.text_area(
        "Enter your prompt:",
        "A beautiful sunset over mountains",
        height=100,
        placeholder="Describe the image you want to generate...",
    )

    if st.button("🎨 Generate", type="primary", use_container_width=True):
        if not prompt.strip():
            st.error("Please enter a prompt.")
        else:
            generator = create_image_generator(Provider(provider), model=model)

            with st.spinner("Generating..."):
                result = await generator.generate_image(prompt)

                if result and result[0].data:
                    img = result[0]
                    st.success("✅ Image generated successfully!")
                    st.image(
                        img.data,
                        caption="Generated Image",
                        use_container_width=True,
                    )

                    # Show metadata
                    with st.expander("📊 Details"):
                        st.write(f"**Provider:** {provider}")
                        st.write(f"**Model:** {model}")
                        st.write(f"**Prompt:** {prompt}")
                        if img.metadata:
                            st.json(img.metadata)
                else:
                    st.error("Failed to generate image. Please try again.")

    st.markdown("---")
    st.caption("Built with Streamlit • Powered by Celeste")


if __name__ == "__main__":
    asyncio.run(main())
