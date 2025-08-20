import streamlit as st
from celeste_core import Provider, list_models
from celeste_core.enums.capability import Capability
from celeste_image_generation import create_image_generator
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Celeste Image Generation", page_icon="🎨", layout="wide")
st.title("🎨 Celeste Image Generation")

# Get providers that support image generation
providers = sorted(
    {m.provider for m in list_models(capability=Capability.IMAGE_GENERATION)},
    key=lambda p: p.value,
)

with st.sidebar:
    st.header("⚙️ Configuration")
    provider = st.selectbox(
        "Provider:", [p.value for p in providers], format_func=str.title
    )
    models = list_models(
        provider=Provider(provider), capability=Capability.IMAGE_GENERATION
    )
    model_names = [m.display_name or m.id for m in models]
    selected_idx = st.selectbox(
        "Model:", range(len(models)), format_func=lambda i: model_names[i]
    )
    model = models[selected_idx].id

    # Optional parameters
    st.subheader("Options")
    num_images = st.slider("Number of images", 1, 4, 1)
    if provider == "luma":
        aspect_ratio = st.selectbox(
            "Aspect Ratio", ["16:9", "1:1", "3:4", "4:3", "9:16", "9:21", "21:9"]
        )

st.markdown(f"*Powered by {provider.title()}*")
prompt = st.text_area(
    "Enter your prompt:",
    "A beautiful sunset over mountains",
    height=100,
    placeholder="Describe the image you want to generate...",
)

if st.button("🎨 Generate", type="primary", use_container_width=True):
    generator = create_image_generator(Provider(provider), model=model)

    async def generate_streaming() -> None:
        with st.spinner("Generating..."):
            # Prepare columns and placeholders
            cols = st.columns(min(num_images, 3))
            slots = [col.empty() for col in cols]

            # Per-image kwargs
            base_kwargs: dict = {}
            if provider == "luma" and "aspect_ratio" in locals():
                base_kwargs["aspect_ratio"] = aspect_ratio

            # Concurrency: avoid parallelism for local diffusers to reduce OOM risk
            max_concurrency = 1 if provider == "local" else min(num_images, 4)

            sem = __import__("asyncio").Semaphore(max_concurrency)

            async def generate_one(idx: int) -> tuple:
                async with sem:
                    res = await generator.generate_image(prompt, **base_kwargs)
                    return idx, res[0] if res else None

            tasks = [generate_one(i) for i in range(num_images)]

            shown = 0
            for fut in __import__("asyncio").as_completed(tasks):
                i, img = await fut
                target = slots[shown % len(slots)]
                with cols[shown % len(cols)]:
                    if img and img.data:
                        target.image(
                            img.data,
                            caption=f"Image {shown + 1}",
                            use_container_width=True,
                        )
                        with st.expander("Metadata"):
                            st.json(img.metadata)
                    else:
                        # Handle case where image generation failed
                        target.error(f"Failed to generate image {shown + 1}")
                shown += 1

    __import__("asyncio").run(generate_streaming())

st.markdown("---")
st.caption("Built with Streamlit • Powered by Celeste")
