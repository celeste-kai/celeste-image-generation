import asyncio
from dotenv import load_dotenv

from celeste_video_generation import create_video_generator
from celeste_video_generation.core.enums import Provider
from celeste_video_generation.core.types import VideoPrompt

load_dotenv()


async def main() -> None:
    """
    Simple test for generating videos using a local provider.
    """
    print("Testing local video generation...")
    print("This will download the model on the first run.")
    print("Please be patient, this process can take several minutes.")

    provider = Provider.LOCAL
    model = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    try:
        video_generator = create_video_generator(provider, model=model)
        video_prompt = VideoPrompt(content="A cinematic shot of a panda drinking a milkshake.")
        
        results = await video_generator.generate_video(video_prompt)

        if results:
            for i, result in enumerate(results, 1):
                with open(f"generated_video_local_{i}.mp4", "wb") as f:
                    f.write(result.video)
                print(f"Generated video {i} and saved as generated_video_local_{i}.mp4")
                print(f"Metadata: {result.metadata}")
        else:
            print("The model did not return any videos.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())