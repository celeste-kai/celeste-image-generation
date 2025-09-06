# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the `celeste-image-generation` package, part of the larger Celeste multi-modal AI framework. It provides a unified interface for image generation across multiple providers including Google, OpenAI, Stability AI, HuggingFace, Replicate, Luma, and xAI.

## Development Commands

### Setup and Dependencies
```bash
uv sync                    # Install dependencies
uv add <package>          # Add new dependency (never use pip install)
```

### Running Examples
```bash
streamlit run example.py   # Run the Streamlit demo app
python example.py          # Run example directly (for CLI testing)
```

### Code Quality
```bash
# No specific lint/test commands found - refer to workspace-level commands
# Check parent workspace CLAUDE.md for build/test/lint commands
```

## Architecture

### Core Components
- **Factory Pattern**: `create_image_generator()` creates provider-specific instances
- **Provider Mapping**: `mapping.py` maps Provider enums to implementation classes
- **Base Classes**: Inherits from `BaseImageGenerator` in `celeste-core`
- **Unified Interface**: All providers implement the same `generate_image()` method

### Provider Structure
Each provider in `src/celeste_image_generation/providers/` implements:
- Inherits from `BaseImageGenerator`
- Implements `generate_image()` returning `List[ImageArtifact]`
- Handles provider-specific authentication and API calls
- Normalizes responses to unified format

### Key Files
- `src/celeste_image_generation/__init__.py`: Main factory function
- `src/celeste_image_generation/mapping.py`: Provider-to-class mapping
- `src/celeste_image_generation/providers/`: Individual provider implementations
- `example.py`: Streamlit demo showcasing all providers

### Provider Implementation Pattern
```python
class ProviderImageGenerator(BaseImageGenerator):
    def __init__(self, model: str = "default-model", **kwargs):
        super().__init__(model=model, provider=Provider.NAME, **kwargs)
        # Initialize provider-specific client
    
    async def generate_image(self, prompt: str, **kwargs) -> List[ImageArtifact]:
        # Provider-specific implementation
        # Returns list of ImageArtifact objects
```

## Dependencies and Integration

- **celeste-core**: Provides base classes, enums, validation, and settings
- **Provider SDKs**: Each provider requires specific SDK dependencies
- **Settings Validation**: Uses `settings.validate_for_provider()` to check API keys
- **Model Catalog**: Leverages centralized model catalog from `celeste-core`

## Important Notes

- Always use `uv add` instead of `pip install`
- Provider classes are dynamically imported via mapping.py
- Authentication handled through `celeste-core` settings system
- Example app demonstrates capability-based model filtering
- Google provider has fallback mechanism (Imagen → Gemini)
- All image data returned as bytes in ImageArtifact.data