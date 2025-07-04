"""
Celeste Search: A unified search interface for multiple search providers.
"""

from typing import Any

from .base import BaseSearchEngine
from .core.enums import Provider
from .core.types import SearchResult

__version__ = "0.1.0"

SUPPORTED_PROVIDERS = [
    "google",
    "github-repository",
    "arxiv",
    "serper",
    "serpapi",
    "duckduckgo",
    "bing",
    "brave",
]


def create_search_engine(provider: str, **kwargs: Any) -> "BaseSearchEngine":
    """
    Factory function to create a search engine instance based on the provider.

    Args:
        provider: The search provider to use (string or Provider enum).
        **kwargs: Additional arguments to pass to the search engine constructor.

    Returns:
        An instance of a search engine.
    """
    # Convert Provider enum to string if needed
    if isinstance(provider, Provider):
        provider = provider.value

    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")

    if provider == "brave":
        from .providers.brave import BraveSearch

        return BraveSearch(**kwargs)
    elif provider == "serpapi":
        from .providers.serpapi import SerpAPISearch

        return SerpAPISearch(**kwargs)
    elif provider == "github-repository":
        from .providers.github import GitHubRepositorySearch

        return GitHubRepositorySearch(**kwargs)
    elif provider == "google":
        raise NotImplementedError(f"Provider {provider} not implemented yet")
    elif provider == "arxiv":
        raise NotImplementedError(f"Provider {provider} not implemented yet")
    elif provider == "serper":
        raise NotImplementedError(f"Provider {provider} not implemented yet")
    elif provider == "duckduckgo":
        raise NotImplementedError(f"Provider {provider} not implemented yet")
    elif provider == "bing":
        raise NotImplementedError(f"Provider {provider} not implemented yet")

    raise ValueError(f"Provider {provider} not implemented")


__all__ = [
    "create_search_engine",
    "BaseSearchEngine",
    "Provider",
    "SearchResult",
    "__version__",
]
