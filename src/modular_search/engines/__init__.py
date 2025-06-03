# src/modular_search/engines/__init__.py

from modular_search.engines.core import SearchEngine
from modular_search.engines.google import GoogleSearchEngine
from modular_search.engines.deep_google import DeepGoogleSearchEngine


__all__ = [
    "SearchEngine",
    "GoogleSearchEngine",
    "DeepGoogleSearchEngine",
    "SearchEngineFactory",
]

class SearchEngineFactory:
    """
    Factory class to create instances of SearchEngine.
    """

    @staticmethod
    def load(engine_type: str) -> SearchEngine:
        if engine_type == "google":
            return GoogleSearchEngine()
        elif engine_type == "deep_google":
            return DeepGoogleSearchEngine()
        else:
            raise ValueError(f"Unknown search engine type: {engine_type}")
    
    def __class_getitem__(cls, engine_type: str) -> SearchEngine:
        """
        Factory method to get an instance of a search engine based on the type.
        """
        return cls.load(engine_type)