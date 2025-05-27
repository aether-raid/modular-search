from abc import ABC, abstractmethod
from typing import List, Generic, TypeVar


O = TypeVar('O')

class SearchEngine(ABC, Generic[O]):
    """
    Abstract base class for a search engine.
    """

    @abstractmethod
    def search(self, query: str) -> List[O]:
        """
        Perform a search and return a list of result dictionaries.
        Each dictionary should contain at least: 'title', 'url', 'snippet' (optional).
        """
        pass
    
    def __call__(self, query: str) -> List[O]:
        """
        Allow the search engine to be called like a function.
        """
        return self.search(query)
