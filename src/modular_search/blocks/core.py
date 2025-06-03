from abc import ABC, abstractmethod
from typing import List, Generic, TypeVar
from modular_search.engines import SearchEngine


O = TypeVar('O')

class UnitSearchBlock(ABC, Generic[O]):
    """
    Abstract base class for all Unit Search Blocks.
    """

    @abstractmethod
    def search(self, query: str) -> List[O]:
        """
        Perform a search using the provided query.
        This method should be implemented by subclasses to define specific search behavior.
        """
        pass

    def __call__(self, query: str) -> List[O]:
        """
        Allow the search block to be called like a function.
        This method will invoke the search method with the provided query.
        """
        return self.search(query)
