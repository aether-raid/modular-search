from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List


R = TypeVar('R')
O = TypeVar('O')

class Extractor(ABC, Generic[R, O]):
    @abstractmethod
    def extract(self, candidates: List[R]) -> List[O]:
        pass

    def __call__(self, candidates: List[R]) -> List[O]:
        """
        Allow the extractor to be called like a function.
        """
        return self.extract(candidates)
