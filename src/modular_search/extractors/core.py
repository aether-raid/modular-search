from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List


R = TypeVar('R')
O = TypeVar('O')

class Extractor(ABC, Generic[R, O]):
    @abstractmethod
    def extract(self, candidates: List[R]) -> List[O]:
        pass
        