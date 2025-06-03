from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List


C = TypeVar('C')
O = TypeVar('O')

class Reranker(ABC, Generic[C, O]):
    @abstractmethod
    def rerank(self, query: str, candidates: List[C]) -> List[O]:
        """
        Reranks the given candidates based on the query.

        Args:
            query (str): The search query.
            candidates (List[C]): The list of candidate documents to rerank.

        Returns:
            List[C]: The reranked list of candidates.
        """
        pass

    def __call__(self, query: str, candidates: List[C]) -> List[O]:
        """
        Allow the reranker to be called like a function.
        """
        return self.rerank(query, candidates)
