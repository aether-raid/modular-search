from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Dict, Any


C = TypeVar('C')

class Reranker(ABC, Generic[C]):
    @abstractmethod
    def rerank(self, query: str, candidates: List[C]) -> List[C]:
        """
        Reranks the given candidates based on the query.
        
        Args:
            query (str): The search query.
            candidates (List[Dict[str, Any]]): The list of candidate documents to rerank.
        
        Returns:
            List[Dict[str, Any]]: The reranked list of candidates.
        """
        pass