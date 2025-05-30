from typing import Generic, TypeVar, List


C = TypeVar('C')

class Reranker(Generic[C]):
    def rerank(self, query: str, candidates: List[C]) -> List[C]:
        """
        Reranks the given candidates based on the query.
        
        Args:
            query (str): The search query.
            candidates (List[C]): The list of candidate documents to rerank.
        
        Returns:
            List[C]: The reranked list of candidates.
        """
        return candidates