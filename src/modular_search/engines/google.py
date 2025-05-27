from typing import List, Set
from googlesearch import search as google_search, SearchResult as GSearchResult

from modular_search.engines.core import SearchEngine


class GoogleSearchEngine(SearchEngine[str]):
    """
    Google Search wrapper using googlesearch-python.
    """
    
    def __init__(self, num_results: int = 10):
        """
        Initializes the GoogleSearchEngine.
        """
        super().__init__()
        self.num_results = num_results

    def search(self, query: str) -> List[str]:
        results = google_search(
            query, num_results=self.num_results,
            # advanced=True
        )
        
        search_results: Set[str] = set()
        
        for result in results:
            if isinstance(result, GSearchResult):
                result = result.url
            if result not in search_results:
                search_results.add(result)
        
        return list(search_results)


