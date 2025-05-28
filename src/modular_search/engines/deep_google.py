from typing import List
import re

from modular_search.engines.google import GoogleSearchEngine
from modular_search.scraper import BS4Scraper


class DeepGoogleSearchEngine(GoogleSearchEngine):
    """
    Extended Google Search engine that goes deeper into each page to extract more links.
    """
    
    def __init__(self, num_results: int = 10, depth: int = 1):
        """
        Initializes the DeepGoogleSearchEngine.
        
        A Depth of 0 would mean no extended search, while a depth of 1 means to extract links from the first page.
        """
        super().__init__(num_results=num_results)
        self.scraper = BS4Scraper()
        self.depth = depth
    
    def extended_search(self, url: str) -> List[str]:
        """
        Perform an extended search on a given URL to extract more links.
        This method should be implemented to scrape the content of the URL and extract links.
        """
        text = self.scraper.extract_content(url)
        links = self.scraper.extract_links(text)
        return links

    def search(self, query: str) -> List[str]:
        results = super().search(query)
        
        overall_results = set(results)
        
        for i in range(self.depth):
            if not results:
                break
            
            extended_results = set()
            for link in results:
                try:
                    extended_links = self.extended_search(link)
                    extended_results |= set(extended_links)
                except Exception as e:
                    print(f"Error processing link {link}: {str(e)}")
                    continue
            
            results = list(extended_results - overall_results)
            overall_results |= extended_results
        
        return results