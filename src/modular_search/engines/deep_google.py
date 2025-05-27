from typing import List
import re
import requests
from bs4 import BeautifulSoup

from modular_search.engines.google import GoogleSearchEngine


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
        self.depth = depth
    
    def extended_search(self, url: str) -> List[str]:
        """
        Perform an extended search on a given URL to extract more links.
        This method should be implemented to scrape the content of the URL and extract links.
        """  
        
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        
        # ensure the response has content
        if not response.text:
            return []
        
        content = str(response.text)
        
        # Determine the content type and parse accordingly
        # Use html.parser for HTML content
        # Use lxml for XML content
        content_type = response.headers.get('Content-Type', '').lower()
        parser = 'lxml-xml' if 'xml' in content_type else 'html.parser'
        
        # extract BeautifulSoup content based on the content type
        soup = BeautifulSoup(content, parser)
        
        # Remove tags using BeautifulSoup
        text = soup.get_text()
        
        # Basic text cleaning
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
        text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines
        text = text.strip()
        
        # If the text is empty, return an empty list
        if not text:
            return []
        
        # Extracts all the links from the content.
        links = []
        
        links = re.findall(r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)', text)

        return [str(link) for link in links]

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