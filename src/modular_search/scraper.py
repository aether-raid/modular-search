from typing import Optional, Tuple, List
import re
import requests
from bs4 import BeautifulSoup


class BS4Scraper:
    """
    A simple web scraper using BeautifulSoup to fetch and parse HTML content.
    """
    
    def fetch_content(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Fetches the HTML content from the URL."""
        content = None
        parser = None
        
        try:
            response = requests.get(url)
            
            # Check if the request was successful
            response.raise_for_status()  # Raise an error for bad responses
            
            # ensure the response has content
            content = response.text.strip() or ""
            
            # Determine the content type and parse accordingly
            # Use html.parser for HTML content
            # Use lxml for XML content
            content_type = response.headers.get('Content-Type', '').lower()
            parser = 'lxml-xml' if 'xml' in content_type else 'html.parser'
        except requests.RequestException as e:
            print(f"Error fetching content from {url}: {e}")
            
        finally:
            return content, parser
    
    def clean_content(self, content: Optional[str], parser: Optional[str] = None) -> str:
        """Parses the fetched HTML content using BeautifulSoup."""
        if not content:
            print("No content to parse.")
            return ""

        # extract BeautifulSoup content based on the content type
        soup = BeautifulSoup(content, parser or "html.parser")
        
        # Remove tags using BeautifulSoup
        text = soup.get_text()
        
        # Basic text cleaning
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
        text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines
        text = text.strip()
        
        return text

    def extract_content(self, url: str) -> str:
        """
        Fetches and cleans the content from the given URL.
        
        Args:
            url (str): The URL to fetch content from.
        
        Returns:
            str: Cleaned text content from the URL.
        """
        content, parser = self.fetch_content(url)
        cleaned_content = self.clean_content(content, parser)
        
        return cleaned_content
    
    def extract_links(self, text: str) -> List[str]:
        # If the text is empty, return an empty list
        if not text:
            return []
        
        # Extracts all the links from the content.
        links = []
        
        links = re.findall(r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)', text)

        return [str(link) for link in links]
        