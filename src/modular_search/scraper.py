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

    def get_repo_content(self, url: str, max_files: int = 5) -> str:
        """Extracts relevant content from a repository."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        content_summary = []
        
        # Get repository description
        description = soup.find('p', {'class': 'f4 my-3'})
        if description:
            content_summary.append(f"Description: {description.get_text().strip()}")

        # Get README content
        readme = soup.find('article', {'class': 'markdown-body'})
        if readme:
            content_summary.append(f"README: {readme.get_text()[:1000]}")  # Limit README length

        # Get code files
        code_elements = soup.find_all(['pre', 'div'], class_=['highlight', 'blob-code'])
        files_added = 0
        for elem in code_elements:
            if files_added >= max_files:
                break
            code = elem.get_text().strip()
            if len(code) > 50:  # Skip very small snippets
                content_summary.append(f"Code Sample {files_added + 1}:\n{code[:500]}")
                files_added += 1

        return "\n\n".join(content_summary)

    def extract_code_from_repo(self, url: str) -> List[str]:
        """Extracts code from a repository URL."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        code_blocks = []
        
        # Find code elements
        if 'github.com' in url.lower():
            # GitHub specific extraction
            code_elements = soup.find_all(['pre', 'div'], class_=['highlight', 'highlight-source', 'blob-code', 'js-file-line'])
        else:
            # Generic code extraction
            code_elements = soup.find_all(['pre', 'code', 'div'], class_=['code', 'snippet', 'source'])
            
        for element in code_elements:
            code = element.get_text().strip()
            if len(code) > 50:  # Filter out small snippets
                code_blocks.append(code)
        
        return code_blocks

