from enum import Enum
from urllib.parse import urlparse
# from concurrent.futures import ThreadPoolExecutor
import requests

from modular_search.blocks.core import UnitSearchBlock

# Refined List of Codebase Domains
CODEBASE_DOMAINS = [
    'github.com',      # Supports README scraping and API access
    'gitlab.com',      # Supports README scraping and API access
    'bitbucket.org',   # Supports README scraping and API access
    'sourceforge.net', # Supports README scraping and project descriptions
    'gitee.com',       # Supports README scraping and project descriptions
]

# Refined List of Technical Article Domains
ARTICLE_DOMAINS = [
    'medium.com',
    'dev.to',                  # Articles often link to GitHub repos
    'freecodecamp.org',        # Tutorials often reference codebases
    'smashingmagazine.com',    # Articles may include code examples and links
    'css-tricks.com',          # Web development articles often reference codebases
    'raywenderlich.com',       # Tutorials often link to GitHub repos
]

# Refined List of Technical Forum Domains
FORUM_DOMAINS = [
    'stackoverflow.com',       # Questions often reference GitHub repos
    'reddit.com/r/programming',# Discussions often link to codebases
    # 'dev.to',                  # Community posts often link to projects
    'codeproject.com',         # Articles may reference codebases
    'hackernews.com',          # Discussions often link to codebases
]

class URLType(str, Enum):
    """
    Enum for URL types.
    This can be used to classify URLs into different categories.
    """
    CODEBASE = 'codebase'
    ARTICLE = 'article'
    FORUM = 'forum'
    USELESS = 'useless'


class CodebaseSearchBlock(UnitSearchBlock[str]):
    """
    A search block for searching codebases.
    This block can be used to search through code repositories or code files.
    """
    
    def check_url_status(self, url: str, timeout=15):
        """Checks if a URL is accessible."""
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            return 200 <= response.status_code < 400
        except:
            try:
                # Some servers block HEAD requests, try GET as fallback
                response = requests.get(url, timeout=timeout, stream=True)
                return 200 <= response.status_code < 400
            except:
                return False
    
    def classify_url(self, url: str) -> URLType:
        codebase_domains = CODEBASE_DOMAINS
        article_domains = ARTICLE_DOMAINS
        forum_domains = FORUM_DOMAINS

        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()

        if any(x in domain for x in codebase_domains):
            return URLType.CODEBASE
        
        elif any(x in domain for x in article_domains):
            return URLType.ARTICLE
        
        elif any(x in domain for x in forum_domains):
            return URLType.FORUM
        
        else:
            return URLType.USELESS

    def search(self, query: str) -> list[str]:
        """
        Perform a search in the codebase using the provided query.
        This method should be implemented to define specific search behavior for codebases.
        
        Args:
            query (str): The search query to use in the codebase.
        
        Returns:
            list: A list of search results from the codebase.
        """
        
        results = self.engine.search(query)
        
        codebases = []
        articles = []
        forums = []
        
        for url in results:
            # TODO: figure out if this should be here or in the search engine
            if not self.check_url_status(url):
                continue
            classification = self.classify_url(url)
            if classification == URLType.CODEBASE:
                codebases.append(url)
            elif classification == URLType.ARTICLE:
                articles.append(url)
            elif classification == URLType.FORUM:
                forums.append(url)
            else:
                continue

        # TODO: process and vectorize content
        # processed_content = {
        #     'articles': [],
        #     'forums': []
        # }
        
        # # Process articles and forums
        # with ThreadPoolExecutor(max_workers=10) as executor:
        #     futures = []
        #     for url in articles:
        #         futures.append(executor.submit(fetch_and_process_content, url, content_type, processed_content))
            
        #     for future in futures:
        #         future.result()
        
        
        # TODO: analyze chunks
        
        
        # Placeholder implementation; should be overridden by subclasses
        return results