from enum import Enum
from typing import List
from collections import Counter
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

import requests
from pydantic import BaseModel

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modular_search.engines import SearchEngine
from modular_search.blocks.core import UnitSearchBlock
from modular_search.scraper import BS4Scraper


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

class CodebaseSearchResult(BaseModel):
    url: str
    occurrences: int


class CodebaseSearchBlock(UnitSearchBlock[CodebaseSearchResult]):
    """
    A search block for searching codebases.
    This block can be used to search through code repositories or code files.
    """

    def __init__(self, engine: SearchEngine):
        """
        Initializes the CodebaseSearchBlock with a search engine.

        Args:
            engine: An instance of a search engine that supports codebase searching.
        """
        self.engine = engine
        self.scraper = BS4Scraper()

    def check_url_status(self, url: str, timeout: int = 15):
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

        if not self.check_url_status(url):
            return URLType.USELESS

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

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 150) -> List[str]:
        """Splits text into overlapping chunks."""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]

            if len(chunk) < chunk_size * 0.5 and chunks:
                chunks[-1] = chunks[-1] + chunk
                break

            chunks.append(chunk)
            start = end - overlap

        return chunks

    def fetch_and_process_content(self, url: str,
                                  processed_content: List[dict]):
        """Fetches and processes content from a URL."""
        print(f"Fetching content from: {url}")
        content = self.scraper.extract_content(url)

        chunks = self.chunk_text(content)

        processed_content.append({
            'url': url,
            'chunks': chunks,
            'original_length': len(content)
        })
        print(f"Successfully processed {len(chunks)} chunks")

    def fetch_and_process_content_multiple(self, urls: List[str]):
        processed = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for url in urls:
                futures.append(executor.submit(self.fetch_and_process_content, url, processed))

            for future in futures:
                future.result()

        # Collect all chunks with their metadata
        all_chunks = []
        metadata = []

        for doc in processed:
            for chunk_idx, chunk in enumerate(doc['chunks']):
                all_chunks.append(chunk)
                metadata.append({
                    'url': doc['url'],
                    'chunk_index': chunk_idx
                })

        return all_chunks, metadata

    def vectorize_and_find_similarities(self, query: str, all_chunks: List[str], metadata: List[dict]):
        """
        Vectorizes the query and all chunks, then analyzes their similarity.

        Args:
            query (str): The search query to vectorize.
            all_chunks (List[str]): The list of text chunks to vectorize.

        Returns:
            List[dict]: A list of dictionaries containing the similarity scores and chunk texts.
        """
        print("\nVectorizing chunks and evaluating top similar chunks...")

        vectorizer = TfidfVectorizer(max_features=5000)

        # Vectorize all chunks & the query
        vectors = vectorizer.fit_transform(all_chunks)
        question_vector = vectorizer.transform([query])

        # Calculate cosine similarity between question and all chunks
        similarities = cosine_similarity(question_vector, vectors)[0]

        return similarities

    def get_top_links(self, similarities, all_chunks, metadata):
        # Get indices of top k most similar chunks
        top_indices = np.argsort(similarities)[-10:][::-1]

        results = []
        codebase_links = []

        for idx in top_indices:
            # Get the original chunk text
            chunk_text = all_chunks[idx]

            # Extract potential codebase links from chunk
            chunk_links = self.scraper.extract_links(chunk_text)

            if not len(chunk_links):
                continue

            codebase_links_local = set()
            for link in chunk_links:
                # Classify each link to find codebases
                link_type = self.classify_url(link)
                if link_type == URLType.CODEBASE:
                    codebase_links_local.add(link)

            codebase_links.extend(codebase_links_local)

            results.append({
                'url': metadata[idx]["url"],
                'similarity_score': similarities[idx],
                'chunk_text': chunk_text,
                'found_codebase_links': codebase_links_local
            })

        return results, codebase_links

    def search(self, query: str) -> List[CodebaseSearchResult]:
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
        articles_and_forums = []

        for url in results:
            classification = self.classify_url(url)
            if classification == URLType.CODEBASE:
                codebases.append(url)
            elif classification == URLType.ARTICLE:
                articles_and_forums.append(url)
            elif classification == URLType.FORUM:
                articles_and_forums.append(url)
            else:
                continue

        all_chunks, metadata = self.fetch_and_process_content_multiple(articles_and_forums)

        if len(all_chunks) == 0:
            print("No chunks found to vectorize.")
            return []

        # Vectorize all chunks & the query
        similarities = self.vectorize_and_find_similarities(query, all_chunks, metadata)
        # Get top links based on similarity scores
        results, codebase_links = self.get_top_links(similarities, all_chunks, metadata)

        counter = Counter()
        for link in codebases+codebase_links:
            if not self.check_url_status(link):
                continue
            counter[link] += 1

        results = [CodebaseSearchResult(url=link, occurrences=count) for link, count in counter.most_common()]

        return results
