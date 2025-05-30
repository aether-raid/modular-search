from typing import List, Optional
from pydantic import BaseModel

from modular_search.llm import LLM
from modular_search.scraper import BS4Scraper
from modular_search.blocks.codebase import CodebaseSearchResult
from modular_search.rerankers.core import Reranker


class CodebaseSearchRerankerResult(BaseModel):
    best_candidate: Optional[str]
    accuracy: float
    # known_repo_content: str  # Uncomment if you want to return the known repository content


class CodebaseSearchReranker(Reranker[CodebaseSearchResult]):
    """
    Reranker for codebase search results using LLM to evaluate repository content.
    
    This reranker fetches content from GitHub repositories and uses an LLM to evaluate
    which repository best answers the user's question based on the content extracted.
    """
    
    def __init__(self, llm: LLM):
        self.llm = llm
        self.scraper = BS4Scraper()
    
    def rerank(self,
               question: str,
               candidates: List[CodebaseSearchResult],
               known_repos: List[str] = [],
               max_candidates: int = 5) -> CodebaseSearchRerankerResult:
        """
        Reranks candidates based on their content relevance to the question.
        
        Arguments:
        - question: The original question (str)
        - candidates: List of candidate repository links with occurrence counts (list of dict)
        - known_repos: List of known correct repositories (list of str)
        - max_candidates: Maximum number of candidates to analyze (int)
        
        Returns:
        - Dictionary with the best candidate link and its accuracy score (dict)
        """

        if not candidates:
            return CodebaseSearchRerankerResult(best_candidate=None, accuracy=0)

        # Get content for top candidates
        candidates_with_content = []
        for candidate in candidates[:max_candidates]:
            content = self.scraper.get_repo_content(candidate.url)
            if content:
                candidates_with_content.append({
                    'url': candidate.url,
                    'content': content,
                    'occurrences': candidate.url
                })

        if len(candidates_with_content) == 0:
            return CodebaseSearchRerankerResult(best_candidate=None, accuracy=0)

        # Prepare evaluation prompt
        known_repo_content = self.scraper.get_repo_content(known_repos[0]) if known_repos else ""

        # Continue evaluating other candidates if accuracy is low
        for candidate in candidates_with_content:
            evaluation_prompt = f"""
            Question: {question}

            Evaluate the following GitHub repository content to determine if it answers the question.
            Use the known repository content as a reference model answer. Rate the candidate repository from 0-100 based on how well it answers the question. 
            IMPORTANT: You must ONLY return a numeric score.
            RULES:
                1. score MUST be a number (e.g. 75.50, 32.40, etc.)
                2. DO NOT use text like "The rate is" or "out of 100" only the number and nothing else.
                Known Repository Content:
                    {known_repo_content}
    
                Candidate Repository Content:
                    {candidates_with_content[0]['content']}
                """

            result = self.llm(evaluation_prompt)
            accuracy = float(result)

            if accuracy >= 70:
                print('best_candidate: ', candidate['url'], '\naccuracy: ', accuracy )
                return CodebaseSearchRerankerResult(
                    best_candidate=candidate.url,
                    accuracy=accuracy
                )

        return CodebaseSearchRerankerResult(
            best_candidate=candidates_with_content[0]['url'],
            accuracy=0
        )
