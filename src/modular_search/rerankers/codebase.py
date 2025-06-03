from typing import List

from modular_search.llm import LLM
from modular_search.scraper import BS4Scraper
from modular_search.blocks.codebase import CodebaseSearchResult
from modular_search.rerankers.core import Reranker


class CodebaseSearchRerankerResult(CodebaseSearchResult):
    accuracy: float = 0


class CodebaseSearchReranker(Reranker[CodebaseSearchResult, CodebaseSearchRerankerResult]):
    """
    Reranker for codebase search results using LLM to evaluate repository content.

    This reranker fetches content from GitHub repositories and uses an LLM to evaluate
    which repository best answers the user's question based on the content extracted.
    """

    def __init__(self, llm: LLM):
        self.llm = llm
        self.scraper = BS4Scraper()

    def rerank(self,
               query: str,
               candidates: List[CodebaseSearchResult]) -> List[CodebaseSearchRerankerResult]:
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

        if not len(candidates):
            return []

        # Get content for top candidates
        candidates_with_content = []
        for candidate in candidates:
            content = self.scraper.get_repo_content(candidate.url)
            if content:
                candidates_with_content.append({
                    'url': candidate.url,
                    'content': content,
                    'occurrences': candidate.occurrences
                })

        if len(candidates_with_content) == 0:
            return []

        results: list[CodebaseSearchRerankerResult] = []
        for candidate in candidates_with_content:
            evaluation_prompt = f"""
            Question: {query}

            Evaluate the following GitHub repository content to determine if it answers the question.
            Rate the candidate repository from 0-100 based on how well it answers the question.
            IMPORTANT: You must ONLY return a numeric score.
            RULES:
                1. score MUST be a number (e.g. 75.50, 32.40, etc.)
                2. DO NOT use text like "The rate is" or "out of 100" only the number and nothing else.

            Candidate Repository Content:
            {candidates_with_content[0]['content']}
            """.strip()

            result = self.llm(evaluation_prompt)
            accuracy = float(result)

            results.append(CodebaseSearchRerankerResult(
                url=candidate['url'],
                occurrences=candidate['occurrences'],
                accuracy=accuracy
            ))

        return sorted(results, key=lambda x: x.accuracy, reverse=True)
