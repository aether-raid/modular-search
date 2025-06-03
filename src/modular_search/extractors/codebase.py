from typing import List
from modular_search.scraper import BS4Scraper
from modular_search.extractors.core import Extractor
from modular_search.blocks.codebase import CodebaseSearchResult
from modular_search.rerankers.codebase import CodebaseSearchRerankerResult


class CodebaseSearchExtractorResult(CodebaseSearchRerankerResult):
    code_blocks: List[str]

class CodebaseSearchExtractor(Extractor[CodebaseSearchResult, CodebaseSearchExtractorResult]):
    def __init__(self):
        self.scraper = BS4Scraper()

    def extract(self, candidates: List[CodebaseSearchResult]) -> List[CodebaseSearchExtractorResult]:
        results = []
        for candidate in candidates:
            snippets = self.scraper.extract_code_from_repo(candidate.url)
            results.append(CodebaseSearchExtractorResult(
                url = candidate.url,
                occurrences = candidate.occurrences,
                accuracy = candidate.accuracy if isinstance(candidate, CodebaseSearchRerankerResult) else 0,
                code_blocks = snippets
            ))
        return results
