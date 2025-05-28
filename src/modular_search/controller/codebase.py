from typing import List
from modular_search.controller.core import SearchController
from modular_search.blocks.codebase import CodebaseSearchBlock, CodebaseSearchResult

class CodebaseSearchController(SearchController[CodebaseSearchResult]):
    """
    Codebase Search Controller Class
    """
    
    def __init__(self, search_block: CodebaseSearchBlock):
        super().__init__({
            "CodebaseSearchBlock": search_block
        })
    
    def search(self, query: str) -> List[CodebaseSearchResult]:
        # since it's just one block, it makes our life easier
        all_results = super().internal_search(query, ["CodebaseSearchBlock"])
        return all_results["CodebaseSearchBlock"]