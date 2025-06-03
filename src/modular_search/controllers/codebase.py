from typing import List, Dict
from modular_search.controllers.core import SearchController
from modular_search.blocks.codebase import CodebaseSearchBlock, CodebaseSearchResult


class CodebaseSearchController(SearchController[CodebaseSearchResult]):
    """
    Codebase Search Controller Class
    """

    def __init__(self, search_block: CodebaseSearchBlock):
        super().__init__({
            "CodebaseSearchBlock": search_block
        })

    def select_blocks(self, query: str) -> List[str]:
        """
        Selects the unit search blocks to be used for the given query.
        For CodebaseSearchController, we always use the CodebaseSearchBlock.
        """
        return ["CodebaseSearchBlock"]

    def aggregate(self, search_results: Dict[str, List[CodebaseSearchResult]]) -> List[CodebaseSearchResult]:
        """ Aggregates the search results from the CodebaseSearchBlock.
        Since we only have one block, we can return the results directly.
        """
        return search_results["CodebaseSearchBlock"]
