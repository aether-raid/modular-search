from typing import Dict, List, Union, TypeVar, Generic
from abc import ABC, abstractmethod
from collections import Counter
from modular_search.blocks.core import UnitSearchBlock


O = TypeVar('O')

class SearchController(ABC, Generic[O]):
    """
    Search Controller Class
    =====================

    The search controller provides 3 roles in our framework:

    (1) It serves as the central management component for all
        unit search blocks within the framework. Each unit
        search block operates independently, allowing the search
        controller to orchestrate their concurrent utilization in a
        parallelized manner. In military operations, this capability
        is particularly advantageous, as it accelerates the retrieval
        of critical information during time-sensitive development
        phases.
    (2) It provides military developers with a configurable user
        interface, enabling them to select specific search engines to
        employ based on the query at hand. This flexibility allows
        developers to tailor the search process to meet diverse operational
        requirements, development priorities, and stringent
        security constraints. For example, a developer tasked with
        retrieving documentation on encryption protocols might
        prioritize local search engines for classified materials while
        simultaneously querying web-based sources for publicly
        available algorithms. By offering centralized control, the
        search controller facilitates seamless coordination of the
        search process while ensuring strict adherence to military
        security protocols and operational standards.
    (3) It also provides the capability to configure which unit
        search blocks are queried for a given developer request.
        This ensures that only the most relevant unit search blocks
        are utilized, minimizing the computational overhead and
        avoiding the inclusion of results from blocks that may not
        contribute meaningful outputs. By selectively engaging the
        appropriate unit search blocks, our framework enhances
        efficiency and ensures that the returned results are
        consistently aligned with the developer's specific needs and
        context.

    In other words, the search controller acts as a router to the various search blocks,
    not unlike a router in a MoE model. It allows for the dynamic selection of search blocks
    based on the query and the active blocks specified by the user. This design enables
    more granular control over the search process, allowing developers to
    tailor the search experience to their specific needs and operational requirements.

    This base class is designed to be extended by specific search controllers
    that implement the actual search logic and configuration for different
    use cases. It provides a framework for managing multiple unit search blocks
    and orchestrating their interactions based on user-defined parameters.

    Hence, the primitive implementation allows users to manually specify the
    unit search blocks to be used in the search process.
    """

    def __init__(self, unit_blocks: Union[Dict[str, UnitSearchBlock[O]], List[UnitSearchBlock[O]]]):
        if isinstance(unit_blocks, list):
            block_dict = {}

            name_counter = Counter()

            for block in unit_blocks:
                name_counter[block.__class__.__name__] += 1
                if not isinstance(block, UnitSearchBlock):
                    raise TypeError(f"Expected UnitSearchBlock, got {type(block).__name__}")
                name = block.__class__.__name__ + str(name_counter[block.__class__.__name__])
                block_dict[name] = block

            unit_blocks = block_dict

        self.unit_blocks = unit_blocks

    @abstractmethod
    def select_blocks(self, query: str) -> List[str]:
        """
        Selects the unit search blocks to be used for the given query.
        This method should be implemented by subclasses to define
        how blocks are selected based on the query.
        """
        pass

    def block_search(self, query: str, active_blocks: List[str]) -> Dict[str, List[O]]:
        # Dispatch to active Unit Search Blocks
        all_results = {}
        for block_name in active_blocks:
            block = self.unit_blocks[block_name]
            results = block.search(query)
            all_results[block_name] = results

        return all_results

    @abstractmethod
    def aggregate(self, search_results: Dict[str, List[O]]) -> List[O]:
        pass

    def search(self, query: str) -> List[O]:
        active_blocks = self.select_blocks(query)

        missing_blocks = set(active_blocks) - set(self.unit_blocks.keys())
        if missing_blocks:
            raise ValueError(f"Active blocks {missing_blocks} are not registered in the controller.")

        search_results = self.block_search(query, active_blocks)

        aggregated_results = self.aggregate(search_results)

        return aggregated_results

    def __call__(self, query: str) -> List[O]:
        """
        Allow the controller to be called like a function.
        """
        return self.search(query)
