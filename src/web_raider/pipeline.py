from .shortlist import codebase_shortlist
from .evaluate import codebase_evaluate
from .model_calls import call_query_simplifier
from .utils import tidy_results, get_unique_codebases
import json

def pipeline(query: str, verbose: bool = False) -> list[dict]:
    """
    Process the query to shortlist and evaluate codebases.

    Parameters
    ----------
    query : str
        The query string to search for codebases.
    verbose : bool, optional
        If True, prints detailed information during the process (default is False).

    Returns
    -------
    list[dict]
        A list of dictionaries containing information about the evaluated codebases.
    """
    codebases = codebase_shortlist(query, verbose)
    desired_info = codebase_evaluate(query, codebases, verbose)

    return desired_info

def pipeline_main(user_query: str) -> list[dict]:
    """
    Main function.

    Parameters
    ----------
    user_query : str
        The query string to search for codebases.

    Returns
    -------
    list[dict]
        A list of dictionaries containing information about the final evaluated codebases.
    """
    potential_codebases = []

    # pipeline(QUERY, True)
    queries = call_query_simplifier(user_query)
    queries = json.loads(queries)

    for query in queries['prompts']:
        potential_codebases.extend(pipeline(query, True))

    final_codebases = codebase_evaluate(user_query, get_unique_codebases(potential_codebases), True)
    final_results = tidy_results(final_codebases)
    data = json.dumps(final_results)

    print(data)
    return data