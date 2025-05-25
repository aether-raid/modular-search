# src/evaluate.py

import builtins

from .utils import useless_func, json_to_table
from .model_calls import call_pro_con, call_scorer, call_ranker

def codebase_evaluate(query: str, codebases: list[dict], verbose: bool = False) -> list[dict]:
    """
    Evaluate codebases based on a query and return the filtered list of codebases.

    Parameters
    ----------
    query : str
        The query string to evaluate the codebases against.
    codebases : list[dict]
        A list of codebases to be evaluated.
    verbose : bool, optional
        If True, prints detailed information during evaluation (default is False).

    Returns
    -------
    list[dict]
        A list of codebases that match the query criteria.
    """
    print = builtins.print if verbose else useless_func

    # no codebases to evaluate
    if not codebases:
        print(f'Your query "{query}" yielded no relevant codebases.')

    pro_con_response = call_pro_con(codebases, query)
    scorer_response = call_scorer(codebases, query, pro_con_response)

    # log to output file for now cuz i need to see output
    with open('output.txt', 'a') as file:
        file.write(f'{"-"*20}\n\n')
        file.write(f'Query: {query}')
        file.write('\n\n')
        file.write(scorer_response)
        file.write('\n\n')
        file.write(json_to_table(pro_con_response))
        file.write(f'{"-"*20}\n\n')

    filtered_codebases = call_ranker(scorer_response).split('\n')
    
    print(f'These are the top {len(filtered_codebases)} codebases filtered out:\n{filtered_codebases}')

    desired_info = []
    for codebase in codebases:
        if codebase['url'] in filtered_codebases:
            desired_info.append(codebase)

    return desired_info
