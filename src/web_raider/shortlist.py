# src/shortlist.py

import builtins

from .search import GoogleSearch
from .article import CodeArticle
from .codebase import Codebase, CodebaseType
from .url_classifier import url_classifier
from .model_calls import call_relevance
from .utils import useless_func

def codebase_shortlist(query: str, verbose: bool = False) -> list[dict]:
    """
    Shortlist codebases based on a query by searching Google and evaluating the results.

    Parameters
    ----------
    query : str
        The query string to search for codebases.
    verbose : bool, optional
        If True, prints detailed information during the process (default is False).

    Returns
    -------
    list[dict]
        A list of dictionaries containing information about the shortlisted codebases.
    """
    #print = builtins.print if verbose else useless_func

    print(f'Searching Google with: "{query}"')

    googlesearch = GoogleSearch(query=query, num_results=25)
    googlesearch.run_query()
    results = googlesearch.get_relevant_urls()

    print(f'Found {len(results)} relevant results!')

    codebases = []
    seen_urls = set()

    for url in results:
        print("analysing", url)
        if url in seen_urls:
            continue

        seen_urls.add(url)

        if url_classifier(url) == 'Codebase':
            print(f'Checking codebase from "{url}"')

            try:
                object = Codebase(url)
            
            except:
                print(f'The codebase pointed to by {url} is unsupported.')
                continue

            if object.is_code(object.repository_url):
                codebase_type = CodebaseType.format_type(object.type)

                # first check if the link does lead to a repository
                if object.check_is_repo():
                    print(f"Found {codebase_type} Repository {object.repository_url}")

                    codebase = {
                        'url': object.repository_url,
                        'info': object.combine_info(),
                    }

                    # relevance check for each codebase separately to circumvent token limit
                    if call_relevance([codebase], query) == 'True':
                        codebases.append(codebase)
        
        elif url_classifier(url) == 'Article':
            print(f'Opening article from "{url}"')

            article = CodeArticle(url)

            for object in article.code_urls():
                # if url already is checked ignore it
                if object.original_url in seen_urls:
                    continue

                seen_urls.add(object.original_url)
                
                codebase_type = CodebaseType.format_type(object.type)

                # first check if the link does lead to a repository
                if object.check_is_repo():
                    print(f"Found {codebase_type} Repository {object.repository_url}")

                    codebase = {
                        'url': object.repository_url,
                        'info': object.combine_info(),
                    }

                    # relevance check for each codebase separately to circumvent token limit
                    if call_relevance([codebase], query) == 'True':
                        codebases.append(codebase)

        elif url_classifier(url) == 'Forum':
            pass
            # object = CodeForum(url)

        else:
            continue
        
    print(f'Found {len(codebases)} codebases!')

    return codebases
