# src/search.py

#from googlesearch import search
from .constants import CODE_BLACKLIST


class GoogleSearch:
    """
    A class to handle a Google Search.

    ...

    Attributes
    ----------
    query: str
        the query of the user
    cap: int
        the (maximum) number of search results you want to have
    blacklist: list[str]
        a list of domains that we want to blacklist (ie. not search for)
    search_results: list[SearchResult]
        a list of Google search results based on the query

    Methods
    -------
    run_query: None
        gets search results of the query and saves it as an attribute

    get_relevant_urls: list[str]
        filters through the links of the search results to see which are relevant
        and returns the list of relevant links
    """
    
    def __init__(self, query: str,
                 num_results: int = 10,
                 blacklist: list[str] = CODE_BLACKLIST):
        """
        Constructs all the necessary attributes for the GoogleSearch object.

        Parameters
        ----------
        query: str
            the query of the user
        cap: int
            the (maximum) number of search results you want to have
        blacklist: list[str]
            a list of domains that we want to blacklist (ie. not search for)
        """

        self.query = query
        self.cap = num_results
        self.blacklist = blacklist
    
    def run_query(self) -> None:
        """
        Runs the query through `googlesearch.search()` and saves the
        results as an attribute.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.search_results = list(search(
                self.query, num_results=self.cap,
                advanced=True
            ))

    def get_relevant_urls(self) -> list[str]:
        """
        Filters the search results through a blacklist and only returns
        the url of search results which pass the filter.

        Parameters
        ----------
        None

        Returns
        -------
        results: list[str]
            the list of urls that passed the filter
        """
        results = set()
        
        for result in self.search_results:
            if result.url not in results and all(domain not in result.url for domain in self.blacklist):
                results.add(result.url)

        return list(results)