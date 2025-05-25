# src/article.py

from newspaper import Article
from typing import Generator
from .codebase import Codebase
from .url_classifier import url_classifier

class CodeArticle(Article):
    def __init__(self, url: str, download: bool = True, parse: bool = True):
        super().__init__(url, keep_article_html=True)        
        if download:
            self.download()
        if parse:
            self.parse()

    # include search_depth to allow for digging of other articles by "DFS"
    # to a certain extent
    def code_urls(self, search_depth=1) -> Generator[Codebase, None, None]:
        results = set()
        if self.clean_top_node is not None:
            for elem, key, url, _ in self.clean_top_node.iterlinks():
                if elem.tag != "a":
                    continue
                assert key == "href"
                
                if Codebase.is_code(url):
                    codebase = Codebase(url)
                    if codebase.repository_url not in results:
                        yield codebase
                        results.add(codebase.repository_url)

                # can change search_depth limit depending on how deep we want to go
                
                # update as of 190924 search_depth = 3 just digs extremely deep but results
                # are not good either
                elif url_classifier(url) and search_depth < 2:
                    try:
                        article = CodeArticle(url)
                        for codebase in article.code_urls(search_depth=search_depth+1):
                            if codebase.repository_url not in results:
                                yield codebase
                                results.add(codebase.repository_url)
                    except:
                        pass