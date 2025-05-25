# tests/test_search.py

import unittest

from src.search import GoogleSearch


PROMPT = "How to implement a React Codebase?"
MAX_CAP = 20

class TestGoogleSearch(unittest.TestCase):
    def setUp(self):
        self.prompt = PROMPT
        self.cap = MAX_CAP
    
    def test_search_init(self):
        GoogleSearch(self.prompt)
    
    def test_search_number(self):
        result = GoogleSearch(self.prompt)
        
        num = 0
        for _ in result.search_results:
            num += 1
        
        assert num == self.cap
    
    def test_search_general(self):
        result = GoogleSearch(self.prompt)
        result_copy = result.copy()
        
        results = set()
        for search_result in result.search_results:
            results.add(search_result.url)
            
        
        num = 0
        for url in result_copy.urls():
            assert url in results
            num += 1
        
        assert num == len(results)
    
    def test_search_relevant(self):
        result = GoogleSearch(self.prompt)
        
        blacklist = result.blacklist
        
        num = 0
        
        for url in result.relevant_urls():
            for item in blacklist:
                assert item not in url
            num += 1
        
        assert num <= self.cap
    
    def test_search_iterator_loopback(self):
        results = []
        
        result = GoogleSearch(self.prompt)
        
        for url in result.relevant_urls():
            results.append(url)

        idx = 0

        for url in result.relevant_urls():
            assert idx < len(results)
            assert url == results[idx]
            idx += 1
        
        assert idx == len(results)
    
    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()

