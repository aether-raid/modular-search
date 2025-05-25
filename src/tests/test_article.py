# tests/test_article.py

from web_raider.article import CodeArticle
from web_raider.codebase import Codebase

ARTICLE_URL = "https://www.freecodecamp.org/news/tag/chatbots/"

article = CodeArticle(ARTICLE_URL)
for url in article.code_urls():
    print(url)