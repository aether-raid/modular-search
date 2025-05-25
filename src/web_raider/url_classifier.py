# src/url_classifier.py

from urllib.parse import urlparse

def url_classifier(url):
    # Refined List of Codebase Domains
    codebase_domains = [
        'github.com',      # Supports README scraping and API access
        'gitlab.com',      # Supports README scraping and API access
        'bitbucket.org',   # Supports README scraping and API access
        'sourceforge.net', # Supports README scraping and project descriptions
        'gitee.com',       # Supports README scraping and project descriptions
    ]

    # Refined List of Technical Article Domains
    article_domains = [
        'medium.com',
        'dev.to',                  # Articles often link to GitHub repos
        'freecodecamp.org',        # Tutorials often reference codebases
        'smashingmagazine.com',    # Articles may include code examples and links
        'css-tricks.com',          # Web development articles often reference codebases
        'raywenderlich.com',       # Tutorials often link to GitHub repos
    ]

    # Refined List of Technical Forum Domains
    forum_domains = [
        'stackoverflow.com',       # Questions often reference GitHub repos
        'reddit.com/r/programming',# Discussions often link to codebases
        # 'dev.to',                  # Community posts often link to projects
        'codeproject.com',         # Articles may reference codebases
        'hackernews.com',          # Discussions often link to codebases
    ]

    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()

    if any(x in domain for x in codebase_domains):
        return 'Codebase'
    
    elif any(x in domain for x in article_domains):
        return 'Article'
    
    elif any(x in domain for x in forum_domains):
        return 'Forum'
    
    else:
        return 'Useless'
