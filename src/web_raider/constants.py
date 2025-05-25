# src/constants.py

__all__ = [
    "CODE_BLACKLIST"
]

# combining this with site:xxxxxxx, once we figure out how to extract code snippets efficiently
# this can become a reference point to narrow down searches
CODE_BLACKLIST = {
    "stackoverflow.com",
    "quora.com",
    "tutorialspoint.com",
    "w3schools.com",
    "programiz.com",
    "javatpoint.com",
    "geeksforgeeks.org"
}