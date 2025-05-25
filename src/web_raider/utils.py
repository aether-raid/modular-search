# assets/utils.py

import json
import pandas as pd
from .codebase import Codebase
from typing import Any

def json_to_table(json_string: str) -> str:
    """
    Convert a JSON string to a formatted table string.

    Note: Somehow, this function only works when AzureOpenAI gpt-4o is used as the model to output in `call_pro_con`.

    Parameters
    ----------
    json_string : str
        The JSON string to be converted.

    Returns
    -------
    str
        The formatted table as a string.
    """
    # Parse the JSON string
    data = json.loads(json_string)
    
    # Extract codebases
    codebases = data.get("codebases", [])
    
    # Prepare data for DataFrame
    rows = []
    for codebase in codebases:
        name = codebase.get("name", "N/A")
        pros = ", ".join(codebase.get("pros", []))  # Join pros into a single string
        cons = ", ".join(codebase.get("cons", []))  # Join cons into a single string
        
        rows.append({"Name": name, "Pros": pros, "Cons": cons})
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Return the formatted table as a string
    return df.to_string(index=False)

def get_unique_codebases(codebases: list[dict]) -> list[dict]:
    """
    Get unique codebases by filtering out duplicates based on their URLs.

    Parameters
    ----------
    codebases : list[dict]
        A list of dictionaries containing information about the codebases.

    Returns
    -------
    list[dict]
        A list of dictionaries with unique codebases.
    """
    seen_urls = set()
    unique_codebases = []

    for codebase in codebases:
        if codebase['url'] not in seen_urls:
            seen_urls.add(codebase['url'])
            unique_codebases.append(codebase)

    return unique_codebases

def tidy_results(codebases: list[dict]) -> list[dict]:
    """
    Tidy up the results by extracting relevant information from each codebase.

    Parameters
    ----------
    codebases : list[dict]
        A list of dictionaries containing information about the codebases.

    Returns
    -------
    list[dict]
        A list of dictionaries with tidied information about the codebases.
    """
    tidied_results = []
    for codebase in codebases:
        cb = Codebase(codebase['url'])

        tidied_results.append(
            {
                'type': cb.type,
                'name': cb.get_id()[1],
                'url': cb.repository_url
            }
        )

    return tidied_results

def useless_func(*_: Any) -> None:
    """
    A function that does nothing.

    Parameters
    ----------
    _ : Any
        Any number of arguments.

    Returns
    -------
    None
    """
    return
