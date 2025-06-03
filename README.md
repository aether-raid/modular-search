# Modular Search Framework for Military Developers
> [Tew En Hao](https://www.linkedin.com/in/en-hao-tew/), [Cheong Sik Feng](https://scholar.google.com.sg/citations?user=xoQuuC0AAAAJ&hl=en), [Aekas Singh Gulati](https://www.linkedin.com/in/aekas-singh-gulati-6b9360278/), [Dillion Lim](https://www.linkedin.com/in/dillion-lim), [Nicholas Lee Wei Jun](https://www.linkedin.com/in/lwj-nicholas/), Jaye Koh Bo Jay, [Aloysius Han Keng Siew](https://www.linkedin.com/in/aloysius-han-5a456a12/), **[Lim Yong Zhi](https://www.linkedin.com/in/limyz/)**

[![PyPI Latest Release](https://img.shields.io/pypi/v/modular-search.svg?logo=python&logoColor=white&color=blue)](https://pypi.org/project/modular-search/)
![GitHub Page Views Count](https://badges.toozhao.com/badges/01JW9DZB3MAEG11FXQP8EVDRAZ/blue.svg)
<!-- [![GitHub Release Date](https://img.shields.io/github/release-date/aether-raid/modular-search?logo=github&label=latest%20release&color=blue)](https://github.com/aether-raid/modular-search/releases/latest) -->
<!-- ![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/aether-raid/modular-search/docs.yml?label=PyPI%20Publish&color=blue) -->

This repository contains all relevant codes and materials prepared for our paper, _"Modular Search Framework for Military Developers"_, at the [2025 International Conference on Military Communication and Information Systems (ICMCIS)](https://icmcis.eu/).


## 📜 Abstract
Military developers often face unique challenges when searching for information due to the restrictive and specialized environments in which they operate. In recent years, Large Language Models (LLMs) have demonstrated exceptional capabilities in generating coherent, human-like text and answering complex queries across a range of natural language tasks. A modular architecture is ideal, where core LLM capabilities (e.g., code understanding, summarization, and retrieval) operate independently of the specific search engine. We propose a modular, adaptable information retrieval framework tailored for military use, which integrates LLMs as a core component and we developed a prototype based on our proposed framework and conducted a preliminary evaluation using a curated dataset. Our prototype achieved a recall of 95.94%. This modular and adaptable approach underscores the importance of integrating advanced information retrieval techniques in military contexts, paving the way for secure, efficient, and context-aware development processes.

## 🛠️ Installation and Set-Up

### Installing from PyPI

Yes, we have published our framework on PyPI! To install Modular Search and all its dependencies, the easiest method would be to use `pip` to query PyPI. This should, by default, be present in your Python installation. To, install run the following command in a terminal or Command Prompt / Powershell:

```bash
$ pip install modular-search
```

Depending on the OS, you might need to use `pip3` instead. If the command is not found, you can choose to use the following command too:

```bash
$ python -m pip install modular-search
```

Here too, `python` or `pip` might be replaced with `py` or `python3` and `pip3` depending on the OS and installation configuration. If you have any issues with this, it is always helpful to consult 
[Stack Overflow](https://stackoverflow.com/).

### Installing from Source

Git is needed to install this repository from source. This is not completely necessary as you can also install the zip file for this repository and store it on a local drive manually. To install Git, follow [this guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

After you have successfully installed Git, you can run the following command in a terminal / Command Prompt:

```bash
$ git clone https://github.com/aether-raid/modular-search.git
```

This stores a copy in the folder `modular-search`. You can then navigate into it using `cd modular-search`. Then, you can run the following:

```bash
$ pip install .
```

This should install `modular-search` to your local Python instance.



## 💻 Getting Started

### Search Engines

Our framework supports a generic `SearchEngine`, which takes in a query and outputs a list of outputs. We currently support two search engines built in, **Google Search** and **Deep Google Search**.

Google Search uses the [`googlesearch-python`](https://github.com/MarioVilas/googlesearch) package to scrape Google Search results. An example of usage is as follows:

```python
from modular_search.engines import GoogleSearchEngine

engine = GoogleSearchEngine(num_results = 5)

results = engine("How to train a LLM?")
for result in results:
    print(result)

# prints 5 lines of URLs
```

Deep Google Search is a modified version of the above Google Search, that goes to each page returned by the Google Search and extracts links from those pages. This process can be made recursive up to a specific depth. An example of the usage is as follows:

```python
from modular_search.engines import DeepGoogleSearchEngine

engine = DeepGoogleSearchEngine(num_results = 5, depth = 2)

results = engine("How to train a LLM?")
for result in results:
    print(result)

# prints a lot more than 5 lines of URLs
```

One can also develop their own engine with the abstract and generic `SearchEngine` class. For instance:

```python
from typing import List
from pydantic import BaseModel
from modular_search.engines import SearchEngine

class MyCustomSearchEngineOutput(BaseModel):
    # ...

class MyCustomSearchEngine(SearchEngine[MyCustomSearchEngineOutput]):
    def search(self, query: str) -> List[MyCustomSearchEngineOutput]:
        list_of_results = []

        # insert logic

        return list_of_results
```

### Unit Search Blocks

Each _unit search block_ is designed with modularity and search engine independence as core principles, enabling developers to easily customize the suite of search engines to align with their familiarity and missionspecific informational needs.

Within each _unit search block_, use case-specific submodules further process the results retrieved by the search engines. These submodules are abstracted within the framework and can be tailored to meet the needs of specific use cases. They also incorporate modular Large Language Model (LLM) components, designed to refine the initial search results.

The modular architecture of the _unit search block_ facilitates seamless adaptation to a wide range of search requirements from general queries to highly specialized ones, while reducing the need for significant modifications to the core framework.

We support a generic `UnitSearchBlock` for defining basic search methods. To define a custom Unit Search Block, users need to define the abstract `search` function. Here is an example:

```python
from pydantic import BaseModel
from modular_search.engines import GoogleSearchEngine
from modular_search.blocks import UnitSearchBlock

class MyCustomSearchResult(BaseModel):
    # ...

class MyCustomSearchBlock(UnitSearchBlock[MyCustomSearchResult]):
    def __init__(self):
        self.engine = GoogleSearchEngine(num_results = 5)

    def search(self, query: str) -> List[MyCustomSearchResult]:
        results = []

        search_results = self.engine.search()

        # logic

        return results
```

We also implement a `CodebaseSearchBlock` based on the proposed implementation in the paper. Here is a sample usage of this class:

```python
from modular_search.engines import GoogleSearchEngine
from modular_search.blocks import CodebaseSearchBlock

engine = GoogleSearchEngine()
block = CodebaseSearchBlock(engine)

results = block("How to train a LLM?")
for result in results:
    print(result.url, result.occurrences)
```

### Search Controllers

The _search controller_ provides 3 roles in our framework:

1. It serves as the central management component for all _unit search blocks_ within the framework. Each _unit search block_ operates independently, allowing the _search controller_ to orchestrate their concurrent utilization in a parallelized manner. In military operations, this capability is particularly advantageous, as it accelerates the retrieval of critical information during time-sensitive development phases.
2. It provides military developers with a configurable user interface, enabling them to select specific search engines to employ based on the query at hand. This flexibility allows developers to tailor the search process to meet diverse operational requirements, development priorities, and stringent security constraints. For example, a developer tasked with retrieving documentation on encryption protocols might prioritize local search engines for classified materials while simultaneously querying web-based sources for publicly available algorithms. By offering centralized control, the _search controller_ facilitates seamless coordination of the search process while ensuring strict adherence to military security protocols and operational standards.
3. It also provides the capability to configure which _unit search blocks_ are queried for a given developer request. This ensures that only the most relevant _unit search blocks_ are utilized, minimizing the computational overhead and avoiding the inclusion of results from blocks that may not contribute meaningful outputs. By selectively engaging the appropriate _unit search blocks_, our framework enhances efficiency and ensures that the returned results are consistently aligned with the developer’s specific needs and context.

In other words, the _search controller_ acts as a router to the various _search blocks_, not unlike a router in a MoE model. It allows for the dynamic selection of search blocks based on the query and the active blocks specified by the user. This design enables more granular control over the search process, allowing developers to tailor the search experience to their specific needs and operational requirements.

We support a generic `SearchController` that is able to select blocks to activate, select from activated blocks and aggregate. To define a custom Search Controller, users need to provide a dictionary of unit blocks, and define the abstract `select_blocks` and `aggregate` functions. Here is an example:

```python
from typing import List, Dict
from pydantic import BaseModel
from modular_search.controllers import SearchController
# from ... import XXXSearchBlock

class MyCustomSearchResult(BaseModel):
    # ...

class MyCustomSearchController(SearchController[MyCustomSearchResult]):
    def __init__(self, blocks: Dict[str, XXXSearchBlock]):
        super().__init__(blocks)

    def select_blocks(self, query: str) -> List[str]:
        active_blocks = []

        # insert logic

        return active_blocks

    def aggregate(self, search_results: Dict[str, List[MyCustomSearchResult]]) -> List[MyCustomSearchResult]:
        results = []

        # insert logic

        return results
```

We also implement a `CodebaseSearchController` based on the proposed implementation in the paper. Here is a sample usage of this class:

```python
from modular_search.engines import GoogleSearchEngine
from modular_search.blocks import CodebaseSearchBlock
from modular_search.controllers import CodebaseSearchController

engine = GoogleSearchEngine()
block = CodebaseSearchBlock(engine)
controller = CodebaseSearchController(block)

results = controller("How to train a LLM?")
for result in results:
    print(result.url, result.occurrences)
```

Notably, the `CodebaseSearchController` only has one block.

## Rerankers & Extractors

In information retrieval, results re-ranking is a critical post-processing step aimed at improving the relevance and accuracy of search results. By reorganizing the retrieved results, re-ranking ensures that the most pertinent information is prioritized, enabling developers to access the most relevant insights quickly and efficiently. This process is particularly valuable in contexts where the quality and order of information significantly impact decision-making, such as military operations.

Within the framework, re-ranking leverages additional contextual and evaluative data collected by the submodules within each _unit search block_. These submodules generate rich metadata such as content relevance, security classifications, and domain-specific metrics that are integral to refining the order and priority of search results.

The implementation of the re-ranking system is intentionally flexible, enabling developers to adopt methodologies aligned with their operational requirements. Potential implementations range from traditional rule-based approaches and heuristic algorithms to advanced neural networks or the integration of LLMs.

After re-ranking, the top $k$ results are filtered and returned to the developer with additional information extracted for the final analysis, where $k$ is a configurable parameter determined by the developer. This parameter allows developers to adjust the breadth of their results pool to balance comprehensiveness with operational efficiency. In scenarios requiring rapid decision-making, a narrower $k$ may be chosen to focus on highly relevant results, while broader values can support exploratory tasks where diverse information is critical.

Our framework supports generic `Reranker` and `Extractor` models that attempt to rerank, filter and extract relevant information. To implement a custom Reranker, users need to define the abstract `rerank` function. An example is shown below:

```python
from typing import List
from pydantic import BaseModel
from modular_search.rerankers import Reranker

class MyCustomSearchResult(BaseModel):
    # ...

class MyCustomSearchRerankerResult(BaseModel):
    # ...

class MyCustomSearchReranker(Reranker[MyCustomSearchResult, MyCustomSearchRerankerResult]):
    def rerank(self, query: str, candidates: List[MyCustomSearchResult]) -> List[MyCustomSearchRerankerResult]:
        results = []

        # logic

        return results
```

To implement a custom Extractor, users need to define the abstract `extract` function. An example is shown below:

```python
from typing import List
from pydantic import BaseModel
from modular_search.extractors import Extractor

class MyCustomSearchRerankerResult(BaseModel):
    # ...

class MyCustomSearchExtractorResult(BaseModel):
    # ...

class MyCustomSearchExtractor(Extractor[MyCustomSearchRerankerResult, MyCustomSearchExtractorResult]):
    def extract(self, candidates: List[MyCustomSearchRerankerResult]) -> List[MyCustomSearchExtractorResult]:
        results = []

        # logic

        return results
```

We also implement a `CodebaseSearchReranker` and `CodebaseSearchExtractor` based on the proposed implementation in the paper. Here is a sample usage of these classes:

```python
from modular_search.blocks import CodebaseSearchResult
from modular_search.rerankers import CodebaseSearchReranker
from modular_search.extractors import CodebaseSearchExtractor

def llm(query: str) -> str:
    # insert logic
    return ""

query = "How to train a LLM?"

results = [
    CodebaseSearchResult(url = "...", occurrences = 4),
    CodebaseSearchResult(url = "...", occurrences = 3),
    CodebaseSearchResult(url = "...", occurrences = 1),
]

reranker = CodebaseSearchReranker(llm)
reranked_results = reranker(query, results)

for result in reranked_results:
    print(result.url, result.occurrences, result.accuracy)

extractor = CodebaseSearchExtractor()
extracted_results = extractor(reranked_results)

for result in extracted_results:
    print(result.url, result.occurrences, result.accuracy, result.code_blocks)
```

## Putting it all Together

We define our own flow for Codebase Search, which you can find below:

```python
from modular_search.engines import GoogleSearchEngine
from modular_search.blocks import CodebaseSearchBlock
from modular_search.controllers import CodebaseSearchController
from modular_search.rerankers import CodebaseSearchReranker
from modular_search.extractors import CodebaseSearchExtractor

def llm(query: str) -> str:
    # insert logic
    return ""

query = "How to train a LLM?"

engine = GoogleSearchEngine()
block = CodebaseSearchBlock(engine)
controller = CodebaseSearchController(block)

results = controller(query)

for result in results:
    print(result.url, result.occurrences)

reranker = CodebaseSearchReranker(llm)
reranked_results = reranker(query, results)

extractor = CodebaseSearchExtractor()
extracted_results = extractor(reranked_results)

for result in extracted_results:
    print(result.url, result.occurrences, result.accuracy, result.code_blocks)
```

This should provide a well-supported list of codebase links.

<!-- ## 🖊️ Citing Modular Search

```bibtex

``` -->

<p align="center">
<a href="https://star-history.com/#aether-raid/modular-search">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=aether-raid/modular-search&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=aether-raid/modular-search&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=aether-raid/modular-search&type=Date" />
  </picture>
</a>
</p>