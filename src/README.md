# web-raider

## Overview

Web Raider is a powerful web scraping and data extraction tool designed to help you gather information from various websites efficiently. It provides a simple interface to configure and run web scraping tasks, making it easy to collect and process data for your projects.

## Setup Guide

1. Clone this repository from GitHub.
2. Open terminal (after redirecting yourself to the repo) and run the following commands:

    - `pip install poetry` (don't create venv through python. does not go well.)
    - `poetry lock` (creates venv for you)
    - `poetry install`

### Setup for Raider Backend

Run `pip install -e .` from the git root directory. Raider Backend will call Web Raider using `pipeline_main(user_query: str)` from `web_raider/pipeline.py`.

## Usage

1. Configure your scraping tasks by editing the configuration files in the `config` directory.
2. Run the scraper using the command: `poetry run python main.py`
3. The scraped data will be saved in the `output` directory.

## How the Repository Works

- **web-raider/**: Contains the core logic of the application.
  - **article.py**: Handles the extraction of codebase URLs from articles.
  - **codebase.py**: Defines the `Codebase` class and its subclasses for different code hosting platforms.
  - **connection_manager.py**: Manages WebSocket connections and message buffering.
  - **evaluate.py**: Evaluates codebases based on a query.
  - **model_calls.py**: Handles calls to external models for query simplification, relevance, scoring, and ranking.
  - **pipeline.py**: Defines the main pipeline for processing user queries.
  - **search.py**: Handles Google search queries and filters results.
  - **shortlist.py**: Shortlists codebases based on a query.
  - **url_classifier.py**: Classifies URLs into different categories.
  - **utils.py**: Contains utility functions.
  - **constants.py**: Defines constants used across the application.
  - **__init__.py**: Initializes the web-raider package.
- **assets/**: Contains auxiliary files and configurations.
  - **key_import.py**: Handles the import of API keys.
  - **prompts.py**: Defines various prompts used in model calls.
  - **__init__.py**: Initializes the assets package.
- **tests/**: Contains unit tests for the application. Run the tests using `pytest` to ensure everything is working correctly.

## Tasklist to complete before Wallaby

1. fix relative/absolute import problem. don't rely on `-m`
2. need to be able to run the code from any directory

## Future Implementations/Improvements

- Use Machine Learning Classification Algorithms to classify types of URLs to their type (Codebase, Article, Forum)
- Find a way to handle Forum URLs (right now they are not processed)
- Find a way to scrape code directly from Articles and Forum URLs (right now only links are scraped)
- Properly implement main query breakdown instead of just whacking LLM
