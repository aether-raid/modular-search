from web_raider.shortlist import codebase_shortlist
from bs4 import BeautifulSoup
import requests
import re
from googlesearch import search
from web_raider.article import CodeArticle
from web_raider.codebase import Codebase, CodebaseType
from web_raider.url_classifier import url_classifier
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import json
import html
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import statistics
import pickle
import time
from typing import List, Dict, Optional
import json
from datetime import datetime
from pydantic import BaseModel
import numpy as np

# query-controller
# - llm_rephrase(prompt): Rephrases a question using the OpenAI API.
# - llm_prompt(prompt): Sends a prompt to the OpenAI API and returns the response.
# - process_questions(path, limit=5): Processes questions from a JSONL file and classifies links.

# search-controller
# - check_url_status(url, timeout=15): Checks if a URL is accessible.
# - filter_dead_links(urls): Filters out dead links using parallel requests.
# - classifier(results): Classifies URLs into codebases, articles, and forums.
# - clean_text(text): Cleans and normalizes text content.

# aggregator
# - chunk_text(text, chunk_size=300, overlap=60): Splits text into overlapping chunks.
# - process_and_vectorize_content(classified_links): Processes and vectorizes content from articles and forums.
# - extract_from_top_candidates(ranked_candidates, k=3): Extracts code from the top k ranked repositories.

# reranker
# - analyze_similarity_and_extract_links(question, processed_content, top_k=25): Analyzes chunk similarity using LSA and extracts codebase links from top chunks.
# - create_candidate_list(classified_links, analysis_results): Creates and sorts a candidate list based on occurrences.
# - rerank_candidates_with_llm(question, candidates, known_repos, max_candidates=5): Re-ranks candidate links using LLM based on repository content.

# web-scraper
# - extract_code_from_repo(url): Extracts code from a repository URL.
# - extract_links(text): Extracts all the links from the HTML content.
# - get_html_content(url): Fetches the HTML content of the web page from a single link.
# - get_repo_content(url, max_files=5): Extracts relevant content from a repository.

# evaluator
# - evaluate_model_accuracy(results, known_repos): Evaluates model accuracy by comparing found repositories with known repositories.

client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key='ollama'
)


if __name__ == "__main__":
    print("Running main function")
    path = "../web-raider/first_200_questions.jsonl"
    skip = 12                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    query_count = 0
    with open(path, "r") as file:
        for line in file:
            if query_count >= 100:
                break
            if query_count < skip:
                query_count += 1
                continue
            # Load question into object
            json_data = json.loads(line)
            question = Question(**json_data)
            print(f"\nProcessing question {question.Id}")
            print(f"Original question: {question.Title}")

            # Clean question body
            cleaned_body = clean_text(question.Body) if question.Body else ""

            # Search web using question title
            #retry = 0
            #while retry < 5:
                #try:
            search_results = list(search(question.Title, stop=10))
            print(f"Found {len(search_results)} search results")
                    #break
                #except:
                    #time.sleep(600)
                    #retry += 1
                
            
            # Extract links from search results
            link_list = []
            for link in search_results:
                try:
                    content = get_html_content(link)
                    content = clean_text(content) if content else ""
                    if content:
                        extracted_links = extract_links(content)
                        link_list.extend(extracted_links)
                except Exception as e:
                    print(f"Error processing link {link}: {str(e)}")
                    continue

            # Classify links
            classified_links = classifier(list(link_list + search_results))
            processed_data = process_and_vectorize_content(classified_links)

            if not processed_data:
                print("No content could be processed")
                continue

            # Analyze chunks
            print("\nEvaluating top similar chunks...")
            analysis = analyze_similarity_and_extract_links(
                question=question.Title,
                processed_content=processed_data,
                top_k=10
            )

            if not analysis:
                print("No analysis results generated")
                continue

            model_answer = ""
            for results in analysis["top_chunks"]:
                model_answer += results["chunk_text"]

            # Generate model answer using question title and body
            model_answer_prompt = f"""
            Question: {question.Title}
            Search Results: {model_answer}

            Please generate a comprehensive answer to the question based on the provided search results. Follow these requirements:

            1. Content Requirements:
            - Start with a clear, direct answer to the main question
            - Support key points with specific references to the search results
            - Ensure all crucial aspects of the question are addressed
            - Maintain logical flow and coherence throughout the response
            - Provide relevant examples or explanations where appropriate

            2. Formatting Requirements:
            - Structure the answer using proper markdown formatting
            - Use headers (##) to organize main sections if the answer is complex
            - Format any code snippets using appropriate markdown code blocks
            - Use italics (*) for emphasis on key terms when relevant
            - Include proper paragraph breaks for readability

            3. Citation Requirements:
            - Reference specific parts of the search results to support claims
            - Use inline citations by mentioning "According to the search results..."
            - Maintain clear connection between assertions and source material

            4. Quality Check:
            - Ensure the answer directly addresses the original question
            - Verify that all information comes from the provided search results
            - Maintain consistent tone and professional language throughout
            - Check that the response flows logically from point to point

            Return ONLY the formatted answer text, with no additional meta-text or formatting instructions.
            """

            model_answer = llm_prompt(model_answer_prompt)
            model_answer_text = model_answer.choices[0].message.content.strip()

            # Score the model answer
            model_answer_score = score_model_answer(question.Title, cleaned_body, model_answer_text)

            # Save results
            save_query_results(
                question_title=question.Title,
                cleaned_body= cleaned_body,
                model_answer=model_answer_text,
                model_answer_score=model_answer_score['score']
            )
                #model_answer_justification=model_answer_score['justification']

            query_count += 1

       

            del()

    print(f"\nFinished processing {query_count} queries")



        