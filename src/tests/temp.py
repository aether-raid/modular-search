def get_repo_content(url: str, max_files: int = 5) -> str:
    """Extracts relevant content from a repository."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        content_summary = []
        
        # Get repository description
        description = soup.find('p', {'class': 'f4 my-3'})
        if description:
            content_summary.append(f"Description: {description.get_text().strip()}")

        # Get README content
        readme = soup.find('article', {'class': 'markdown-body'})
        if readme:
            content_summary.append(f"README: {readme.get_text()[:1000]}")  # Limit README length

        # Get code files
        code_elements = soup.find_all(['pre', 'div'], class_=['highlight', 'blob-code'])
        files_added = 0
        for elem in code_elements:
            if files_added >= max_files:
                break
            code = elem.get_text().strip()
            if len(code) > 50:  # Skip very small snippets
                content_summary.append(f"Code Sample {files_added + 1}:\n{code[:500]}")
                files_added += 1

        return "\n\n".join(content_summary)
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return ""
    

    
def extract_code_from_repo(url: str) -> dict:
    """Extracts code from a repository URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        code_blocks = []
        
        # Find code elements
        if 'github.com' in url.lower():
            # GitHub specific extraction
            code_elements = soup.find_all(['pre', 'div'], class_=['highlight', 'highlight-source', 'blob-code', 'js-file-line'])
        else:
            # Generic code extraction
            code_elements = soup.find_all(['pre', 'code', 'div'], class_=['code', 'snippet', 'source'])
            
        for element in code_elements:
            code = element.get_text().strip()
            if len(code) > 50:  # Filter out small snippets
                code_blocks.append(code)
        
        return {
            'url': url,
            'code_blocks': code_blocks,
            'success': True
        }
    except Exception as e:
        print(f"Error extracting code from {url}: {e}")
        return {
            'url': url,
            'code_blocks': [],
            'success': False,
            'error': str(e)
        }

def extract_from_top_candidates(ranked_candidates: List[dict], k: int = 3) -> List[dict]:
    """Extracts code from the top k ranked repositories."""
    results = []
    
    for candidate in ranked_candidates[:k]:
        url = candidate['url']
        print(f"\nExtracting code from: {url}")
        
        extraction_result = extract_code_from_repo(url)
        if extraction_result['success'] and extraction_result['code_blocks']:
            results.append({
                'url': url,
                'rank_score': candidate.get('combined_score', candidate.get('occurrences', 0)),
                'llm_reasoning': candidate.get('reasoning', 'N/A'),
                'code_blocks': extraction_result['code_blocks']
            })
    
    return results

def normalize_repo_url(url: str) -> str:
    """Normalize GitHub URLs for comparison"""
    url = url.strip('/')  # Removed .lower()
    # Remove protocol
    url = re.sub(r'https?://', '', url)
    # Remove www
    url = re.sub(r'www\.', '', url)
    # Remove .git extension
    url = re.sub(r'\.git$', '', url)
    return url

  
def update_process_questions(path: str, limit: int = 5):
    """Modified process_questions function to include result storage."""
    title_repo, classified_links = process_questions(path, limit)
    
    for title, data in title_repo.items():
        processed_content = data.get('processed_content')
        if processed_content:
            try:
                # Get query type
                query_type = check_query_type(title)
                
                # Get original question body from data
                question_body = data.get('Body', '')
                
                # Get reference answers/repos
                reference_answers = data.get('Repos', [])
                
                # Analyze similarity
                analysis = analyze_similarity_and_extract_links(
                    question=title,
                    processed_content=processed_content,
                    top_k=25
                )
                
                # Store evaluations
                evaluations = []
                if analysis:
                    for chunk in analysis['top_chunks']:
                        evaluation_prompt = f"""You are a content evaluator. Score how well this content answers the question.
                        Return ONLY this JSON format:
                        {{
                            "score": XX.XX,
                            "justification": "Detailed analysis of relevance"
                        }}

                        Question: {title}
                        Content to evaluate: {chunk['chunk_text']}

                        Scoring Guidelines:
                        1. Base score (0-40): Relevant programming concepts
                        2. Additional points (0-30): Code examples
                        3. Quality points (0-30): Clarity of explanation
                        """
                        
                        try:
                            evaluation = llm_prompt(evaluation_prompt)
                            response_text = evaluation.choices[0].message.content.strip()
                            response_text = response_text.replace('\n', ' ').replace('\r', '')
                            
                            # Clean and parse JSON response
                            if not response_text.startswith('{'): 
                                response_text = '{' + response_text.split('{', 1)[1]
                            if not response_text.endswith('}'): 
                                response_text = response_text.rsplit('}', 1)[0] + '}'
                                
                            result = json.loads(response_text)
                            evaluations.append(result)
                            
                        except Exception as e:
                            print(f"Evaluation failed: {str(e)}")
                            continue
                
            except Exception as e:
                print(f"Error processing question {title}: {str(e)}")
                continue
    
    return title_repo, classified_links

def analyze_similarity_and_extract_links(question: str, processed_content: dict, top_k: int = 25):
    """Analyzes chunk similarity using LSA and extracts codebase links from top chunks."""
    if not processed_content:
        return None
        
    vectors = processed_content['vectors']
    metadata = processed_content['metadata']
    content = processed_content['content']
    vectorizer = processed_content['vectorizer']
    
    # Vectorize the question
    question_vector = vectorizer.transform([question])
    
    # Calculate cosine similarity between question and all chunks
    similarities = cosine_similarity(question_vector, vectors)[0]
    
    # Get indices of top k most similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    codebase_links = set()
    
    # Process top chunks
    for idx in top_indices:
        chunk_metadata = metadata[idx]
        content_type = chunk_metadata['type']
        url = chunk_metadata['url']
        chunk_idx = chunk_metadata['chunk_index']
        
        # Get the original chunk text
        doc = next(doc for doc in content[content_type] if doc['url'] == url)
        chunk_text = doc['chunks'][chunk_idx]
        
        # Extract potential codebase links from chunk
        chunk_links = extract_links(chunk_text)
        if chunk_links:
            # Classify links to find codebases
            classified = classifier(chunk_links)
            codebase_links.update(classified['codebases'])
        
        results.append({
            'url': url,
            'content_type': content_type,
            'similarity_score': similarities[idx],
            'chunk_text': chunk_text,
            'found_codebase_links': classified['codebases'] if chunk_links else []
        })
    
    return {
        'top_chunks': results,
        'all_codebase_links': list(codebase_links)
    }

def create_candidate_list(classified_links: dict, analysis_results: dict) -> dict:
    """Creates and sorts a candidate list based on occurrences."""

    codebase_counts = Counter()
    forum_counts = Counter()
    article_counts = Counter()
    
    # Count in original classified links
    for link in classified_links.get['codebases', []]:
        codebase_counts[link] += 1

    for link in classified_links.get('forums', []):
        forum_counts[link] += 1

    for link in classified_links.get('articles', []):
        article_counts[link] += 1 

    for chunk in analysis_results.get('top_chunks', []):
        for link in chunk.get('found_codebase_links', []):
            codebase_counts[link] += 1        
        for link in chunk.get('found_forum_links', []):
            forum_counts[link] += 1        
        for link in chunk.get('found_article_links', []):
            article_counts[link] += 1    

    sorted_candidates = {
        'codebases': [{'url': link, 'occurrences': count} for link, count in codebase_counts.most_common()],
        'forums': [{'url': link, 'occurrences': count} for link, count in forum_counts.most_common()],
        'articles': [{'url': link, 'occurrences': count} for link, count in article_counts.most_common()],
    }

    return sorted_candidates

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 150) -> List[str]:
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        if len(chunk) < chunk_size * 0.5 and chunks:
            chunks[-1] = chunks[-1] + chunk
            break
            
        chunks.append(chunk)
        start = end - overlap
        
    return chunks

def process_and_vectorize_content(classified_links: dict) -> dict:
    """Processes and vectorizes content from articles and forums."""
    processed_content = {
        'articles': [],
        'forums': []
    }
    
    # Process articles and forums
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for content_type in ['articles', 'forums']:
            for url in classified_links[content_type]:
                futures.append(executor.submit(fetch_and_process_content, url, content_type, processed_content))
        
        for future in futures:
            future.result()
    
    # Vectorize all chunks
    all_chunks = []
    chunk_metadata = []
    
    # Collect all chunks with their metadata
    for content_type in ['articles', 'forums']:
        for doc in processed_content[content_type]:
            for chunk_idx, chunk in enumerate(doc['chunks']):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'url': doc['url'],
                    'type': content_type,
                    'chunk_index': chunk_idx
                })
    
    # Vectorize if we have chunks
    if all_chunks:
        print("\nVectorizing chunks...")
        vectorizer = TfidfVectorizer(max_features=5000)
        vectors = vectorizer.fit_transform(all_chunks)
        
        return {
            'content': processed_content,
            'vectors': vectors,
            'metadata': chunk_metadata,
            'vectorizer': vectorizer
        }
    
    return None

def fetch_and_process_content(url, content_type, processed_content):
    """Fetches and processes content from a URL."""
    try:
        print(f"Fetching content from: {url}")
        content = get_html_content(url)
        
        if content:
            # Clean the content
            cleaned_content = clean_text(content)
            # Chunk the content
            chunks = chunk_text(cleaned_content)
            
            processed_content[content_type].append({
                'url': url,
                'chunks': chunks,
                'original_length': len(cleaned_content)
            })
            print(f"Successfully processed {len(chunks)} chunks")
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")

def process_questions(path: str, limit: int = 5):
    """Processes questions from a JSONL file and classifies links."""
    lines = []
    with open(path, "r") as file:
        for i in range(limit):
            line = file.readline()
            if line:
                lines.append(line)

    for line in lines:
        try:
            json_data = json.loads(line)
            org_question = Question(**json_data)
            check_query_type(org_question.Title)

            print("\nProcessing question:", org_question.Id)
            print("Original question:", org_question.Title)
            
            # Get search results
            search_results = list(search(org_question.Title, stop=10))  # Convert generator to list
            print("Found initial results:", len(search_results))

            # Extract links from content
            link_list = []
            for link in search_results:
                content = get_html_content(link)
                content = clean_text(content)
                if content:
                    extracted_links = extract_links(content)
                    link_list.extend(extracted_links)

            # Normalize and classify all links
            classified_links = classifier(list(link_list + search_results))
            
            # Process and vectorize content
            processed_data = process_and_vectorize_content(classified_links)
            
            # Store results with additional metadata
            title_repo[org_question.Title] = {
                'classified_links': classified_links,
                'processed_content': processed_data,
                'original_question': org_question.Title,
                #'rephrased_question': rephrased,
                'question_id': org_question.Id
            }

            # Print statistics
            print(f"\nClassified results:")
            print(f"- Codebases: {len(classified_links['codebases'])}")
            print(f"- Articles: {len(classified_links['articles'])}")
            print(f"- Forums: {len(classified_links['forums'])}")
            
            if processed_data:
                print(f"\nProcessed content:")
                print(f"- Total chunks: {processed_data['vectors'].shape[0]}")
                print(f"- Vector dimensions: {processed_data['vectors'].shape[1]}")

        except Exception as e:
            print(f"Error processing question: {str(e)}")
            continue

    return title_repo, classified_links
 
 def llm_rephrase(title: str, body: str = ""):
    """Rephrases a question using both title and body for context."""
    prompt = f"""
    Title: {title}
    Additional Context: {body}

    Rephrase the question using both the title and context provided. 
    Your response should:
    1. Be a single question
    2. Incorporate relevant details from both title and context
    3. Not be too lengthy
    4. Keep the original meaning
    5. Only return the rephrased question with no additional text
    """
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ],
        model='llamma3.1',
        temperature=0,
    )
    return chat_completion

def llm_prompt(prompt):
    """Rephrases a question using the OpenAI API."""
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': f'{prompt}',
            }
        ],
        model='llama3.1',
        temperature=0,
    )
    return chat_completion

def save_query_results(
    question_title: str,
    cleaned_body: str,
    model_answer: str,
    model_answer_score: float,
    file_path: str = 'query_results.jsonl'
) -> bool:
    try:
        query_result = {
            "question_title": question_title,
            "question_body": cleaned_body,
            "model_answer": model_answer,
            "model_answer_score": model_answer_score
            #"model_answer_justification": model_answer_justification
        }
        
        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(query_result, f)
            f.write('\n')
        
        return True
    
    except Exception as e:
        print(f"Error saving query results: {str(e)}")
        return False

def get_query_results(query_id: Optional[str] = None, file_path: str = 'detailed_query_results.jsonl') -> List[QueryResults]:
    """
    Retrieves query results from the JSONL file.
    
    Args:
        query_id: Optional specific query ID to retrieve
        file_path: Path to results file
        
    Returns:
        List of QueryResults objects
    """
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    result = QueryResults(**data)
                    if query_id is None or result.query_id == query_id:
                        results.append(result)
                except Exception as e:
                    print(f"Error parsing result: {str(e)}")
                    continue
    except Exception as e:
        print(f"Error reading results: {str(e)}")
    
    return results

# load json object from jsonl file
def check_json_file(question, file_name='results.jsonl'):
    try:
        with open(file_name, 'r', encoding="UTF-8") as f:
            for line in f:
                data = json.loads(line)
                if data["question"] == question:
                    #return QueryType(**data)
                    return QueryType(**data)
    except:
        return None
    return None

# check for existing query in jsonl file, if it is not there evaluate the query type and save object to json file.
def check_query_type(question, file_name='results.jsonl'):
    """use llm to evaluate the most suitable type of platform for answering the question, if it is a codebase, article or forum"""
    if check_json_file(question) == None:
        try:
            # prompt engineering
            query_type = llm_prompt(
                f"""Given the question: {question}, what is the most suitable type of platform to find an answer? Is it a codebase, article, or forum
                give your answer in terms of the platform that is most likely to provide the best answer to the question.
                and the confidence level of your answer.
                in the following dictionary format:                     
                """ + f"""
                {{
                    "question": "{question}",
                    "justification": "<brief-analysis-here>",
                    "choices": [<"codebase", "article", "forum">],
                    "confidence": "<XX.XX>"
                }}
                Remember to consider the nature of the question and the type of information that is likely to be found on each platform.
                1. Codebases are repositories of code that can provide direct solutions to programming problems.
                2. Articles are detailed explanations or tutorials that can provide in-depth knowledge on a topic.
                3. Forums are discussion platforms where users can ask questions and receive answers from the community.
                4. Justification should be a brief analysis of why you think the platform you chose is the most suitable.
                5. Choices should be a list of strings with the options "codebase", "article", and "forum" where the most suitable option is placed first, and the least suitable is placed last, remove the other options if they are not suitable.
                6. Confidence level should be a number between 0.00-100.00, with an accuracy of up to 2 decimal points.
                7. ONLY reply with the dictionary format and do not add any other unnecessary text or symbols.
                """)
            # save object to json object
            json_obj = json.loads(query_type.choices[0].message.content)
            
            # save object to jsonl file if object is valid and not already in the file
            with open(file_name, 'a') as f:
                json.dump(json_obj, f)
                f.write('\n')

            return json_obj["choices"]
        except Exception as e:
            print(f"LLM evaluation failed: {str(e)}")
            return None
    return check_json_file(question).choices

def check_url_status(url, timeout=15):
    """Checks if a URL is accessible."""
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return 200 <= response.status_code < 400
    except:
        try:
            # Some servers block HEAD requests, try GET as fallback
            response = requests.get(url, timeout=timeout, stream=True)
            return 200 <= response.status_code < 400
        except:
            return False

def filter_dead_links(urls):
    """Filters out dead links using synchronous requests."""
    return [url for url in urls if check_url_status(url)]

def classifier(results):
    """Classifies URLs into codebases, articles, and forums."""
    codebases = []
    articles = []
    forums = []

    # First filter out dead links
    live_urls = filter_dead_links(results)
    
    for url in live_urls:

        try:
            url_type = url_classifier(url)
            
            if url_type == 'Codebase':
                codebases.append(url)
            elif url_type == 'Article':
                articles.append(url)
            elif url_type == 'Forum':
                forums.append(url)
            else:
                continue
        except:
             pass

    return {
        'codebases': codebases,
        'articles': articles,
        'forums': forums
    }

def extract_links(text):
    """Extracts all the links from the HTML content."""
    links = []
    try:
        # Extract all the links using regex
        links = re.findall(r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)', text)
    except:
        pass
    return links

def get_html_content(url: str):
    """Fetches the HTML content of the web page from a single link."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        content_type = response.headers.get('Content-Type', '').lower()
        if 'xml' in content_type:
            return BeautifulSoup(response.text, 'lxml-xml', features="xml")  # Use lxml for XML content
        else:
            return BeautifulSoup(response.text, 'html.parser')  # Use html.parser for HTML content
    except:
        # print(f"Failed to fetch {url}")
        return 0

def clean_text(text: str) -> str:
    """Cleans and normalizes text content."""
    if not text:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    
    # Basic text cleaning
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines
    text = text.strip()
    
    return text