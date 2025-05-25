def evaluate_candidates_with_llm(question: str, candidates: List[dict], known_repos: List[str], max_candidates: int = 5) -> dict:
    """
    Evaluates candidate links using LLM based on repository content.
    
    Arguments:
    - question: The original question (str)
    - candidates: List of candidate repository links with occurrence counts (list of dict)
    - known_repos: List of known correct repositories (list of str)
    - max_candidates: Maximum number of candidates to analyze (int)
    
    Returns:
    - Dictionary with the best candidate link and its accuracy score (dict)
    """
    if not candidates:
        return {'best_candidate': None, 'accuracy': 0}, known_repo_content

    # Get content for top candidates
    candidates_with_content = []
    for candidate in candidates[:max_candidates]:
        content = get_repo_content(candidate['url'])
        if content:
            candidates_with_content.append({
                'url': candidate['url'],
                'content': content,
                'occurrences': candidate['occurrences']
            })

    if not candidates_with_content:
        return {'best_candidate': None, 'accuracy': 0}, known_repo_content

    # Prepare evaluation prompt
    known_repo_content = get_repo_content(known_repos[0]) if known_repos else ""

    try:
        # Continue evaluating other candidates if accuracy is low
        for candidate in candidates_with_content:
            evaluation_prompt = f"""
            Question: {question}

            Evaluate the following GitHub repository content to determine if it answers the question.
            Use the known repository content as a reference model answer. Rate the candidate repository from 0-100 based on how well it answers the question. 
            IMPORTANT: You must ONLY return a numeric score.
            RULES:
                1. score MUST be a number (e.g. 75.50, 32.40, etc.)
                2. DO NOT use text like "The rate is" or "out of 100" only the number and nothing else.
                Known Repository Content:
                    {known_repo_content}
    
                Candidate Repository Content:
                    {candidates_with_content[0]['content']}
                """

            evaluation = llm_prompt(evaluation_prompt)
            result = evaluation.choices[0].message.content.strip()
            accuracy = float(result)

            if accuracy >= 70:
                print('best_candidate: ', candidate['url'], '\naccuracy: ', accuracy )
                return {'best_candidate': candidate['url'], 'accuracy': accuracy}, known_repo_content

        return {'best_candidate': candidates_with_content[0]['url'], 'accuracy': accuracy}, known_repo_content

    except Exception as e:
        print(f"LLM evaluation failed: {str(e)}")
        return {'best_candidate': None, 'accuracy': 0}, known_repo_content



def evaluate_model_accuracy(results: dict, known_repos: dict) -> dict:
    """Evaluates model accuracy by comparing found repositories with known repositories."""
    total_matches = 0
    total_repos = 0
    question_metrics = {}
    total_accuracy = 0
    question_count = 0
    
    for title, data in results.items():
        # Get all found repos
        found_repos = set()
        if 'classified_links' in data:
            # Add repos from classified links
            for repo in data['classified_links']['codebases']:
                found_repos.add(repo)

        # Get known repos
        known = set(known_repos[title])

        # Find matches
        matches = found_repos.intersection(known)
        
        # Update counts
        matches_count = len(matches)
        known_count = len(known_repos[title])  # Use original count
        total_matches += matches_count
        total_repos += known_count
        
        # Calculate accuracy
        accuracy = matches_count / known_count if known_count > 0 else 0
        total_accuracy += accuracy
        question_count += 1
        
        # Store metrics
        question_metrics[title] = {
            'accuracy': accuracy,
            'matches_found': list(matches),
            'total_found': len(found_repos),
            'total_known': known_count,
            'all_found_repos': list(found_repos)
        }
        
        # Print detailed results for debugging
        print(f"\nQuestion: {title}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Found {matches_count} out of {known_count} known repositories")
        if matches:
            print("Matched repositories:")
            for repo in matches:
                print(f"- {repo}")

    # Calculate overall accuracy as the average accuracy of each question
    overall_accuracy = total_accuracy / question_count if question_count > 0 else 0
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_matches': total_matches,
        'total_repos': total_repos,
        'question_metrics': question_metrics
    }

def score_model_answer(question_title: str, cleaned_body: str, model_answer: str) -> dict:
    """
    Scores the model answer based on how well it answers the original query.
    
    Args:
        question_title (str): The original question title.
        question_body (str): The original question body.
        model_answer (str): The generated model answer.
        
    Returns:
        A dictionary with the score and justification.
    """
    try:
        score_prompt = f"""
        You are an expert evaluator tasked with assessing the quality and comprehensiveness of model answers to technical questions. Your evaluation should be thorough and balanced.

        Original Question:
        Title: {question_title}
        Body: {cleaned_body}

        Model Answer:
        {model_answer}

        Evaluate the answer considering:
        1. Direct relevance to the question
        2. Comprehensiveness of the explanation
        3. Technical accuracy and depth
        4. Use of cross-references and citations
        5. Clarity and organization
        6. Practical applicability

        Scoring Guidelines:
        - 90-100: Exceptional answer that exceeds expectations
        - 80-89: Strong answer with comprehensive coverage
        - 70-79: Solid answer that meets requirements
        - 60-69: Adequate answer with room for improvement
        - Below 60: Answer needs significant improvement

        Additional points are awarded for:
        - Effective cross-referencing and source integration (+5-10)
        - Practical examples and implementation details (+5-10)
        - Addressing edge cases or potential issues (+5-10)

        IMPORTANT: The score must be a numerical value between 00.00 and 100.00.
        IMPORTANT: No line breaks or special characters
        IMPORTANT: Use only regular quotes ('') and escape them if needed
        IMPORTANT: Justification should be concise and in a single line

        Provide your evaluation in the following JSON format:
        {{
            "score": XX.XX,
            "justification": "Detailed analysis of the answer's strengths and areas for improvement, with specific examples from the response."
        }}
        """
        
        score_response = llm_prompt(score_prompt)
        score_result = json.loads(score_response.choices[0].message.content.strip())
        return score_result
    
    except Exception as e:
        print(f"Error scoring model answer: {str(e)}")
        return {"score": 0, "justification": "Error occurred during evaluation."}