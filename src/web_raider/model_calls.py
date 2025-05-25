# assets/relevance.py

import litellm
from textwrap import dedent
from openai import OpenAI
from .assets.prompts import Prompts
#from .assets.key_import import AZURE_ENDPOINT, AZURE_KEY, AZURE_MODEL, AZURE_API_VERSION

'''client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
    api_key=AZURE_KEY
)'''

'''litellm.model_alias_map = {
    'sonnet-3': 'anthropic.claude-3-sonnet-20240229-v1:0',
    'sonnet-3.5': 'anthropic.claude-3-5-sonnet-20240620-v1:0'
}'''

def call_query_simplifier(query: str) -> str:
    # simplifier = client.chat.completions.create(
    #     model=AZURE_MODEL,
    #     messages = [
    #         {
    #             'role': 'system',
    #             'content': dedent(Prompts.QUERY_PROMPT)
    #         },
    #         {
    #             'role': 'user',
    #             'content': query
    #         },
    #     ],
    #     temperature=0,
    #     # response_format=response_format
    # )
    client = OpenAI(
        base_url= "http://localhost:11434/v1/",


        api_key = 'ollama'
    )

    litellm_simplifier = litellm.completion(
    model="llama2",
    messages = [
            {
                'role': 'system',
                'content': dedent(Prompts.QUERY_PROMPT)
            },
            {
                'role': 'user',
                'content': query
            },
        ],
        temperature=0,
    )

    # response = simplifier.choices[0].message.content
    response = litellm_simplifier['choices'][0]['message']['content']
    return response

def consolidate_codebases_info(codebases: list[dict]) -> str:
    """Helper function to help consolidate information about codebases"""
    codebases_info = ''

    for i in range(len(codebases)):
        # check if codebase information was not extracted
        # could have various reasons:
        # 1. check_is_repo is False so codebase[i]['info'] == None
        # 2. just didn't work or smth
        if codebases[i]['info'] is not None:
            codebases_info += f'Codebase {i + 1}\n\n'
            codebases_info += f'URL: {codebases[i]["url"]}\n\n'
            codebases_info += f'Topics:\n{codebases[i]["info"]["topics"]}\n\n'
            codebases_info += f'README:\n{codebases[i]["info"]["readme"]}\n\n'
            codebases_info += f'Description:\n{codebases[i]["info"]["description"]}\n\n'

    return codebases_info

def call_relevance(codebases: list[dict], query: str) -> str:
    codebases_info = consolidate_codebases_info(codebases)

    # response_format = json.dumps({
    #     "json_schema": {
    #         'name': 'codebase_relevance',
    #         'description': 'to determine if a codebase is relevant to a user provided query',
    #         "schema": {
    #             "codebase_names": {
    #                 "type": "array",
    #                 "items": {"type": "string"}
    #             },
    #             "relevance": {
    #                 "type": "array",
    #                 "items": {"type": "boolean"}
    #             }
    #         },
    #         'strict': True
    #     },
    #     "type": "json_schema"
    # })

    # relevance = client.chat.completions.create(
    #     model=AZURE_MODEL,
    #     messages = [
    #         {
    #             'role': 'system',
    #             'content': dedent(Prompts.RELEVANCE_PROMPT)
    #         },
    #         {
    #             'role': 'user',
    #             'content': f'Codebase Information: {codebases_info}\n\nUser Query: {query}'
    #         },
    #     ],
    #     temperature=0,
    #     # response_format=response_format
    # )

    litellm_relevance = litellm.completion(
    model="llama2",
    messages = [
            {
                'role': 'system',
                'content': dedent(Prompts.RELEVANCE_PROMPT)
            },
            {
                'role': 'user',
                'content': f'Codebase Information: {codebases_info}\n\nUser Query: {query}'
            },
        ],
        temperature=0,
    )

    # response = relevance.choices[0].message.content
    response = litellm_relevance['choices'][0]['message']['content']
    return response

def call_pro_con(codebases: list[dict], query: str) -> str:
    """
    Note: Somehow, only when the AzureOpenAI model is used can the json string be outputted correctly.
    """
    codebases_info = consolidate_codebases_info(codebases)

    pro_con = client.chat.completions.create(
        model='llama2',
        messages = [
            {
                'role': 'system',
                'content': dedent(Prompts.PRO_CON_PROMPT)
            },
            {
                'role': 'user',
                'content': f'Codebase Information: {codebases_info}\n\nUser Query: {query}'
            },
        ],
        temperature=0,
        # response_format=response_format
    )

    # litellm_pro_con = litellm.completion(
    # model="sonnet-3.5",
    # messages = [
    #         {
    #             'role': 'system',
    #             'content': dedent(Prompts.PRO_CON_PROMPT)
    #         },
    #         {
    #             'role': 'user',
    #             'content': f'Codebase Information: {codebases_info}\n\nUser Query: {query}'
    #         },
    #     ],
    #     temperature=0,
    # )

    response = pro_con.choices[0].message.content
    # response = litellm_pro_con['choices'][0]['message']['content']
    return response

def call_scorer(codebases: list[dict], query: str, pro_con: str) -> str:
    codebases_info = consolidate_codebases_info(codebases)

    # scorer = client.chat.completions.create(
    #     model=AZURE_MODEL,
    #     messages = [
    #         {
    #             'role': 'system',
    #             'content': dedent(Prompts.SCORER_PROMPT)
    #         },
    #         {
    #             'role': 'user',
    #             'content': f'Codebase Information: {codebases_info}\n\nUser Query: {query}\n\nPro/Con Information: {pro_con}'
    #         },
    #     ],
    #     temperature=0,
    # )

    litellm_scorer = litellm.completion(
    model="llama2",
    messages = [
            {
                'role': 'system',
                'content': dedent(Prompts.SCORER_PROMPT)
            },
            {
                'role': 'user',
                'content': f'Codebase Information: {codebases_info}\n\nUser Query: {query}\n\nPro/Con Information: {pro_con}'
            },
        ],
        temperature=0,
    )

    # response = scorer.choices[0].message.content
    response = litellm_scorer['choices'][0]['message']['content']
    return response

def call_ranker(scorer: str) -> str:
    ranker = litellm.completion(
    model="llama2",
    messages = [
            {
                'role': 'system',
                'content': dedent(Prompts.RANKER_PROMPT)
            },
            {
                'role': 'user',
                'content': f'Scoring of Codebases: {scorer}'
            },
        ],
        temperature=0,
    )

    response = ranker['choices'][0]['message']['content']
    return response