'''# assets/key_import.py

import os

# Azure OpenAI gpt4o keys
AZURE_ENDPOINT = os.environ['AZURE_API_BASE']
AZURE_KEY = os.environ['AZURE_API_KEY']
AZURE_MODEL = 'azure/gpt4o'
AZURE_API_VERSION = os.environ['AZURE_API_VERSION']

# AWS Bedrock Keys
AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
AWS_REGION_NAME = os.environ['AWS_REGION_NAME']

# Codebase Keys
GITHUB_TOKEN = os.environ['GITHUB_TOKEN']'''