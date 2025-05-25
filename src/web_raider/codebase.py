from enum import Enum
import requests
import base64
import re
from typing import Union
from urllib.parse import urlparse, urlunparse
#from .assets.key_import import GITHUB_TOKEN

class CodebaseType(str, Enum):
    """
    A class to define a set of codebase types by extending `Enum`.

    Members
    -------
    GITHUB: str
        represents GitHub   
    GITLAB: str
        represents GitLab
    BITBUCKET: str
        represents BitBucket
    SOURCEFORGE: str
        represents SourceForge
    GITEE: str
        represents Gitee
    """
    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"
    # SOURCEFORGE = "SourceForge"
    GITEE = "gitee"

    @classmethod
    def format_type(cls, codebase_type: 'CodebaseType') -> str:
        """
        Format the codebase type to a specific string representation.

        Parameters
        ----------
        codebase_type : CodebaseType
            The codebase type to format.

        Returns
        -------
        str
            The formatted string representation of the codebase type.
        """
        """
        Format the codebase type to a specific string representation.

        Parameters
        ----------
        codebase_type : CodebaseType
            The codebase type to format.

        Returns
        -------
        str
            The formatted string representation of the codebase type.
        """
        return codebase_type.value

class Codebase:
    """
    A class that holds the required information of codebases.

    Attributes
    ----------
    original_url: str
        The original URL of the repository.
    repository_url: str
        The URL of the repository with only the repository part.
    type: CodebaseType
        The type of the codebase (e.g., GitHub, GitLab, etc.).

    Methods
    -------
    __new__(cls, url: str)
        Determines the type of codebase based on the domain and returns an instance of the corresponding subclass.
    
    __init__(self, url: str)
        Initializes the Codebase instance with the given URL.
    
    __str__(self)
        Returns a string representation of the Codebase instance.
    
    __repr__(self)
        Returns a string representation of the Codebase instance.
    
    is_code(url: str) -> bool
        Determines if the given URL is a code repository URL.
    """
    def __new__(cls, url: str) -> 'Codebase':
        """
        Determines the type of codebase based on the domain and returns an instance of the corresponding subclass.

        Parameters
        ----------
        url : str
            The URL of the repository.

        Returns
        -------
        Codebase
            An instance of the corresponding subclass of Codebase.
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        if "github.com" in domain:
            return super(Codebase, GitHubCodebase).__new__(GitHubCodebase)
        elif "gitlab.com" in domain:
            return super(Codebase, GitLabCodebase).__new__(GitLabCodebase)
        elif "bitbucket.org" in domain:
            return super(Codebase, BitBucketCodebase).__new__(BitBucketCodebase)
        # elif "sourceforge.net" in domain:
        #     return super(Codebase, SourceForgeCodebase).__new__(SourceForgeCodebase)
        elif "gitee.com" in domain:
            return super(Codebase, GiteeCodebase).__new__(GiteeCodebase)
        else:
            raise ValueError("Unsupported codebase type")
        
    def __init__(self, url: str) -> None:
        """
        Initializes the GitHubCodebase instance with the given URL.

        Parameters
        ----------
        url : str
            The URL of the GitHub repository.
        """
        """
        Initializes the Codebase instance with the given URL.

        Parameters
        ----------
        url : str
            The URL of the repository.
        """
        self.original_url = url
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        path = parsed_url.path.lower()

        if "github.com" in domain:
            self.type = CodebaseType.GITHUB
            path_parts = path.split('/')[:3]  # Keep only username and repo name
        elif "gitlab.com" in domain:
            self.type = CodebaseType.GITLAB
            path_parts = path.split('/')[:3]  # Keep only username and repo name
        elif "bitbucket.org" in domain:
            self.type = CodebaseType.BITBUCKET
            path_parts = path.split('/')[:3]  # Keep only username and repo name
        # elif "sourceforge.net" in domain:
        #     self.type = CodebaseType.SOURCEFORGE
        #     path_parts = path.split('/')[:3]  # Keep only username and repo name
        elif "gitee.com" in domain:
            self.type = CodebaseType.GITEE
            path_parts = path.split('/')[:3]  # Keep only username and repo name
        else:
            raise ValueError("Unsupported codebase type")
        
        # Reconstruct the URL with only the repository part
        truncated_path = '/'.join(path_parts)
        truncated_parsed_url = parsed_url._replace(path=truncated_path, query='', fragment='')
        self.repository_url = urlunparse(truncated_parsed_url)
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Codebase instance.

        Returns
        -------
        str
            A string representation of the Codebase instance.
        """
        return f'{CodebaseType.format_type(self.type)} Codebase({self.original_url})'
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the Codebase instance.

        Returns
        -------
        str
            A string representation of the Codebase instance.
        """
        return str(self)

    @staticmethod
    def is_code(url: str) -> bool:
        """
        Determines if the given URL is a code repository URL.

        Parameters
        ----------
        url : str
            The URL to check.

        Returns
        -------
        bool
            True if the URL is a code repository URL, False otherwise.
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        path = parsed_url.path.lower()

        if "github.com" in domain and 'docs.github.com' not in domain and '/settings/' not in path and '/sponsors/' not in path:
            return True
        elif "gitlab.com" in domain and "/-/blob/" not in path:
            return True
        elif "bitbucket.org" in domain and "/src/" not in path:
            return True
        # elif "sourceforge.net" in domain and "/projects/" in path:
        #     return True
        elif "gitee.com" in domain and "/blob/" not in path:
            return True
        else:
            return False


class GitHubCodebase(Codebase):
    def __init__(self, url: str) -> None:
        super().__init__(url)
        if self.type != CodebaseType.GITHUB:
            raise ValueError("This is not a GitHub codebase")
        
    def check_is_repo(self) -> bool:
        """
        Checks if the URL is a GitHub repository URL.

        Returns
        -------
        bool
            True if the URL is a GitHub repository URL, False otherwise.
        """
        # Regular expression patterns for profile and repository
        profile_pattern = r'^https://github\.com/[^/]+$'
        repo_pattern = r'^https://github\.com/[^/]+/[^/]+$'
        
        if re.match(profile_pattern, self.repository_url):
            return False
        elif re.match(repo_pattern, self.repository_url):
            return True
        else:
            return False

    def get_id(self) -> tuple[str, str]:
        """
        Extracts the owner and repository name from the GitHub repository URL.

        Returns
        -------
        tuple
            A tuple containing the owner and repository name.
        """
        # Extract owner and repo from the repository URL
        path_parts = urlparse(self.repository_url).path.split('/')
        owner, repo = path_parts[1], path_parts[2]
        
        return owner, repo
    
    def get_topics(self) -> list:
        """
        Fetches the topics of the GitHub repository.

        Returns
        -------
        list
            A list of topics associated with the GitHub repository.
        """
        # Extract owner and repo from the repository URL
        owner, repo = self.get_id()

        # Construct the GitHub API URL
        api_url = f"https://api.github.com/repos/{owner}/{repo}/topics"

        # Make a GET request to the GitHub API
        headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {GITHUB_TOKEN}"}
        response = requests.get(api_url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Extract and return the topics
            return response.json()["names"]
        else:
            # Handle errors (e.g., repository not found, API rate limit exceeded)
            print(f"Error fetching topics: {response.status_code}")
            return []
        
    def get_readme(self) -> str:
        """
        Fetches the README content of the GitHub repository.

        Returns
        -------
        str
            The decoded content of the README file.
        """
        # Extract owner and repo from the repository URL
        owner, repo = self.get_id()

        # Construct the GitHub API URL for the README
        api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"

        # Make a GET request to the GitHub API
        headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {GITHUB_TOKEN}"}
        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            content = response.json()["content"]
            decoded_content = base64.b64decode(content).decode('utf-8')
            return decoded_content
        else:
            print(f"Error fetching README: {response.status_code}")
            return ''

    def get_repo_desc(self) -> str:
        """
        Fetches the description of the GitHub repository.

        Returns
        -------
        str
            The description of the GitHub repository.
        """
        # Extract owner and repo from the repository URL
        owner, repo = self.get_id()
        
        # Construct the GitHub API URL for the README
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        # Make a GET request to the GitHub API
        headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {GITHUB_TOKEN}"}
        response = requests.get(api_url, headers=headers)
        
        if response.status_code == 200:
            return response.json()["description"]
        else:
            print(f"Error fetching repo description: {response.status_code}")
            return ''
        
    def get_license(self) -> str:
        # Extract owner and repo from the repository URL
        owner, repo = self.get_id()

        # Construct the GitHub API URL for the README
        api_url = f"https://api.github.com/repos/{owner}/{repo}/license"
        # Make a GET request to the GitHub API
        headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {GITHUB_TOKEN}"}
        response = requests.get(
            api_url, 
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json().get('license', {}).get('name', '')
        else:
            print(f"Error fetching repo license: {response.status_code}")
            return ''
        
    def combine_info(self) -> Union[dict, None]:
        """
        Combines the topics, README content, and description of the GitHub repository into a dictionary.

        Returns
        -------
        dict
            A dictionary containing the topics, README content, and description of the GitHub repository.
        """
        # if correct type
        if self.check_is_repo():
            topics = self.get_topics()
            readme = self.get_readme()
            desc = self.get_repo_desc()
            license = self.get_license()
            
            info_dict = {
                'topics': topics,
                'readme': readme,
                'description': desc,
                'license': license
            }

            return info_dict
        
        else:
            return None

class GitLabCodebase(Codebase):
    def __init__(self, url: str) -> None:
        super().__init__(url)
        if self.type != CodebaseType.GITLAB:
            raise ValueError("This is not a GitLab codebase")
        
    def check_is_repo(self) -> bool:
        """
        Checks if the URL is a GitLab repository URL.

        Returns
        -------
        bool
            True if the URL is a GitLab repository URL, False otherwise.
        """
        profile_pattern = r'^https://gitlab\.com/[^/]+$'
        repo_pattern = r'^https://gitlab\.com/[^/]+/[^/]+$'

        if re.match(profile_pattern, self.repository_url):
            return False
        
        elif re.match(repo_pattern, self.repository_url):
            return True
        else:
            return False

    def get_id(self) -> tuple[str, str]:
        """
        Extracts the owner and repository name from the GitLab repository URL.

        Returns
        -------
        tuple
            A tuple containing the owner and repository name.
        """
        path_parts = urlparse(self.repository_url).path.split('/')
        owner, repo = path_parts[1], path_parts[2]
        
        return owner, repo
    
    def get_topics(self) -> list:
        """
        Fetches the topics of the GitLab repository.

        Returns
        -------
        list
            A list of topics associated with the GitLab repository.
        """
        owner, repo = self.get_id()
        api_url = f"https://gitlab.com/api/v4/projects/{owner}%2F{repo}"
        # headers = {"PRIVATE-TOKEN": "YOUR_GITLAB_API_KEY"}
        response = requests.get(
            api_url,
            # headers=headers
        )

        if response.status_code == 200:
            return response.json().get("tag_list", [])
        else:
            print(f"Error fetching topics: {response.status_code}")
            return []
        
    def get_readme(self) -> str:
        """
        Fetches the README content of the GitLab repository.

        Returns
        -------
        str
            The decoded content of the README file.
        """
        owner, repo = self.get_id()
        api_url = f"https://gitlab.com/api/v4/projects/{owner}%2F{repo}/repository/files/README.md/raw"
        # headers = {"PRIVATE-TOKEN": "YOUR_GITLAB_API_KEY"}
        response = requests.get(
            api_url,
            # headers=headers
        )

        if response.status_code == 200:
            return response.text
        else:
            print(f"Error fetching README: {response.status_code}")
            return ''

    def get_repo_desc(self) -> str:
        """
        Fetches the description of the GitLab repository.

        Returns
        -------
        str
            The description of the GitLab repository.
        """
        owner, repo = self.get_id()
        api_url = f"https://gitlab.com/api/v4/projects/{owner}%2F{repo}"
        # headers = {"PRIVATE-TOKEN": "YOUR_GITLAB_API_KEY"}
        response = requests.get(
            api_url,
            # headers=headers
        )

        if response.status_code == 200:
            return response.json().get("description", "")
        else:
            print(f"Error fetching repo description: {response.status_code}")
            return ''
        
    def get_license(self) -> str:
        # Extract owner and repo from the repository URL
        owner, repo = self.get_id()

        # Construct the GitLab API URL for the README
        api_url = f"https://gitlab.com/api/v4/projects/{owner}%2F{repo}/repository/files/LICENSE/raw"
        # Make a GET request to the GitLab API
        # headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {GITLAB_TOKEN}"}
        response = requests.get(
            api_url, 
            # headers=headers
        )
        
        if response.status_code == 200:
            return response.text
        else:
            print(f"Error fetching repo license: {response.status_code}")
            return ''
        
    def combine_info(self) -> Union[dict, None]:
        """
        Combines the topics, README content, and description of the GitLab repository into a dictionary.

        Returns
        -------
        dict
            A dictionary containing the topics, README content, and description of the GitLab repository.
        """

        # initially implemented check_is_repo here
        # but shifted it to src/pipeline.py instead
        topics = self.get_topics()
        readme = self.get_readme()
        desc = self.get_repo_desc()
        license = self.get_license()
        
        info_dict = {
            'topics': topics,
            'readme': readme,
            'description': desc,
            'license': license
        }

        return info_dict

class BitBucketCodebase(Codebase):
    def __init__(self, url: str) -> None:
        super().__init__(url)
        if self.type != CodebaseType.BITBUCKET:
            raise ValueError("This is not a BitBucket codebase")
        
    def check_is_repo(self) -> bool:
        """
        Checks if the URL is a BitBucket repository URL.

        Returns
        -------
        bool
            True if the URL is a BitBucket repository URL, False otherwise.
        """
        profile_pattern = r'^https://bitbucket\.org/[^/]+/workspace$'
        repo_pattern = r'^https://bitbucket\.org/[^/]+/[^/]+$'
        
        if re.match(profile_pattern, self.repository_url):
            return False
        elif re.match(repo_pattern, self.repository_url):
            return True
        else:
            return False

    def get_id(self) -> tuple[str, str]:
        """
        Extracts the owner and repository name from the BitBucket repository URL.

        Returns
        -------
        tuple
            A tuple containing the owner and repository name.
        """
        path_parts = urlparse(self.repository_url).path.split('/')
        owner, repo = path_parts[1], path_parts[2]
        
        return owner, repo
    
    def get_topics(self) -> list:
        """
        Fetches the topics of the BitBucket repository.

        Returns
        -------
        list
            A list of topics associated with the BitBucket repository.
        """
        owner, repo = self.get_id()
        api_url = f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}"
        response = requests.get(api_url)

        if response.status_code == 200:
            return response.json().get("tags", [])
        else:
            print(f"Error fetching topics: {response.status_code}")
            return []
        
    def get_readme(self) -> str:
        """
        Fetches the README content of the BitBucket repository.

        Returns
        -------
        str
            The decoded content of the README file.
        """
        owner, repo = self.get_id()
        api_url = f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}/src/master/README.md"
        response = requests.get(api_url)

        if response.status_code == 200:
            return response.text
        else:
            print(f"Error fetching README: {response.status_code}")
            return ''

    def get_repo_desc(self) -> str:
        """
        Fetches the description of the BitBucket repository.

        Returns
        -------
        str
            The description of the BitBucket repository.
        """
        owner, repo = self.get_id()
        api_url = f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}"
        response = requests.get(api_url)

        if response.status_code == 200:
            return response.json().get("description", "")
        else:
            print(f"Error fetching repo description: {response.status_code}")
            return ''
        
    def get_license(self) -> str:
        # Extract owner and repo from the repository URL
        owner, repo = self.get_id()

        # Construct the BitBucket API URL for the README
        api_url = f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}/src/master/LICENSE"
        # Make a GET request to the BitBucket API
        # headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {BITBUCKET_TOKEN}"}
        response = requests.get(
            api_url, 
            # headers=headers
        )
        
        if response.status_code == 200:
            return response.text
        else:
            print(f"Error fetching repo license: {response.status_code}")
            return ''
        
    def combine_info(self) -> Union[dict, None]:
        """
        Combines the topics, README content, and description of the BitBucket repository into a dictionary.

        Returns
        -------
        dict
            A dictionary containing the topics, README content, and description of the BitBucket repository.
        """
        if self.check_is_repo():
            topics = self.get_topics()
            readme = self.get_readme()
            desc = self.get_repo_desc()
            license = self.get_license()
            
            info_dict = {
                'topics': topics,
                'readme': readme,
                'description': desc,
                'license': license
            }

            return info_dict
        
        else:
            return None

# class SourceForgeCodebase(Codebase):
#     def __init__(self, url: str) -> None:
#         super().__init__(url)
#         if self.type != CodebaseType.SOURCEFORGE:
#             raise ValueError("This is not a SourceForge codebase")
        
#     def check_is_repo(self) -> bool:
#         """
#         Checks if the URL is a SourceForge repository URL.

#         Returns
#         -------
#         bool
#             True if the URL is a SourceForge repository URL, False otherwise.
#         """
#         repo_pattern = r'^https://sourceforge\.net/projects/[^/]+$'
        
#         if re.match(repo_pattern, self.repository_url):
#             return True
#         else:
#             return False

#     def get_id(self) -> tuple[str, str]:
#         """
#         Extracts the project name from the SourceForge repository URL.

#         Returns
#         -------
#         str
#             The project name.
#         """
#         path_parts = urlparse(self.repository_url).path.split('/')
#         project = path_parts[2]
        
#         return project
    
#     def get_topics(self) -> list:
#         """
#         Fetches the topics of the SourceForge repository.

#         Returns
#         -------
#         list
#             A list of topics associated with the SourceForge repository.
#         """
#         project = self.get_id()
#         api_url = f"https://sourceforge.net/rest/p/{project}"
#         response = requests.get(api_url)

#         if response.status_code == 200:
#             return response.json().get("tags", [])
#         else:
#             print(f"Error fetching topics: {response.status_code}")
#             return []
        
#     def get_readme(self) -> str:
#         """
#         Fetches the README content of the SourceForge repository.

#         Returns
#         -------
#         str
#             The decoded content of the README file.
#         """
#         project = self.get_id()
#         api_url = f"https://sourceforge.net/projects/{project}/files/README.md/download"
#         response = requests.get(api_url)

#         if response.status_code == 200:
#             return response.text
#         else:
#             print(f"Error fetching README: {response.status_code}")
#             return ''

#     def get_repo_desc(self) -> str:
#         """
#         Fetches the description of the SourceForge repository.

#         Returns
#         -------
#         str
#             The description of the SourceForge repository.
#         """
#         project = self.get_id()
#         api_url = f"https://sourceforge.net/rest/p/{project}"
#         response = requests.get(api_url)

#         if response.status_code == 200:
#             return response.json().get("short_description", "")
#         else:
#             print(f"Error fetching repo description: {response.status_code}")
#             return ''
        
#     def combine_info(self) -> Union[dict, None]:
#         """
#         Combines the topics, README content, and description of the SourceForge repository into a dictionary.

#         Returns
#         -------
#         dict
#             A dictionary containing the topics, README content, and description of the SourceForge repository.
#         """
#         if self.check_is_repo():
#             topics = self.get_topics()
#             readme = self.get_readme()
#             desc = self.get_repo_desc()
            
#             info_dict = {
#                 'topics': topics,
#                 'readme': readme,
#                 'description': desc
#             }

#             return info_dict
        
#         else:
#             return None

class GiteeCodebase(Codebase):
    def __init__(self, url):
        super().__init__(url)
        if self.type != CodebaseType.GITEE:
            raise ValueError("This is not a Gitee codebase")
        
    def check_is_repo(self) -> bool:
        """
        Checks if the URL is a Gitee repository URL.

        Returns
        -------
        bool
            True if the URL is a Gitee repository URL, False otherwise.
        """
        profile_pattern = r'^https://gitee\.com/[^/]+$'
        repo_pattern = r'^https://gitee\.com/[^/]+/[^/]+$'
        
        if re.match(profile_pattern, self.repository_url):
            return False
        elif re.match(repo_pattern, self.repository_url):
            return True
        else:
            return False

    def get_id(self) -> tuple[str, str]:
        """
        Extracts the owner and repository name from the Gitee repository URL.

        Returns
        -------
        tuple
            A tuple containing the owner and repository name.
        """
        path_parts = urlparse(self.repository_url).path.split('/')
        owner, repo = path_parts[1], path_parts[2]
        
        return owner, repo
    
    def get_topics(self) -> str:
        """
        Fetches the topics of the Gitee repository.

        Returns
        -------
        list
            A list of topics associated with the Gitee repository.
        """
        owner, repo = self.get_id()
        api_url = f"https://gitee.com/api/v5/repos/{owner}/{repo}/topics"
        # headers = {"Authorization": "YOUR_GITEE_API_KEY"}
        response = requests.get(
            api_url, 
            # headers=headers
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching topics: {response.status_code}")
            return []
        
    def get_readme(self) -> str:
        """
        Fetches the README content of the Gitee repository.

        Returns
        -------
        str
            The decoded content of the README file.
        """
        owner, repo = self.get_id()
        api_url = f"https://gitee.com/api/v5/repos/{owner}/{repo}/readme"
        # headers = {"Authorization": "YOUR_GITEE_API_KEY"}
        response = requests.get(
            api_url, 
            # headers=headers
        )

        if response.status_code == 200:
            content = response.json()["content"]
            decoded_content = base64.b64decode(content).decode('utf-8')
            return decoded_content
        else:
            print(f"Error fetching README: {response.status_code}")
            return ''

    def get_repo_desc(self) -> str:
        """
        Fetches the description of the Gitee repository.

        Returns
        -------
        str
            The description of the Gitee repository.
        """
        owner, repo = self.get_id()
        api_url = f"https://gitee.com/api/v5/repos/{owner}/{repo}"
        # headers = {"Authorization": "YOUR_GITEE_API_KEY"}
        response = requests.get(
            api_url, 
            # headers=headers
        )

        if response.status_code == 200:
            return response.json().get("description", "")
        else:
            print(f"Error fetching repo description: {response.status_code}")
            return ''
        
    def get_license(self) -> str:
        # Extract owner and repo from the repository URL
        owner, repo = self.get_id()

        # filter for potential license files
        license_types = ['LICENSE', 'LICENSE.md', 'LICENSE.txt']

        for license in license_types:
            # Construct the Gitee API URL for the README
            api_url = f"https://gitee.com/api/v5/repos/{owner}/{repo}/contents/{license}"
            # Make a GET request to the Gitee API
            # headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {GITHUB_TOKEN}"}
            response = requests.get(
                api_url,
                # headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result != []:
                    return base64.b64decode(result.get('content', '')).decode('utf-8')
            else:
                print(f"Error fetching repo license: {response.status_code}")
        
        # failed to get any results
        return ''
        
    def combine_info(self) -> Union[dict, None]:
        """
        Combines the topics, README content, and description of the Gitee repository into a dictionary.

        Returns
        -------
        dict
            A dictionary containing the topics, README content, and description of the Gitee repository.
        """
        if self.check_is_repo():
            topics = self.get_topics()
            readme = self.get_readme()
            desc = self.get_repo_desc()
            license = self.get_license()
            
            info_dict = {
                'topics': topics,
                'readme': readme,
                'description': desc,
                'license': license
            }

            return info_dict
        
        else:
            return None
