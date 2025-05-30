from typing import Protocol, List, Dict, Union

class LLM(Protocol):
    def __call__(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> str: ...