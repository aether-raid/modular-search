from pydantic import BaseModel, HttpUrl

class Question(BaseModel):
    Id: int
    AnswerIds: List[int]
    Repos: List[str]
    Title: str
    Body: str

class QueryType(BaseModel):
    id: int
    question: str
    justification: str
    choices: List[str]
    confidence: float

class ChunkResult:
    def __init__(self, content_type: str, url: str, similarity_score: float, chunk_text: str, found_codebase_links: List[str] = []):
        self.content_type = content_type
        self.url = url
        self.similarity_score = similarity_score
        self.chunk_text = chunk_text
        self.found_codebase_links = found_codebase_links

    def to_dict(self):
        return {
            'content_type': self.content_type,
            'url': self.url,
            'similarity_score': self.similarity_score,
            'chunk_text': self.chunk_text,
            'found_codebase_links': self.found_codebase_links
        }
    
class QueryEvaluation(BaseModel):
    score: float
    justification: str

class QueryResults(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    query_id: str
    timestamp: str
    original_question: str
    rephrased_question: Optional[str]
    query_type: List[str]
    top_chunks: List[ChunkResult]
    chunk_evaluations: List[QueryEvaluation]
    average_score: float
    reference_answers: List[str]