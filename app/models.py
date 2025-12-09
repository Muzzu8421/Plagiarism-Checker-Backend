from pydantic import BaseModel
from typing import List, Optional

class Match(BaseModel):
    source: Optional[str] = None
    similarity: float
    matchedText: str
    startIndex: int = 0
    endIndex: int = 0

class PlagiarismResult(BaseModel):
    plagiarismScore: float
    originalityScore: float
    matches: List[Match]
    text: str
    wordCount: int
    characterCount: int
    sources: int