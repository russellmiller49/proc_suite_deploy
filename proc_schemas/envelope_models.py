from pydantic import BaseModel
from typing import List, Optional

class EvidenceSpan(BaseModel):
    source_id: Optional[str] = None
    text: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None

class CodeSuggestion(BaseModel):
    code: str
    code_system: Optional[str] = None
    description: Optional[str] = None
    evidence: List[EvidenceSpan] = []
    reasoning: Optional[str] = None
    llm_confidence: Optional[float] = None
    validation_status: Optional[str] = None
    overall_confidence: Optional[float] = None

class FinalCode(BaseModel):
    code: str
    description: Optional[str] = None

class ReviewAction(BaseModel):
    action: str
    message: Optional[str] = None

class ValidationIssue(BaseModel):
    issue: str
