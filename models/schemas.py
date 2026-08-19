from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ConfidenceLevel(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"
    NONE   = "none"


class SourceClause(BaseModel):
    source_file   : str            = Field(description="Name of the PDF file")
    page_number   : Optional[int]  = Field(description="Page number in the document")
    section       : Optional[str]  = Field(description="Section or clause identifier")
    excerpt       : str            = Field(description="The exact relevant text from the document")


class LegalAnswer(BaseModel):
    answer          : str                  = Field(description="Plain English answer to the question")
    confidence      : ConfidenceLevel      = Field(description="Confidence level of the answer")
    sources         : List[SourceClause]   = Field(description="Source clauses supporting the answer")
    needs_lawyer    : bool                 = Field(description="Whether professional legal advice is recommended")
    ambiguity_flag  : bool                 = Field(description="Whether the relevant clause is ambiguous")
    warning         : Optional[str]        = Field(default=None, description="Any important warning for the user")
    comparison_mode : bool                 = Field(default=False, description="Whether this answer compares multiple documents")


class DocumentInfo(BaseModel):
    filename    : str  = Field(description="Name of the uploaded file")
    page_count  : int  = Field(description="Total number of pages")
    chunk_count : int  = Field(description="Number of chunks created")
    doc_type    : str  = Field(default="legal", description="Type of document")