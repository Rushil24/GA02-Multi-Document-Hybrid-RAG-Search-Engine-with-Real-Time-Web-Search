from dataclasses import dataclass
from typing import Dict

@dataclass
class Document:
    source_id: str
    source_type: str      # pdf | web
    title: str
    content: str
    metadata: Dict

@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str

@dataclass
class AnswerSource:
    source_type: str      # doc | web
    reference: str
    