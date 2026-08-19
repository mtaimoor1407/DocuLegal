import json
from typing import List, Dict, Optional
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.documents import Document
import os
from models.schemas import LegalAnswer, SourceClause, ConfidenceLevel
from core.prompts import (
    LEGAL_RAG_PROMPT,
    COMPARISON_RAG_PROMPT,
    REPHRASE_PROMPT
)


def build_llm(provider: str = "groq"):

    if provider == "groq":
        return ChatGroq(
            model       = "openai/gpt-oss-120b",
            temperature = 0,
            max_tokens  = 2048,
            api_key     = os.getenv("GROQ_API_KEY")
        )

    elif provider == "groq-fast":
        return ChatGroq(
            model       = "openai/gpt-oss-20b",
            temperature = 0,
            max_tokens  = 2048,
            api_key     = os.getenv("GROQ_API_KEY")
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

def parse_legal_answer(raw_json: dict, retrieved_docs: List[Document]) -> LegalAnswer:
    """
    Parse the LLM's JSON output into a LegalAnswer object.
    Enriches source information from retrieved documents.
    """

    sources = []
    for src in raw_json.get("sources", []):
        source = SourceClause(
            source_file = src.get("source_file", "unknown"),
            page_number = src.get("page_number"),
            section     = src.get("section"),
            excerpt     = src.get("excerpt", "")
        )
        sources.append(source)

    if not sources and retrieved_docs:
        for doc in retrieved_docs[:2]:
            source = SourceClause(
                source_file = doc.metadata.get("filename", "unknown"),
                page_number = doc.metadata.get("page"),
                section     = doc.metadata.get("section"),
                excerpt     = doc.page_content[:200] + "..."
            )
            sources.append(source)

    confidence_str = raw_json.get("confidence", "low")
    try:
        confidence = ConfidenceLevel(confidence_str)
    except ValueError:
        confidence = ConfidenceLevel.LOW

    return LegalAnswer(
        answer          = raw_json.get("answer", "Unable to parse answer."),
        confidence      = confidence,
        sources         = sources,
        needs_lawyer    = raw_json.get("needs_lawyer", False),
        ambiguity_flag  = raw_json.get("ambiguity_flag", False),
        warning         = raw_json.get("warning"),
        comparison_mode = raw_json.get("comparison_mode", False)
    )


def format_docs_for_context(docs: List[Document]) -> str:
    """Format retrieved documents into a clean context string."""

    pieces = []
    for i, doc in enumerate(docs):
        filename    = doc.metadata.get("filename", "unknown")
        page        = doc.metadata.get("page", "?")
        chunk_type  = doc.metadata.get("chunk_type", "")

        header = f"[Document {i+1} | File: {filename} | Page: {page}]"
        pieces.append(f"{header}\n{doc.page_content}")

    return "\n\n{'─'*50}\n\n".join(pieces)


def _extract_sources_from_docs(
    retrieved_docs: List[Document]
) -> List[SourceClause]:
    """Build source list directly from retrieved documents as fallback."""
    sources = []
    for doc in retrieved_docs[:3]:
        source = SourceClause(
            source_file = doc.metadata.get("filename", "unknown"),
            page_number = doc.metadata.get("page"),
            section     = doc.metadata.get("section"),
            excerpt     = doc.page_content[:200] + "..."
                          if len(doc.page_content) > 200
                          else doc.page_content
        )
        sources.append(source)
    return sources

class DocuLegalChain:
    """
    Main RAG chain for DocuLegal.
    Handles both single-document Q&A and multi-document comparison.
    """

    def __init__(self, retriever, provider: str = "groq"):
        self.retriever      = retriever
        self.llm            = build_llm(provider)   # ← changed
        self.json_parser    = JsonOutputParser()
        self.chat_history   : List = []

        self.history_aware_retriever = create_history_aware_retriever(
            llm         = self.llm,
            retriever   = self.retriever,
            prompt      = REPHRASE_PROMPT
        )

    def ask(
        self,
        question        : str,
        comparison_mode : bool = False
    ) -> LegalAnswer:
        """
        Ask a question and get a structured legal answer.
        Maintains conversation history automatically.
        """

        if comparison_mode:
            prompt = COMPARISON_RAG_PROMPT
        else:
            prompt = LEGAL_RAG_PROMPT

        retrieved_docs = self.history_aware_retriever.invoke({
            "input"      : question,
            "chat_history"  : self.chat_history
        })

        context = format_docs_for_context(retrieved_docs)

        filled_prompt = prompt.invoke({
            "context"       : context,
            "question"      : question,
            "chat_history"  : self.chat_history
        })

        response = self.llm.invoke(filled_prompt)

        legal_answer = self._parse_response(response.content, retrieved_docs)

        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response.content))

        return legal_answer

    def _parse_response(
        self,
        raw_content     : str,
        retrieved_docs  : List[Document]
    ) -> LegalAnswer:
        """
        Parse LLM response into LegalAnswer with multi-layer fallback.
        Handles plain JSON, markdown code blocks, and truncated responses.
        """

        import re

        cleaned = raw_content.strip()

        # ── Layer 1: strip ALL markdown code fences ──────────────────
        # Handles ```json, ```JSON, ``` variations
        cleaned = re.sub(r'^```[a-zA-Z]*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```$',          '', cleaned)
        cleaned = cleaned.strip()

        # ── Layer 2: direct JSON parse ────────────────────────────────
        try:
            raw_json = json.loads(cleaned)
            return parse_legal_answer(raw_json, retrieved_docs)
        except json.JSONDecodeError:
            pass

        # ── Layer 3: find JSON block with regex ───────────────────────
        try:
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                raw_json = json.loads(json_match.group())
                return parse_legal_answer(raw_json, retrieved_docs)
        except Exception:
            pass

        # ── Layer 4: fix truncated JSON ───────────────────────────────
        # Sometimes the LLM hits token limit mid-JSON
        # Try to close open brackets and parse
        try:
            partial = cleaned
            open_braces   = partial.count('{') - partial.count('}')
            open_brackets = partial.count('[') - partial.count(']')
            open_quotes   = partial.count('"') % 2

            if open_quotes:
                partial += '"'
            if open_brackets > 0:
                partial += ']' * open_brackets
            if open_braces > 0:
                partial += '}' * open_braces

            raw_json = json.loads(partial)
            result = parse_legal_answer(raw_json, retrieved_docs)
            result.warning = "Response was truncated — answer may be incomplete."
            return result
        except Exception:
            pass

        # ── Layer 5: extract answer field only ────────────────────────
        try:
            answer_match = re.search(
                r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"',
                cleaned,
                re.DOTALL
            )
            if answer_match:
                answer_text = answer_match.group(1)
                answer_text = answer_text.replace('\\n', '\n').replace('\\"', '"')
                return LegalAnswer(
                    answer          = answer_text,
                    confidence      = ConfidenceLevel.MEDIUM,
                    sources         = _extract_sources_from_docs(retrieved_docs),
                    needs_lawyer    = False,
                    ambiguity_flag  = False,
                    warning         = None,
                    comparison_mode = False
                )
        except Exception:
            pass

        # ── Layer 6: raw text fallback ────────────────────────────────
        # If everything fails, just show whatever the LLM said
        # Strip any remaining JSON artifacts for cleaner display
        display_text = cleaned
        if display_text.startswith('{'):
            # Try to extract readable text from broken JSON
            text_match = re.search(r'"answer"\s*:\s*"(.+)', display_text, re.DOTALL)
            if text_match:
                display_text = text_match.group(1).split('",')[0]
                display_text = display_text.replace('\\n', '\n').replace('\\"', '"')

        return LegalAnswer(
            answer          = display_text,
            confidence      = ConfidenceLevel.LOW,
            sources         = _extract_sources_from_docs(retrieved_docs),
            needs_lawyer    = False,
            ambiguity_flag  = False,
            warning         = None,
            comparison_mode = False
        )

    def clear_history(self):
        """Clear conversation history for a new session."""
        self.chat_history = []
        print("Conversation history cleared.")

    def get_history_length(self) -> int:
        """Return number of messages in history."""
        return len(self.chat_history)