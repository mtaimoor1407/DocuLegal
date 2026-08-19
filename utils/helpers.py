from models.schemas import LegalAnswer, ConfidenceLevel


def get_confidence_badge(confidence: ConfidenceLevel) -> str:
    """Return colored emoji badge for confidence level."""
    badges = {
        ConfidenceLevel.HIGH    : "🟢 High",
        ConfidenceLevel.MEDIUM  : "🟡 Medium",
        ConfidenceLevel.LOW     : "🔴 Low",
        ConfidenceLevel.NONE    : "⚫ Not Found"
    }
    return badges.get(confidence, "⚫ Unknown")


def format_answer_for_display(legal_answer: LegalAnswer) -> dict:
    """
    Convert a LegalAnswer into a display-ready dictionary
    for the Streamlit UI.
    """

    sources_display = []
    for src in legal_answer.sources:
        source_str = f"📄 {src.source_file}"
        if src.page_number is not None:
            source_str += f" — Page {src.page_number + 1}"
        if src.section:
            source_str += f" — {src.section}"
        sources_display.append({
            "label"     : source_str,
            "excerpt"   : src.excerpt
        })

    warnings = []
    if legal_answer.needs_lawyer:
        warnings.append("⚖️ This matter may benefit from professional legal advice.")
    if legal_answer.ambiguity_flag:
        warnings.append("⚠️ The relevant clause(s) contain ambiguous language.")
    if legal_answer.warning:
        warnings.append(f"📌 {legal_answer.warning}")

    return {
        "answer"        : legal_answer.answer,
        "confidence"    : get_confidence_badge(legal_answer.confidence),
        "sources"       : sources_display,
        "warnings"      : warnings,
        "is_comparison" : legal_answer.comparison_mode
    }


def truncate_text(text: str, max_chars: int = 300) -> str:
    """Truncate text with ellipsis if it exceeds max_chars."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."