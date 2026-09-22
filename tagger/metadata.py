from .targets import TargetProfile

ELLIPSIS = "..."


def truncate_text(text: str, limit: int) -> str:
    """Shorten text to a limit, preferring a word boundary.

    Args:
        text: Text to shorten
        limit: Maximum number of characters in the result

    Returns:
        The original text when it already fits, otherwise a truncated copy
        ending with an ellipsis.
    """
    text = " ".join(text.split())

    if limit <= 0 or len(text) <= limit:
        return text

    if limit <= len(ELLIPSIS):
        return text[:limit]

    head = text[: limit - len(ELLIPSIS)]
    boundary = head.rfind(" ")
    if boundary > 0:
        head = head[:boundary]

    return head.rstrip(" ,;:-") + ELLIPSIS


def normalize_keywords(keywords, max_keywords: int) -> list[str]:
    """Clean, deduplicate and cap a list of keywords.

    Keywords are lower cased, stripped of surrounding punctuation and
    deduplicated while preserving the order returned by the model, which is
    ordered by relevance.

    Args:
        keywords: Raw keyword values returned by the model
        max_keywords: Maximum number of keywords to keep

    Returns:
        The cleaned keyword list
    """
    cleaned: list[str] = []
    seen: set[str] = set()

    for keyword in keywords or []:
        if not isinstance(keyword, str):
            continue

        normalized = " ".join(keyword.split()).strip("#,.;:-").lower()
        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        cleaned.append(normalized)

        if max_keywords > 0 and len(cleaned) >= max_keywords:
            break

    return cleaned


def normalize_metadata(raw: dict, profile: TargetProfile) -> dict:
    """Validate and normalize the metadata returned by the model.

    Args:
        raw: Parsed JSON payload returned by the model
        profile: Target profile describing the destination

    Returns:
        A mapping with the title, description and keywords keys

    Raises:
        ValueError: If a required field is missing or empty
    """
    if not isinstance(raw, dict):
        raise ValueError("Model response is not a JSON object")

    title = raw.get("title")
    description = raw.get("description")

    if not isinstance(title, str) or not title.strip():
        raise ValueError("Model response is missing a title")

    if not isinstance(description, str) or not description.strip():
        raise ValueError("Model response is missing a description")

    keywords = normalize_keywords(raw.get("keywords"), profile.max_keywords)
    if not keywords:
        raise ValueError("Model response is missing keywords")

    return {
        "title": truncate_text(title, profile.title_max_chars),
        "description": truncate_text(description, profile.description_max_chars),
        "keywords": keywords,
    }
