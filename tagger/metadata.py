from .categories import CATEGORY_IDS, PRIMARY_FIELD, SECONDARY_FIELD
from .prompts import SPECIFIC_PLACE_FIELD
from .targets import TargetProfile

ELLIPSIS = "..."
SPECIFIC_PLACE_MAX_CHARS = 100


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


def normalize_category(raw: dict) -> dict:
    """Return the primary and secondary category, after validating them.

    Args:
        raw: Parsed JSON payload returned by the model

    Returns:
        A mapping with the primary category id and the secondary one, which
        is None when the model named none or repeated the primary one

    Raises:
        ValueError: If the primary category is missing or unknown, or the
            secondary one is neither null nor a known category
    """
    primary = raw.get(PRIMARY_FIELD)
    if primary not in CATEGORY_IDS:
        raise ValueError(f"Model response has an unknown category: {primary}")

    secondary = raw.get(SECONDARY_FIELD)
    if secondary is not None and secondary not in CATEGORY_IDS:
        raise ValueError(
            f"Model response has an unknown secondary category: {secondary}"
        )

    return {
        "primary": primary,
        "secondary": None if secondary == primary else secondary,
    }


def normalize_specific_place(raw: dict, location: str) -> str | None:
    """Return the specific place the model recognized, or None.

    Args:
        raw: Parsed JSON payload returned by the model
        location: Place where the whole photo set was taken

    Returns:
        The cleaned place name, or None when the model named none or only
        repeated a part of the given location

    Raises:
        ValueError: If the field is missing or not a string or null
    """
    if SPECIFIC_PLACE_FIELD not in raw:
        raise ValueError("Model response is missing the specific place")

    place = raw[SPECIFIC_PLACE_FIELD]
    if place is None:
        return None
    if not isinstance(place, str):
        raise ValueError(f"Model response has an invalid specific place: {place}")

    place = truncate_text(place, SPECIFIC_PLACE_MAX_CHARS)
    given_parts = {
        part.strip().casefold() for part in location.split(",") if part.strip()
    }
    given_parts.add(" ".join(location.split()).casefold())
    if not place or place.casefold() in given_parts:
        return None

    return place


def normalize_metadata(
    raw: dict,
    profile: TargetProfile,
    location: str | None = None,
) -> dict:
    """Validate and normalize the metadata returned by the model.

    Args:
        raw: Parsed JSON payload returned by the model
        profile: Target profile describing the destination
        location: Place where the whole photo set was taken, if known

    Returns:
        A mapping with the title, description, keywords and category keys,
        plus a location mapping with the given and the detected place when a
        location was given

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

    metadata = {
        "title": truncate_text(title, profile.title_max_chars),
        "description": truncate_text(description, profile.description_max_chars),
        "keywords": keywords,
        "category": normalize_category(raw),
    }

    if location:
        metadata["location"] = {
            "given": location,
            "detected": normalize_specific_place(raw, location),
        }

    return metadata
