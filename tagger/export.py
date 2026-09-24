import csv
import io
import json
from collections.abc import Callable
from dataclasses import dataclass

from .categories import CATEGORY_IDS
from .metadata import normalize_keywords, truncate_text

ADOBE_STOCK = "adobe-stock"

ADOBE_STOCK_CATEGORIES = {
    1: "Animals",
    2: "Buildings and Architecture",
    3: "Business",
    4: "Drinks",
    5: "The Environment",
    6: "States of Mind",
    7: "Food",
    8: "Graphic Resources",
    9: "Hobbies and Leisure",
    10: "Industry",
    11: "Landscapes",
    12: "Lifestyle",
    13: "People",
    14: "Plants and Flowers",
    15: "Culture and Religion",
    16: "Science",
    17: "Social Issues",
    18: "Sports",
    19: "Technology",
    20: "Transport",
    21: "Travel",
}

# Every photo-tagger category maps to exactly one Adobe Stock category code.
ADOBE_STOCK_CATEGORY_MAP = {
    "wildlife": 1,
    "pets": 1,
    "landmark": 2,
    "architecture": 2,
    "interior": 2,
    "cityscape": 2,
    "landscape": 11,
    "seascape": 11,
    "sky_weather": 11,
    "environment": 5,
    "plants_flowers": 14,
    "food": 7,
    "drinks": 4,
    "people": 13,
    "lifestyle": 12,
    "business": 3,
    "industry": 10,
    "agriculture": 10,
    "technology": 19,
    "science_health": 16,
    "sports": 18,
    "hobbies": 9,
    "travel": 21,
    "transport": 20,
    "culture_religion": 15,
    "celebrations": 15,
    "social_issues": 17,
    "emotions_concepts": 6,
    "backgrounds_textures": 8,
}

ADOBE_STOCK_COLUMNS = ("Filename", "Title", "Keywords", "Category", "Releases")
ADOBE_STOCK_TITLE_MAX_CHARS = 200
ADOBE_STOCK_MAX_KEYWORDS = 49

_REQUIRED_TEXT_FIELDS = ("filename", "title")


@dataclass(frozen=True)
class ExportFormat:
    """Describes a file format a tagging JSON document can be converted to.

    Attributes:
        name: Identifier used by the --format command line flag
        summary: Short human readable description of the format
        extension: File extension of the default output file
        render: Callable turning the list of photos into the file contents
        category_map: Code of the destination category for every
            photo-tagger category, None when the format has no category
        unique_filenames: Whether the destination matches entries to files
            by file name alone, so two photos may not share one
    """

    name: str
    summary: str
    extension: str
    render: Callable[[list[dict]], str]
    category_map: dict[str, int] | None = None
    unique_filenames: bool = False


def load_document(path: str) -> dict:
    """Read and validate a JSON document written by photo_tagger.py.

    Args:
        path: Path to the JSON document

    Returns:
        The parsed document

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is not a photo-tagger document
    """
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(f"Tags file not found: {path}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from e
    except (OSError, UnicodeDecodeError) as e:
        raise ValueError(f"Unable to read {path}: {e}") from e

    photos = document.get("photos") if isinstance(document, dict) else None
    if not isinstance(photos, list):
        raise ValueError(f"{path} is not a photo-tagger document: no photos list")

    for index, photo in enumerate(photos):
        if not isinstance(photo, dict):
            raise ValueError(f"{path}: photo #{index + 1} is not an object")
        for name in _REQUIRED_TEXT_FIELDS:
            value = photo.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{path}: photo #{index + 1} has no {name}")
        if not isinstance(photo.get("keywords"), list):
            raise ValueError(f"{path}: photo #{index + 1} has no keywords list")

    return document


def find_category(photo: dict) -> str | None:
    """Return the photo's primary category id, or None when absent."""
    category = photo.get("category")
    if not isinstance(category, dict):
        return None

    primary = category.get("primary")
    return primary if primary in CATEGORY_IDS else None


def find_duplicate_filenames(photos: list[dict]) -> list[str]:
    """Return every file name used by more than one photo, in input order."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for photo in photos:
        name = photo["filename"]
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    return duplicates


def build_adobe_stock_row(photo: dict) -> dict:
    """Build one Adobe Stock CSV row from a photo of the tagging document.

    Args:
        photo: Photo entry of the tagging JSON document

    Returns:
        Mapping of the Adobe Stock column names to cell values
    """
    # Adobe Stock splits the cell on commas, so a comma inside a keyword
    # would silently turn it into two. Replacing it can produce a duplicate,
    # so the list is deduplicated and capped only afterwards.
    keywords = normalize_keywords(
        [
            keyword.replace(",", " ")
            for keyword in photo["keywords"]
            if isinstance(keyword, str)
        ],
        ADOBE_STOCK_MAX_KEYWORDS,
    )
    category = find_category(photo)

    return {
        "Filename": photo["filename"],
        "Title": truncate_text(photo["title"], ADOBE_STOCK_TITLE_MAX_CHARS),
        "Keywords": ", ".join(keywords),
        "Category": ADOBE_STOCK_CATEGORY_MAP.get(category, ""),
        "Releases": "",
    }


def render_adobe_stock_csv(photos: list[dict]) -> str:
    """Render photos as an Adobe Stock metadata upload CSV.

    Args:
        photos: Photo entries of the tagging JSON document

    Returns:
        The CSV text, header included
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=ADOBE_STOCK_COLUMNS)
    writer.writeheader()
    for photo in photos:
        writer.writerow(build_adobe_stock_row(photo))
    return buffer.getvalue()


FORMATS = {
    ADOBE_STOCK: ExportFormat(
        name=ADOBE_STOCK,
        summary="Adobe Stock metadata upload CSV",
        extension=".csv",
        render=render_adobe_stock_csv,
        category_map=ADOBE_STOCK_CATEGORY_MAP,
        unique_filenames=True,
    ),
}

FORMAT_NAMES = tuple(FORMATS)


def get_format(name: str) -> ExportFormat:
    """Return the export format registered under the given name.

    Args:
        name: Format identifier, for example "adobe-stock"

    Returns:
        The matching ExportFormat

    Raises:
        ValueError: If the format is unknown
    """
    try:
        return FORMATS[name]
    except KeyError:
        known = ", ".join(FORMAT_NAMES)
        raise ValueError(
            f"Unknown format: {name}. Available formats: {known}"
        ) from None
