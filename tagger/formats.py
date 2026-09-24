import csv
import io
import os
from collections.abc import Callable
from dataclasses import dataclass, field

from .metadata import normalize_keywords, truncate_text
from .output import build_document, write_document, write_text
from .tagging import PhotoResult
from .targets import TargetProfile

JSON = "json"
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

ADOBE_STOCK_COLUMNS = ("Filename", "Title", "Keywords", "Category", "Releases")
ADOBE_STOCK_TITLE_MAX_CHARS = 200
ADOBE_STOCK_MAX_KEYWORDS = 49


@dataclass(frozen=True)
class OutputFormat:
    """Describes a file format the tagging results can be written to.

    Attributes:
        name: Identifier used by the --format command line flag
        summary: Short human readable description of the format
        extension: File extension of the default output file
        write: Callable writing (results, profile, model, path, indent)
        categories: Category codes the model has to choose from, empty when
            the format has no category column
        unique_filenames: Whether rows are matched to uploads by file name
            alone, so two photos may not share one
    """

    name: str
    summary: str
    extension: str
    write: Callable[[list[PhotoResult], TargetProfile, str, str, int], None]
    categories: dict[int, str] = field(default_factory=dict)
    unique_filenames: bool = False

    @property
    def default_output(self) -> str:
        """Return the file name used when no --output is given."""
        return f"tags{self.extension}"


def write_json(
    results: list[PhotoResult],
    profile: TargetProfile,
    model: str,
    path: str,
    indent: int = 2,
) -> None:
    """Write the results as the photo-tagger JSON document."""
    write_document(build_document(results, profile, model), path, indent=indent)


def build_adobe_stock_row(result: PhotoResult) -> dict:
    """Build one Adobe Stock CSV row for a successfully tagged photo.

    Args:
        result: A successful photo result

    Returns:
        Mapping of the Adobe Stock column names to cell values
    """
    metadata = result.metadata
    # Adobe Stock splits the cell on commas, so a comma inside a keyword
    # would silently turn it into two. Replacing it can produce a duplicate,
    # so the list is deduplicated and capped only afterwards.
    keywords = normalize_keywords(
        [keyword.replace(",", " ") for keyword in metadata["keywords"]],
        ADOBE_STOCK_MAX_KEYWORDS,
    )

    return {
        "Filename": os.path.basename(result.path),
        "Title": truncate_text(metadata["title"], ADOBE_STOCK_TITLE_MAX_CHARS),
        "Keywords": ", ".join(keywords),
        "Category": metadata.get("category", ""),
        "Releases": "",
    }


def write_adobe_stock_csv(
    results: list[PhotoResult],
    profile: TargetProfile,
    model: str,
    path: str,
    indent: int = 2,
) -> None:
    """Write the results as an Adobe Stock metadata upload CSV.

    Photos that failed are left out; they are reported on standard error.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=ADOBE_STOCK_COLUMNS)
    writer.writeheader()
    for result in results:
        if result.succeeded:
            writer.writerow(build_adobe_stock_row(result))

    write_text(buffer.getvalue(), path)


FORMATS = {
    JSON: OutputFormat(
        name=JSON,
        summary="photo-tagger JSON document",
        extension=".json",
        write=write_json,
    ),
    ADOBE_STOCK: OutputFormat(
        name=ADOBE_STOCK,
        summary="Adobe Stock metadata upload CSV",
        extension=".csv",
        write=write_adobe_stock_csv,
        categories=ADOBE_STOCK_CATEGORIES,
        unique_filenames=True,
    ),
}

FORMAT_NAMES = tuple(FORMATS)


def get_format(name: str) -> OutputFormat:
    """Return the output format registered under the given name.

    Args:
        name: Format identifier, for example "json" or "adobe-stock"

    Returns:
        The matching OutputFormat

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


def check_paths(output_format: OutputFormat, paths: list[str]) -> None:
    """Reject input the output format cannot represent, before any API call.

    Args:
        output_format: Format the results will be written in
        paths: Photos that are about to be tagged

    Raises:
        ValueError: If the format needs unique file names and two photos
            share one
    """
    if not output_format.unique_filenames:
        return

    seen: dict[str, str] = {}
    for path in paths:
        name = os.path.basename(path)
        if name in seen:
            raise ValueError(
                f"{seen[name]} and {path} share the file name {name}; the "
                f"{output_format.name} format identifies photos by file name only"
            )
        seen[name] = path
