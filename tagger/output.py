import json
import os
import sys
from datetime import datetime, timezone

from .tagging import PhotoResult
from .targets import TargetProfile

STDOUT_PATH = "-"


def build_document(
    results: list[PhotoResult],
    profile: TargetProfile,
    model: str,
    generated_at: str | None = None,
) -> dict:
    """Assemble the JSON document describing a tagging run.

    Args:
        results: Per photo results, in input order
        profile: Target profile describing the destination
        model: Model identifier used for the run
        generated_at: ISO timestamp, defaults to the current UTC time

    Returns:
        The document ready to be serialized
    """
    photos = []
    errors = []

    for result in results:
        if result.succeeded:
            photo = {
                "file": result.path,
                "filename": os.path.basename(result.path),
                "title": result.metadata["title"],
                "description": result.metadata["description"],
                "keywords": result.metadata["keywords"],
            }
            if "category" in result.metadata:
                photo["category"] = result.metadata["category"]
            photos.append(photo)
        else:
            errors.append({"file": result.path, "error": result.error})

    document = {
        "target": profile.name,
        "model": model,
        "generated_at": generated_at
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "photos": photos,
    }

    if errors:
        document["errors"] = errors

    return document


def write_text(payload: str, path: str) -> None:
    """Write text to a file or to standard output.

    Args:
        payload: Text to write
        path: Destination file, or "-" for standard output

    Raises:
        RuntimeError: If the file cannot be written
    """
    if path == STDOUT_PATH:
        sys.stdout.write(payload)
        return

    directory = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
    except OSError as e:
        raise RuntimeError(f"Unable to write {path}: {e}") from e


def write_document(document: dict, path: str, indent: int = 2) -> None:
    """Serialize the document to a file or to standard output.

    Args:
        document: Document returned by build_document
        path: Destination file, or "-" for standard output
        indent: Number of spaces used for indentation

    Raises:
        RuntimeError: If the file cannot be written
    """
    payload = json.dumps(document, indent=indent, ensure_ascii=False)
    write_text(payload + "\n", path)


def print_summary(
    results: list[PhotoResult], profile: TargetProfile, path: str
) -> None:
    """Print a one line summary of the run to standard error."""
    tagged = sum(1 for result in results if result.succeeded)
    failed = len(results) - tagged
    destination = "standard output" if path == STDOUT_PATH else path

    summary = f"Tagged {tagged} photo(s) for the {profile.name} target"
    if failed:
        summary += f", {failed} failed"

    print(f"{summary} -> {destination}", file=sys.stderr)
