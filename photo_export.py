import argparse
import os
import sys

from tagger.export import (
    ADOBE_STOCK,
    FORMAT_NAMES,
    FORMATS,
    find_category,
    find_duplicate_filenames,
    get_format,
    load_document,
)
from tagger.output import STDOUT_PATH, write_text

__version__ = "0.1.0"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert a JSON document written by photo_tagger.py into "
        "the metadata file a destination expects, without calling the API again.",
        epilog="Examples:\n"
        "  %(prog)s tags.json\n"
        "  %(prog)s --format adobe-stock tags.json -o adobe.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        metavar="TAGS_JSON",
        help="JSON document written by photo_tagger.py",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=FORMAT_NAMES,
        default=ADOBE_STOCK,
        help="Format to write: "
        + ", ".join(f"'{name}' for the {fmt.summary}" for name, fmt in FORMATS.items())
        + " (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help=f"File to write, or '{STDOUT_PATH}' for standard output "
        "(default: next to the input, e.g. tags.adobe-stock.csv)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Do not print warnings and the summary",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def default_output_path(input_path: str, export_format) -> str:
    """Return the output path derived from the input path and the format.

    Args:
        input_path: Path to the tagging JSON document
        export_format: Format being written

    Returns:
        For tags.json and adobe-stock, tags.adobe-stock.csv in the same folder
    """
    root = os.path.splitext(input_path)[0]
    return f"{root}.{export_format.name}{export_format.extension}"


def collect_warnings(document: dict, export_format) -> list[str]:
    """Describe what the export leaves out or leaves incomplete.

    Args:
        document: Tagging document being exported
        export_format: Format being written

    Returns:
        Human readable warnings, empty when the export is complete
    """
    warnings = []

    failed = document.get("errors") or []
    if failed:
        warnings.append(
            f"{len(failed)} photo(s) failed tagging and are not in the document"
        )

    category_set = export_format.category_set
    if category_set is not None:
        uncategorized = [
            photo["filename"]
            for photo in document["photos"]
            if find_category(photo, category_set) is None
        ]
        if uncategorized:
            warnings.append(
                f"{len(uncategorized)} photo(s) have no {category_set.label} "
                f"category, the column is left empty: {', '.join(uncategorized)}"
            )

    return warnings


def run(args) -> int:
    """Execute an export for already parsed arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        The process exit code
    """
    export_format = get_format(args.format)
    document = load_document(args.input)
    photos = document["photos"]

    if export_format.unique_filenames:
        duplicates = find_duplicate_filenames(photos)
        if duplicates:
            raise ValueError(
                f"The {export_format.name} format identifies photos by file name "
                f"only, but these names are used more than once: "
                f"{', '.join(duplicates)}"
            )

    output = args.output or default_output_path(args.input, export_format)
    write_text(export_format.render(photos), output)

    if not args.quiet:
        for warning in collect_warnings(document, export_format):
            print(f"Warning: {warning}", file=sys.stderr)

        destination = "standard output" if output == STDOUT_PATH else output
        print(
            f"Exported {len(photos)} photo(s) as {export_format.name} -> {destination}",
            file=sys.stderr,
        )

    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return run(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
