import argparse
import sys

from tagger.export import (
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
        description="Convert a JSON document written by photo_tagger.py into the\n"
        "metadata file a destination expects, without calling the API again.",
        epilog="Examples:\n"
        "  %(prog)s -f adobe-stock -o madeira.csv madeira.json\n"
        "  %(prog)s -f adobe-stock -o - madeira.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        metavar="TAGS_JSON",
        help="JSON document written by photo_tagger.py",
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-f",
        "--format",
        choices=FORMAT_NAMES,
        required=True,
        help="Format to write: "
        + ", ".join(f"'{name}' for the {fmt.summary}" for name, fmt in FORMATS.items()),
    )
    required.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        required=True,
        help=f"File to write, or '{STDOUT_PATH}' for standard output; "
        "an existing file is replaced",
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

    if export_format.category_map is not None:
        uncategorized = [
            photo["filename"]
            for photo in document["photos"]
            if find_category(photo) is None
        ]
        if uncategorized:
            warnings.append(
                f"{len(uncategorized)} photo(s) have no category, the "
                f"{export_format.name} category is left empty: "
                f"{', '.join(uncategorized)}"
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

    output = args.output
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
