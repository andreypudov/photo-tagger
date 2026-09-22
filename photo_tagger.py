import argparse
import sys

from tagger.image_loader import DEFAULT_MAX_DIMENSION
from tagger.openai_client import (
    DEFAULT_DETAIL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    DETAIL_LEVELS,
    MODEL_VARIABLE,
    MetadataGenerator,
    resolve_default_model,
)
from tagger.output import STDOUT_PATH, build_document, print_summary, write_document
from tagger.tagging import DEFAULT_JOBS, collect_photo_paths, tag_photos
from tagger.targets import STOCK, TARGET_NAMES, get_profile

__version__ = "0.1.0"

DEFAULT_OUTPUT = "tags.json"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate titles, descriptions and keywords for photos "
        "using the OpenAI API.",
        epilog="Examples:\n"
        "  %(prog)s photo.jpg\n"
        "  %(prog)s --target gallery portrait.jpg landscape.tif -o labels.json\n"
        "  %(prog)s --target stock ./shoot -o stock.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="Photos to tag, or directories containing photos",
    )
    parser.add_argument(
        "-t",
        "--target",
        choices=TARGET_NAMES,
        default=STOCK,
        help="Metadata style to follow: 'stock' for microstock listings, "
        "'gallery' for museum and exhibition wall labels (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=DEFAULT_OUTPUT,
        help=f"JSON file to write, or '{STDOUT_PATH}' for standard output "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=resolve_default_model(),
        help="Vision capable OpenAI model to use, overrides the "
        f"{MODEL_VARIABLE} environment variable (default: %(default)s)",
    )
    parser.add_argument(
        "--detail",
        choices=DETAIL_LEVELS,
        default=DEFAULT_DETAIL,
        help="Image detail requested from the model; 'low' is faster and "
        "cheaper but reads fine detail less well (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        metavar="KEY",
        default=None,
        help="OpenAI API key (default: the OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=DEFAULT_MAX_DIMENSION,
        metavar="PIXELS",
        help="Longest edge of the image sent to the model (default: %(default)s)",
    )
    parser.add_argument(
        "--no-exif",
        action="store_true",
        help="Do not read EXIF metadata from the photos",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        metavar="COUNT",
        help="Number of photos processed concurrently (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        metavar="SECONDS",
        help="Per request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        metavar="COUNT",
        help="Automatic retries on transient API failures (default: %(default)s)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        metavar="SPACES",
        help="Indentation of the generated JSON (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Do not print progress information",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def validate_args(args) -> None:
    """Validate argument combinations that argparse cannot express.

    Args:
        args: Parsed command line arguments

    Raises:
        ValueError: If an argument value is out of range
    """
    if args.max_dimension < 64:
        raise ValueError("--max-dimension must be at least 64 pixels")

    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")

    if not args.model.strip():
        raise ValueError("--model cannot be empty")

    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")

    if args.retries < 0:
        raise ValueError("--retries cannot be negative")

    if args.indent < 0:
        raise ValueError("--indent cannot be negative")


def build_progress_reporter(quiet: bool):
    """Return a progress callback, or None when running quietly."""
    if quiet:
        return None

    def report(index: int, total: int, result) -> None:
        status = (
            result.metadata["title"] if result.succeeded else f"failed: {result.error}"
        )
        print(f"[{index}/{total}] {result.path}: {status}", file=sys.stderr)

    return report


def run(args) -> int:
    """Execute a tagging run for already parsed arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        The process exit code
    """
    validate_args(args)

    profile = get_profile(args.target)
    paths = collect_photo_paths(args.files)

    generator = MetadataGenerator(
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=args.retries,
        detail=args.detail,
    )

    results = tag_photos(
        paths,
        profile,
        generator,
        max_dimension=args.max_dimension,
        include_exif=not args.no_exif,
        on_progress=build_progress_reporter(args.quiet),
        jobs=args.jobs,
    )

    document = build_document(results, profile, args.model)
    write_document(document, args.output, indent=args.indent)

    if not args.quiet:
        print_summary(document, args.output)

    return 0 if all(result.succeeded for result in results) else 1


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
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
