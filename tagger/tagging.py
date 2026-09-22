import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .errors import FatalError
from .image_loader import DEFAULT_MAX_DIMENSION, SUPPORTED_EXTENSIONS, load_photo
from .targets import TargetProfile

DEFAULT_JOBS = 1


@dataclass
class PhotoResult:
    """Outcome of tagging a single photo.

    Attributes:
        path: Path to the original file
        metadata: Generated metadata, empty when the photo failed
        error: Error message, empty when the photo succeeded
    """

    path: str
    metadata: dict = field(default_factory=dict)
    error: str = ""

    @property
    def succeeded(self) -> bool:
        """Return True when metadata was generated for this photo."""
        return not self.error


def collect_photo_paths(paths: list[str]) -> list[str]:
    """Expand the command line arguments into a list of file paths.

    Directories are expanded one level deep so that a shoot folder can be
    passed directly. The resulting order is stable and free of duplicates.

    Args:
        paths: Files and directories given on the command line

    Returns:
        The expanded list of file paths

    Raises:
        ValueError: If no paths were given
    """
    if not paths:
        raise ValueError("No photos given")

    collected: list[str] = []
    seen: set[str] = set()

    for path in paths:
        candidates = [path]
        if os.path.isdir(path):
            candidates = sorted(
                os.path.join(path, name)
                for name in os.listdir(path)
                if name.lower().endswith(SUPPORTED_EXTENSIONS)
                and os.path.isfile(os.path.join(path, name))
            )

        for candidate in candidates:
            key = os.path.abspath(candidate)
            if key in seen:
                continue
            seen.add(key)
            collected.append(candidate)

    if not collected:
        raise ValueError("No supported image files found in the given paths")

    return collected


def tag_photo(
    path: str,
    profile: TargetProfile,
    generator,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    include_exif: bool = True,
) -> PhotoResult:
    """Generate metadata for a single photo, capturing failures.

    Args:
        path: Path to the image file
        profile: Target profile describing the destination
        generator: Object exposing a generate(photo, profile) method
        max_dimension: Longest edge of the image sent to the model
        include_exif: Whether EXIF metadata should be forwarded

    Returns:
        A PhotoResult holding either the metadata or the error message

    Raises:
        FatalError: If the failure would repeat for every other photo
    """
    try:
        photo = load_photo(path, max_dimension=max_dimension, include_exif=include_exif)
        metadata = generator.generate(photo, profile)
    except FatalError:
        raise
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        return PhotoResult(path=path, error=str(e))

    return PhotoResult(path=path, metadata=metadata)


def tag_photos(
    paths: list[str],
    profile: TargetProfile,
    generator,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    include_exif: bool = True,
    on_progress: Callable[[int, int, PhotoResult], None] | None = None,
    jobs: int = DEFAULT_JOBS,
) -> list[PhotoResult]:
    """Generate metadata for every photo in the list.

    A failing photo never aborts the run: its error is recorded and the
    remaining photos are still processed. Only a FatalError, which would
    repeat for every photo, stops the run.

    Args:
        paths: Paths to the image files
        profile: Target profile describing the destination
        generator: Object exposing a thread safe generate(photo, profile) method
        max_dimension: Longest edge of the image sent to the model
        include_exif: Whether EXIF metadata should be forwarded
        on_progress: Optional callback invoked as (completed, total, result)
        jobs: Number of photos processed concurrently

    Returns:
        One PhotoResult per input path, in input order

    Raises:
        FatalError: If a failure would repeat for every other photo
    """
    total = len(paths)

    def process(path: str) -> PhotoResult:
        return tag_photo(
            path,
            profile,
            generator,
            max_dimension=max_dimension,
            include_exif=include_exif,
        )

    def report(completed: int, result: PhotoResult) -> None:
        if on_progress is not None:
            on_progress(completed, total, result)

    if jobs <= 1 or total <= 1:
        results: list[PhotoResult] = []
        for path in paths:
            results.append(process(path))
            report(len(results), results[-1])
        return results

    # Queued photos are cancelled on a fatal error or Ctrl-C; only the
    # requests already in flight are waited for.
    executor = ThreadPoolExecutor(max_workers=jobs)
    try:
        futures = {
            executor.submit(process, path): index for index, path in enumerate(paths)
        }
        ordered: list[PhotoResult] = [None] * total
        completed = 0

        for future in as_completed(futures):
            result = future.result()
            completed += 1
            ordered[futures[future]] = result
            report(completed, result)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    return ordered
