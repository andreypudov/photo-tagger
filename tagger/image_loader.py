import base64
import os
from dataclasses import dataclass, field
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.ExifTags import GPSTAGS, TAGS

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".bmp",
)

DEFAULT_MAX_DIMENSION = 1024
JPEG_QUALITY = 85

_EXIF_FIELDS = (
    "Make",
    "Model",
    "LensModel",
    "DateTimeOriginal",
    "FNumber",
    "ExposureTime",
    "ISOSpeedRatings",
    "FocalLength",
    "ImageDescription",
)


@dataclass(frozen=True)
class Photo:
    """A photo prepared for submission to the vision model.

    Attributes:
        path: Path to the original file on disk
        data_url: Downscaled JPEG encoded as a base64 data URL
        width: Width of the original image in pixels, as displayed
        height: Height of the original image in pixels, as displayed
        exif: Selected EXIF values, empty when unavailable or disabled
    """

    path: str
    data_url: str
    width: int
    height: int
    exif: dict = field(default_factory=dict)

    @property
    def filename(self) -> str:
        """Return the base name of the original file."""
        return os.path.basename(self.path)


def _check_supported_extension(path: str) -> None:
    """Validate that the file extension is a supported image format.

    Args:
        path: Path to the image file

    Raises:
        ValueError: If the extension is not supported
    """
    extension = os.path.splitext(path)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise ValueError(
            f"Unsupported image format: {path}. Supported formats: {supported}"
        )


def _format_exif_value(name: str, value: object) -> str | None:
    """Convert a raw EXIF value into a short human readable string."""
    if value is None:
        return None

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")

    if name == "FNumber":
        try:
            return f"f/{float(value):g}"
        except (TypeError, ValueError):
            return None

    if name == "ExposureTime":
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return None
        if seconds <= 0:
            return None
        return f"{seconds:g}s" if seconds >= 1 else f"1/{round(1 / seconds)}s"

    if name == "FocalLength":
        try:
            return f"{float(value):g}mm"
        except (TypeError, ValueError):
            return None

    if name == "ISOSpeedRatings":
        if isinstance(value, (tuple, list)):
            value = value[0] if value else None
        try:
            return f"ISO {int(value)}"
        except (TypeError, ValueError):
            return None

    text = str(value).strip().replace("\x00", "")
    return text or None


def _extract_gps_coordinates(exif: dict) -> str | None:
    """Return decimal GPS coordinates from an EXIF GPS block, if present."""
    gps_ifd = exif.get("GPSInfo")
    if not isinstance(gps_ifd, dict):
        return None

    gps = {GPSTAGS.get(key, key): value for key, value in gps_ifd.items()}

    def to_degrees(values, reference) -> float | None:
        try:
            degrees, minutes, seconds = (float(part) for part in values)
        except (TypeError, ValueError):
            return None
        decimal = degrees + minutes / 60 + seconds / 3600
        if str(reference).upper() in ("S", "W"):
            decimal = -decimal
        return decimal

    latitude = to_degrees(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
    longitude = to_degrees(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))

    if latitude is None or longitude is None:
        return None

    return f"{latitude:.5f}, {longitude:.5f}"


def extract_exif(image: Image.Image) -> dict:
    """Extract a small, readable subset of EXIF metadata from an image.

    Args:
        image: Opened Pillow image

    Returns:
        Mapping of EXIF field names to formatted values. Empty when the image
        carries no usable EXIF data.
    """
    try:
        raw = image.getexif()
    except Exception:
        return {}

    if not raw:
        return {}

    named: dict = {}
    for tag_id, value in raw.items():
        named[TAGS.get(tag_id, tag_id)] = value

    try:
        from PIL.ExifTags import IFD

        named.update(
            {
                TAGS.get(tag_id, tag_id): value
                for tag_id, value in raw.get_ifd(IFD.Exif).items()
            }
        )
        named["GPSInfo"] = dict(raw.get_ifd(IFD.GPSInfo))
    except Exception:
        pass

    metadata: dict = {}
    for name in _EXIF_FIELDS:
        formatted = _format_exif_value(name, named.get(name))
        if formatted:
            metadata[name] = formatted

    coordinates = _extract_gps_coordinates(named)
    if coordinates:
        metadata["GPSCoordinates"] = coordinates

    return metadata


def _encode_data_url(image: Image.Image, max_dimension: int) -> str:
    """Downscale an image and encode it as a base64 JPEG data URL.

    Args:
        image: Opened Pillow image
        max_dimension: Longest edge of the encoded image in pixels

    Returns:
        A data URL string usable as an OpenAI image input
    """
    prepared = image.copy()
    if prepared.mode not in ("RGB", "L"):
        prepared = prepared.convert("RGB")

    if max_dimension > 0:
        prepared.thumbnail((max_dimension, max_dimension), Image.LANCZOS)

    buffer = BytesIO()
    prepared.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return f"data:image/jpeg;base64,{encoded}"


def load_photo(
    path: str,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    include_exif: bool = True,
) -> Photo:
    """Load an image from disk and prepare it for the vision model.

    Args:
        path: Path to the image file
        max_dimension: Longest edge of the downscaled image in pixels
        include_exif: Whether EXIF metadata should be read and forwarded

    Returns:
        A Photo carrying the encoded image and its metadata

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the path is not a file, the format is unsupported or
            the image cannot be decoded
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Photo not found: {path}")

    if not os.path.isfile(path):
        raise ValueError(f"Path is not a file: {path}")

    _check_supported_extension(path)

    try:
        with Image.open(path) as image:
            image.load()
            exif = extract_exif(image) if include_exif else {}
            # Camera and phone files are often stored sideways with an EXIF
            # Orientation tag; send the image the way a viewer would see it.
            upright = ImageOps.exif_transpose(image)
            width, height = upright.size
            data_url = _encode_data_url(upright, max_dimension)
    except UnidentifiedImageError as e:
        raise ValueError(f"Unable to decode image {path}: not a readable image") from e
    except Image.DecompressionBombError as e:
        raise ValueError(f"Image {path} is too large to process: {e}") from e
    except OSError as e:
        raise ValueError(f"Unable to read image {path}: {e}") from e

    return Photo(
        path=path,
        data_url=data_url,
        width=width,
        height=height,
        exif=exif,
    )
