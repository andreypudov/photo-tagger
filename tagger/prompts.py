from .image_loader import Photo
from .targets import TargetProfile

SCHEMA_NAME = "photo_metadata"


def build_system_prompt(profile: TargetProfile) -> str:
    """Build the system prompt that defines the voice for a target.

    Args:
        profile: Target profile describing the destination

    Returns:
        The system prompt text
    """
    guidelines = "\n".join(f"- {rule}" for rule in profile.guidelines)

    return (
        f"You are {profile.voice}.\n"
        f"You write metadata for {profile.summary}.\n"
        "You describe only what is actually visible in the photograph and "
        "never speculate about facts the image does not show.\n"
        "\n"
        "Rules for this target:\n"
        f"{guidelines}\n"
        "\n"
        "Constraints:\n"
        f"- The title is at most {profile.title_max_chars} characters.\n"
        f"- The description is at most {profile.description_max_chars} characters.\n"
        f"- Provide between {profile.min_keywords} and {profile.max_keywords} "
        "keywords, all distinct.\n"
        "- Write in English."
    )


def build_user_prompt(photo: Photo, profile: TargetProfile) -> str:
    """Build the per-photo instruction sent alongside the image.

    Args:
        photo: The prepared photo
        profile: Target profile describing the destination

    Returns:
        The user prompt text
    """
    lines = [
        f"Generate a title, a description and keywords for this photograph "
        f"for the {profile.name} target.",
        "",
        "Technical context (use it only when it genuinely informs the text, "
        "never quote it verbatim):",
        f"- File name: {photo.filename}",
        f"- Original size: {photo.width}x{photo.height} px",
    ]

    orientation = describe_orientation(photo.width, photo.height)
    if orientation:
        lines.append(f"- Orientation: {orientation}")

    for name, value in photo.exif.items():
        lines.append(f"- {name}: {value}")

    return "\n".join(lines)


def describe_orientation(width: int, height: int) -> str | None:
    """Return a human readable orientation for the given pixel dimensions."""
    if width <= 0 or height <= 0:
        return None

    ratio = width / height
    if ratio > 1.05:
        return "landscape"
    if ratio < 0.95:
        return "portrait"
    return "square"


def build_response_schema(profile: TargetProfile) -> dict:
    """Build the strict JSON schema requested from the model.

    Args:
        profile: Target profile describing the destination

    Returns:
        A JSON schema definition compatible with structured outputs
    """
    return {
        "name": SCHEMA_NAME,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["title", "description", "keywords"],
            "properties": {
                "title": {
                    "type": "string",
                    "description": (
                        f"Title of at most {profile.title_max_chars} characters"
                    ),
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Description of at most "
                        f"{profile.description_max_chars} characters"
                    ),
                },
                "keywords": {
                    "type": "array",
                    "description": (
                        f"Between {profile.min_keywords} and "
                        f"{profile.max_keywords} distinct keywords"
                    ),
                    "items": {"type": "string"},
                },
            },
        },
    }
