from .categories import CategorySet
from .image_loader import Photo
from .targets import TargetProfile

SCHEMA_NAME = "photo_metadata"


SPECIFIC_PLACE_FIELD = "specific_place"


def build_system_prompt(
    profile: TargetProfile,
    category_sets: tuple[CategorySet, ...] = (),
    location: str | None = None,
) -> str:
    """Build the system prompt that defines the voice for a target.

    Args:
        profile: Target profile describing the destination
        category_sets: Category lists the model has to choose from
        location: Place where the whole photo set was taken, if known

    Returns:
        The system prompt text
    """
    guidelines = "\n".join(f"- {rule}" for rule in profile.guidelines)

    prompt = (
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

    for category_set in category_sets:
        choices = "\n".join(
            f"{code}. {name}" for code, name in category_set.choices.items()
        )
        prompt += (
            "\n\n"
            f"Choose the single {category_set.label} category that best matches "
            "the main subject of the photograph and return its number in "
            f"{category_set.field}:\n"
            f"{choices}"
        )

    if location:
        prompt += "\n\n" + build_location_rules(location)

    return prompt


def build_location_rules(location: str) -> str:
    """Build the prompt section describing the location of the photo set.

    Args:
        location: Place where the whole photo set was taken, as given by the
            photographer

    Returns:
        The prompt text
    """
    return (
        "Location:\n"
        f"Every photograph in this set was taken in {location}. The "
        "photographer confirmed this; treat it as a fact and never contradict "
        "it.\n"
        f"- Set {SPECIFIC_PLACE_FIELD} to the name of a specific landmark, "
        "building, site or natural feature only when it is clearly visible, "
        "unmistakable and lies within this location. Use its common English "
        "name without repeating the given location.\n"
        f"- Set {SPECIFIC_PLACE_FIELD} to null for generic scenes such as a "
        "street, a beach, a café or a close-up, whenever you are not certain, "
        "and when the photograph appears to be from somewhere else.\n"
        "- Name the place in the title only when the place itself is the "
        "subject, such as a landmark, cityscape or landscape. The description "
        "may mention it where it reads naturally.\n"
        "- After the subject keywords, add the specific place when there is "
        "one and the place names of the given location, at most four location "
        "keywords in total."
    )


def build_user_prompt(
    photo: Photo,
    profile: TargetProfile,
    category_sets: tuple[CategorySet, ...] = (),
    location: str | None = None,
) -> str:
    """Build the per-photo instruction sent alongside the image.

    Args:
        photo: The prepared photo
        profile: Target profile describing the destination
        category_sets: Category lists the model has to choose from
        location: Place where the whole photo set was taken, if known

    Returns:
        The user prompt text
    """
    fields = ["a title", "a description", "keywords"]
    if category_sets:
        fields.append("categories")
    if location:
        fields.append("the specific place")
    requested = ", ".join(fields[:-1]) + " and " + fields[-1]

    lines = [
        f"Generate {requested} for this photograph for the {profile.name} target.",
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


def build_response_schema(
    profile: TargetProfile,
    category_sets: tuple[CategorySet, ...] = (),
    location: str | None = None,
) -> dict:
    """Build the strict JSON schema requested from the model.

    Args:
        profile: Target profile describing the destination
        category_sets: Category lists the model has to choose from
        location: Place where the whole photo set was taken, if known

    Returns:
        A JSON schema definition compatible with structured outputs
    """
    properties = {
        "title": {
            "type": "string",
            "description": (f"Title of at most {profile.title_max_chars} characters"),
        },
        "description": {
            "type": "string",
            "description": (
                f"Description of at most {profile.description_max_chars} characters"
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
    }
    required = ["title", "description", "keywords"]

    for category_set in category_sets:
        properties[category_set.field] = {
            "type": "integer",
            "description": (
                f"Number of the {category_set.label} category that best "
                "matches the photo"
            ),
            "enum": list(category_set.choices),
        }
        required.append(category_set.field)

    if location:
        properties[SPECIFIC_PLACE_FIELD] = {
            "type": ["string", "null"],
            "description": (
                "Name of a clearly recognizable landmark, site or natural "
                f"feature within {location}, or null when not certain"
            ),
        }
        required.append(SPECIFIC_PLACE_FIELD)

    return {
        "name": SCHEMA_NAME,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        },
    }
