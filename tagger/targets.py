from dataclasses import dataclass

STOCK = "stock"
GALLERY = "gallery"


@dataclass(frozen=True)
class TargetProfile:
    """Describes how metadata should be written for a specific destination.

    Attributes:
        name: Identifier used by the --target command line flag
        summary: Short human readable description of the destination
        voice: Instruction describing the tone the model has to adopt
        title_max_chars: Hard limit applied to the generated title
        description_max_chars: Hard limit applied to the generated description
        min_keywords: Lower bound requested from the model
        max_keywords: Upper bound enforced on the generated keyword list
        guidelines: Destination specific rules injected into the prompt
    """

    name: str
    summary: str
    voice: str
    title_max_chars: int
    description_max_chars: int
    min_keywords: int
    max_keywords: int
    guidelines: tuple[str, ...]


STOCK_PROFILE = TargetProfile(
    name=STOCK,
    summary="Microstock marketplaces such as Adobe Stock, Shutterstock or Getty",
    voice=(
        "a stock photography metadata specialist who optimises listings for "
        "buyer search queries"
    ),
    title_max_chars=70,
    description_max_chars=200,
    min_keywords=25,
    max_keywords=49,
    guidelines=(
        "Write a literal, descriptive title that names the main subject, the "
        "action and the setting, in that order.",
        "Avoid poetic language, artist names, invented place names and any "
        "wording that a buyer would not type into a search box.",
        "Never invent brands, trademarks, celebrity names or recognisable "
        "private property; describe what is visible in generic terms.",
        "The description repeats the title information and adds context that "
        "helps a buyer judge usage: composition, lighting, mood, copy space.",
        "Keywords are single words or short noun phrases, lower case, ordered "
        "from most to least relevant.",
        "Cover subject, action, setting, colours, lighting, mood, season, "
        "composition and plausible commercial concepts.",
        "Do not use hashtags, punctuation, camera settings or file names in "
        "the keywords.",
    ),
)

GALLERY_PROFILE = TargetProfile(
    name=GALLERY,
    summary="Gallery, museum and exhibition wall labels",
    voice=(
        "a museum curator writing the wall label that accompanies a "
        "photographic print in an exhibition"
    ),
    title_max_chars=60,
    description_max_chars=600,
    min_keywords=8,
    max_keywords=15,
    guidelines=(
        "The title is an artwork title, not a search phrase: evocative, "
        "concise, and usually a noun phrase without an article.",
        "Do not end the title with a full stop and do not describe the "
        "photographic technique in it.",
        "The description is a curatorial wall text of two or three sentences "
        "written in the present tense.",
        "Open with what the viewer sees, then move to composition, light and "
        "material qualities, and close with the theme or reading the work "
        "invites.",
        "Use measured, observational language; avoid sales vocabulary, "
        "superlatives and second person address.",
        "Do not fabricate biographical facts, dates, locations, print sizes "
        "or exhibition history that the image does not support.",
        "Keywords are curatorial and thematic: genre, movement, motif, "
        "subject matter and formal qualities.",
    ),
)

PROFILES = {
    STOCK: STOCK_PROFILE,
    GALLERY: GALLERY_PROFILE,
}

TARGET_NAMES = tuple(PROFILES)


def get_profile(name: str) -> TargetProfile:
    """Return the target profile registered under the given name.

    Args:
        name: Target identifier, for example "stock" or "gallery"

    Returns:
        The matching TargetProfile

    Raises:
        ValueError: If the target is unknown
    """
    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(TARGET_NAMES)
        raise ValueError(
            f"Unknown target: {name}. Available targets: {known}"
        ) from None
