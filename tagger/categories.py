from dataclasses import dataclass

PRIMARY_FIELD = "primary_category"
SECONDARY_FIELD = "secondary_category"


@dataclass(frozen=True)
class Category:
    """One entry of the photo-tagger category list.

    The list is a superset of the categories stock sites use: every category
    maps to exactly one category of each site, so a tagged photo can be
    exported to any site without asking the model again. Mapping to a site
    is the job of the export format.

    Attributes:
        id: Stable identifier stored in the JSON document
        description: What the category covers, shown to the model
    """

    id: str
    description: str


CATEGORIES = (
    Category("wildlife", "Wild animals, birds, insects and marine life"),
    Category("pets", "Dogs, cats and other domestic or farm animals"),
    Category("landmark", "Famous buildings, monuments and sights"),
    Category("architecture", "Building exteriors and architectural details"),
    Category("interior", "Rooms and interior design"),
    Category("cityscape", "Skylines, streets and urban views"),
    Category("landscape", "Mountains, fields, countryside, lakes and rivers"),
    Category("seascape", "Coasts, beaches and the sea"),
    Category("sky_weather", "Sky, clouds, sunsets, sunrises and weather"),
    Category("environment", "Ecology, pollution, climate change and renewable energy"),
    Category("plants_flowers", "Plants, flowers and trees as the subject"),
    Category("food", "Dishes, ingredients and cooking"),
    Category("drinks", "Beverages such as coffee, tea, wine and cocktails"),
    Category("people", "Portraits and people as the subject"),
    Category("lifestyle", "Everyday life, home, family and leisure scenes"),
    Category("business", "Office, work, finance and commerce"),
    Category("industry", "Factories, construction, mining and power plants"),
    Category("agriculture", "Farming, crops, orchards and harvest"),
    Category("technology", "Devices, computers, electronics and digital life"),
    Category("science_health", "Science, medicine, laboratories and healthcare"),
    Category("sports", "Sports, fitness and athletes"),
    Category("hobbies", "Crafts, music, games and outdoor recreation"),
    Category("travel", "Tourism, travellers, luggage and sightseeing"),
    Category("transport", "Vehicles, roads, trains, boats and aircraft"),
    Category("culture_religion", "Traditions, religion, rituals, art and heritage"),
    Category("celebrations", "Holidays, parties, festivals and events"),
    Category("social_issues", "Poverty, protest, inequality and other issues"),
    Category("emotions_concepts", "Moods, feelings and abstract ideas as the subject"),
    Category(
        "backgrounds_textures",
        "Textures, patterns, abstract images and copy space backgrounds",
    ),
)

CATEGORY_IDS = tuple(category.id for category in CATEGORIES)

# Rules for photos that fit more than one category, so that similar photos
# of an album end up in the same category.
CATEGORY_RULES = (
    "A recognizable landmark is landmark, even when it is also architecture "
    "or a travel subject.",
    "Use people when a person is the subject and lifestyle when the activity "
    "or scene matters more than who is in it.",
    "Use travel only when tourism itself is shown, such as travellers, "
    "luggage or sightseeing; a place on its own is landscape, cityscape or "
    "landmark.",
    "Religious buildings are landmark or architecture; culture_religion is "
    "for ceremonies, rituals, traditions and art.",
    "Use backgrounds_textures when there is no distinct subject.",
    "Use emotions_concepts only when a feeling or an idea is clearly the "
    "point of the photograph.",
)
