from dataclasses import dataclass

ADOBE_STOCK = "adobe_stock"


@dataclass(frozen=True)
class CategorySet:
    """A fixed list of categories a destination sorts its photos into.

    The model picks one category from every registered set for every photo,
    so the tagging JSON already holds what each export format needs.

    Attributes:
        name: Key under which the chosen code is stored in the JSON document
        label: Human readable name of the destination
        choices: Mapping of the numeric category codes to their names
    """

    name: str
    label: str
    choices: dict[int, str]

    @property
    def field(self) -> str:
        """Return the property name used in the model response schema."""
        return f"{self.name}_category"


ADOBE_STOCK_CATEGORIES = CategorySet(
    name=ADOBE_STOCK,
    label="Adobe Stock",
    choices={
        1: "Animals",
        2: "Buildings and Architecture",
        3: "Business",
        4: "Drinks",
        5: "The Environment",
        6: "States of Mind",
        7: "Food",
        8: "Graphic Resources",
        9: "Hobbies and Leisure",
        10: "Industry",
        11: "Landscapes",
        12: "Lifestyle",
        13: "People",
        14: "Plants and Flowers",
        15: "Culture and Religion",
        16: "Science",
        17: "Social Issues",
        18: "Sports",
        19: "Technology",
        20: "Transport",
        21: "Travel",
    },
)

CATEGORY_SETS = (ADOBE_STOCK_CATEGORIES,)
