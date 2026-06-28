from pathlib import Path

import yaml

BASE_DIR = Path(__file__).parent.parent.parent


def _load_catalog() -> tuple[dict[str, int], tuple[str, ...], list[list[int]]]:
    with open(
        BASE_DIR / "config" / "products.yaml",
        "r",
        encoding="utf-8",
    ) as file:
        catalog = yaml.safe_load(file)

    map_price = catalog["prices"]
    class_names = tuple(map_price)
    map_colour = catalog["colours"]

    missing_colours = sorted(set(class_names) - set(map_colour))
    extra_colours = sorted(set(map_colour) - set(class_names))
    if missing_colours or extra_colours:
        raise ValueError(
            "Product catalog colour keys do not match price keys: "
            f"missing={missing_colours}, extra={extra_colours}"
        )

    colours = [map_colour[name] for name in class_names]
    return map_price, class_names, colours


PRICES, CLASS_NAMES, COLOURS = _load_catalog()


class ProductCatalog:
    CLASS_NAMES = CLASS_NAMES
    COLOURS = COLOURS
    PRICES = PRICES
