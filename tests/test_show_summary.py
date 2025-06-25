import os

import numpy as np
import pytest

from supermarketscanner import SupermarketScanner


@pytest.fixture
def model():
    return SupermarketScanner(os.path.join("models", "fine_tune.onnx"))


@pytest.mark.parametrize(
    "amt, qty",
    [
        (
            {"blueberry": 20, "bread": 10, "chicken": 60, "egg": 30, "juice": 20, "melon": 70, "sushi": 50, "watermelon": 80},
            {"blueberry": 1, "bread": 1, "chicken": 1, "egg": 1, "juice": 1, "melon": 1, "sushi": 1, "watermelon": 1},
        ),
        (
            {"blueberry": 40, "chicken": 60, "melon": 70, "sushi": 50, "watermelon": 80},
            {"blueberry": 2, "chicken": 1, "melon": 1, "sushi": 1, "watermelon": 1},
        ),
    ]
)
def test_summarise_basket(model, amt, qty):
    original = np.zeros((model.imgsz, model.imgsz, 3), dtype=np.uint8)
    amended = model._show_summary(original.copy(), amt, qty)

    assert not np.array_equal(original, amended)
