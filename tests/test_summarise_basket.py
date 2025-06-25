import os

import numpy as np
import pytest
import supervision as sv

from supermarketscanner import SupermarketScanner


@pytest.fixture
def model():
    return SupermarketScanner(os.path.join("models", "fine_tune.onnx"))


@pytest.mark.parametrize(
    "cnt, expected_amt, expected_qty",
    [
        ((1, 1, 1, 1, 1, 1, 1, 1), 340, 8),
        ((0, 1, 0, 0, 1, 0, 0, 0), 30, 2),
        ((0, 0, 0, 0, 1, 0, 1, 0), 70, 2),
        ((0, 0, 0, 0, 5, 0, 0, 0), 100, 5),
        ((2, 0, 1, 0, 0, 1, 1, 1), 300, 6),
    ]
)
def test_summarise_basket(model, cnt, expected_amt, expected_qty):
    class_name = []
    for item, freq in zip(model.prices, cnt):
        for _ in range(freq):
            class_name.append(item)

    detections = sv.Detections.empty()
    detections.data["class_name"] = class_name

    original = np.zeros((model.imgsz, model.imgsz, 3), dtype=np.uint8)
    amended, amt, qty = model._summarise_basket(original.copy(), detections)

    assert not np.array_equal(original, amended)
    assert sum(amt.values()) == expected_amt
    assert sum(qty.values()) == expected_qty
