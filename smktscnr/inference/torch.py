import supervision as sv
from ultralytics import YOLO


class TorchModel(YOLO):
    def __init__(self, weights: str):
        super().__init__(weights, task="segment")

    def predict(self, **kwargs,):
        result = super().predict(**kwargs)[0]

        return sv.Detections.from_ultralytics(result)
