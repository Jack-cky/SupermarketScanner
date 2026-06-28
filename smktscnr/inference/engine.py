from pathlib import Path


class InferenceEngine:
    def __init__(self, weights: str):
        self.model = self._load_model(weights)

    def _load_model(self, weights: str):
        match Path(weights).suffix:
            case ".onnx" | ".pt":
                from .torch import TorchModel

                return TorchModel(weights)
            case ".hef":
                from .hailo import HailoModel

                return HailoModel(weights)
            case _:
                raise ValueError(f"Unsupported model format: {weights}")

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)
