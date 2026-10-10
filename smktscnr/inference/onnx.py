import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv

from ..utils import ProductCatalog
from .ops import nms, sigmoid


class OnnxModel:
    """YOLOv8-seg exported to ONNX (nms=False), run with onnxruntime only."""

    def __init__(self, weights: str):
        self.class_names = list(ProductCatalog.PRICES)
        providers = [
            p
            for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()
        ]
        self.session = ort.InferenceSession(weights, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.model_h, self.model_w = self.session.get_inputs()[0].shape[2:]

        class_count = self.session.get_outputs()[0].shape[1] - 4 - 32
        if class_count != len(self.class_names):
            raise ValueError(
                f"Incompatible ONNX model: expected {len(self.class_names)} "
                f"classes for the supermarket catalog, but `{weights}` "
                f"exposes {class_count}."
            )

    def predict(self, **kwargs) -> sv.Detections:
        source = kwargs.get("source")
        conf = kwargs.get("conf", 0.5)
        iou = kwargs.get("iou", 0.7)

        scene = cv2.imread(source) if isinstance(source, str) else source
        if scene is None:
            raise ValueError(f"Unable to load input source: {source}")

        blob, scale, pad = self._letterbox(scene)
        preds, protos = self.session.run(None, {self.input_name: blob})

        return self._postprocess(
            preds[0].T,
            protos[0],
            scene.shape[:2],
            scale,
            pad,
            conf,
            iou,
        )

    def _letterbox(
        self,
        scene: np.ndarray,
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        h, w = scene.shape[:2]
        scale = min(self.model_h / h, self.model_w / w)
        new_w, new_h = round(w * scale), round(h * scale)
        pad_x = (self.model_w - new_w) // 2
        pad_y = (self.model_h - new_h) // 2

        canvas = np.full((self.model_h, self.model_w, 3), 114, np.uint8)
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = cv2.resize(
            scene,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )
        blob = canvas[..., ::-1].transpose(2, 0, 1)[None]
        return np.ascontiguousarray(blob, np.float32) / 255.0, scale, (
            pad_x,
            pad_y,
        )

    def _postprocess(
        self,
        preds: np.ndarray,
        protos: np.ndarray,
        orig_hw: tuple[int, int],
        scale: float,
        pad: tuple[int, int],
        conf: float,
        iou: float,
    ) -> sv.Detections:
        class_scores = preds[:, 4 : 4 + len(self.class_names)]
        scores = class_scores.max(axis=1)
        keep = scores >= conf
        if not np.any(keep):
            return sv.Detections.empty()

        preds, scores = preds[keep], scores[keep]
        class_ids = class_scores[keep].argmax(axis=1).astype(np.int32)
        cx, cy, bw, bh = preds[:, :4].T
        boxes = np.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), 1)

        keep = nms(boxes, scores, class_ids, iou_threshold=iou)[:300]
        boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
        coeffs = preds[keep, 4 + len(self.class_names) :]

        proto_c, proto_h, proto_w = protos.shape
        masks = sigmoid(coeffs @ protos.reshape(proto_c, -1))
        masks = masks.reshape(-1, proto_h, proto_w)
        masks = np.stack(
            [cv2.resize(m, (self.model_w, self.model_h)) for m in masks]
        )
        ys = np.arange(self.model_h)[None, :, None]
        xs = np.arange(self.model_w)[None, None, :]
        x1, y1, x2, y2 = (boxes[:, i, None, None] for i in range(4))
        masks *= (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)

        orig_h, orig_w = orig_hw
        pad_x, pad_y = pad
        new_h, new_w = round(orig_h * scale), round(orig_w * scale)
        masks = masks[:, pad_y : pad_y + new_h, pad_x : pad_x + new_w]
        masks = np.stack(
            [cv2.resize(m, (orig_w, orig_h)) > 0.5 for m in masks]
        )
        boxes = (boxes - [pad_x, pad_y, pad_x, pad_y]) / scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)

        return sv.Detections(
            xyxy=boxes.astype(np.float32),
            mask=masks,
            confidence=scores.astype(np.float32),
            class_id=class_ids,
            data={
                "class_name": np.array(self.class_names)[class_ids],
            },
        )
