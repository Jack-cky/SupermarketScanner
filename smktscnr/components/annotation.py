from collections import Counter

import cv2
import numpy as np
import supervision as sv

from ..utils import ProductCatalog


class Cashier:
    def __init__(self):
        self.prices = ProductCatalog.PRICES
        _colours = sv.ColorPalette([
            sv.Color(*rgb) for rgb in ProductCatalog.COLOURS
        ])

        self.annotator_label = sv.LabelAnnotator(
            color=_colours,
            text_color=sv.Color.BLACK,
            text_scale=1,
            text_thickness=2,
            text_padding=5,
            text_position=sv.Position.CENTER,
        )
        self.annotator_mask = sv.MaskAnnotator(color=_colours)
        self.annotator_polygon = sv.PolygonAnnotator(color=_colours)

    def annotate(self, scene: np.ndarray, detections: sv.Detections) -> None:
        if detections.is_empty():
            return

        self.annotator_mask.annotate(scene, detections)
        self.annotator_polygon.annotate(scene, detections)
        self.annotator_label.annotate(scene, detections)

    def summarise_basket(
        self,
        scene: np.ndarray,
        detections: sv.Detections,
    ) -> tuple[dict[str, int], Counter[str]]:
        """Summarise and annotate basket items at the top left corner."""
        if detections.is_empty():
            return {}, Counter()

        qty = Counter(sorted(detections.data["class_name"]))
        amt = {cid: count * self.prices[cid] for cid, count in qty.items()}

        msgs = []
        w_max, h_max = 0, 0

        for item, unit in qty.items():
            msgs.append(f"- {unit} {item}: HKD {amt[item]}")

            (w, h), _ = cv2.getTextSize(
                text=msgs[-1],
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=0.8,
                thickness=1,
            )

            w_max, h_max = max(w_max, w), max(h_max, h)

        x1, y1 = 20, 10
        x2, y2 = 40 + w_max, 10 + (h_max + 10) * len(msgs)
        roi = scene[y1:y2, x1:x2]

        scene[y1:y2, x1:x2] = cv2.addWeighted(
            src1=roi,
            alpha=0.1,
            src2=np.full(roi.shape, 255, dtype=np.uint8),
            beta=0.9,
            gamma=0,
        )

        for line, msg in enumerate(msgs):
            cv2.putText(
                img=scene,
                text=msg,
                org=(30, 30 + line * 30),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=0.8,
                color=sv.Color.BLACK.as_bgr(),
                thickness=1,
            )

        return amt, qty

    def show_receipt(
        self,
        scene: np.ndarray,
        amt: dict[str, int],
        qty: Counter[str],
    ) -> None:
        """Overlay a transaction summary on top of the image."""
        h, w, _ = scene.shape
        x1, y1 = w // 4, h // 4
        x2, y2 = w * 3 // 4, h * 3 // 4

        x = np.mean([x1, x2]).astype("int16")
        y = np.mean([y1, y2]).astype("int16")
        margin = (y - y1) // 3

        msgs = [
            f"Found {sum(qty.values())} items on the desk",
            f"Total HKD {sum(amt.values())} - Thank you",
        ]

        cv2.GaussianBlur(src=scene, ksize=(21, 21), sigmaX=0, dst=scene)

        cv2.rectangle(
            img=scene,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=(220, 220, 220),
            thickness=-1,
        )

        cv2.rectangle(
            img=scene,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=sv.Color.BLACK.as_bgr(),
            thickness=2,
        )

        for idx, msg in enumerate(msgs):
            (w, h), _ = cv2.getTextSize(
                text=msg,
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=0.9,
                thickness=1,
            )

            cv2.putText(
                img=scene,
                text=msg,
                org=(x - w // 2, y + margin * (-1) ** (idx + 1)),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=0.9,
                color=sv.Color.BLACK.as_bgr(),
                thickness=1,
            )
