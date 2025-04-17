import logging
import os
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import supervision as sv
import uuid
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.data import build

from .class_balancer import YOLOWeightedDataset


logging.getLogger().handlers.clear()
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


class SupermarketScanner(YOLO):
    def __init__(self, weights: str):
        super().__init__(weights, task="segment")
        build.YOLODataset = YOLOWeightedDataset

        self.pth_output = os.path.join("runs", "segment", "supermarketscanner")
        os.makedirs(self.pth_output, exist_ok=True)

        self.imgsz = 1_024

        self.prices = {
            "blueberry": 20,
            "bread": 10,
            "chicken": 60,
            "egg": 30,
            "juice": 20,
            "melon": 70,
            "sushi": 50,
            "watermelon": 80,
        }

        self.segments = sv.ColorPalette([
            sv.Color(100, 149, 237),  # blueberry: cornflowerblue
            sv.Color(221, 160, 221),  # bread: plum
            sv.Color(255, 255, 255),  # chicken: white
            sv.Color(255, 215, 0),    # egg: gold
            sv.Color(255, 99, 71),    # juice: tomato
            sv.Color(152, 251, 152),  # melon: palegreen
            sv.Color(255, 228, 181),  # sushi: moccasin
            sv.Color(255, 192, 203),  # watermelon: pink
        ])

        self.annotator_label = sv.LabelAnnotator(
            color=self.segments,
            text_color=sv.Color.BLACK,
            text_scale=1,
            text_thickness=2,
            text_padding=5,
            text_position=sv.Position.CENTER,
        )
        self.annotator_mask = sv.MaskAnnotator(color=self.segments)
        self.annotator_polygon = sv.PolygonAnnotator(color=self.segments)

    def _summarise_basket(
            self,
            scene: np.ndarray,
            detections: sv.Detections,
        ) -> tuple[np.ndarray, defaultdict[str, int], defaultdict[str, int]]:
        """Summarise and annotate basket items at the top left corner."""
        amt, qty = defaultdict(int), defaultdict(int)
        for cid in sorted(detections.data["class_name"]):
            amt[cid] += self.prices[cid]
            qty[cid] += 1

        msgs = []
        w_max, h_max = 0, 0

        for (item, unit), price in zip(qty.items(), amt.values()):
            msgs.append(f"- {unit} {item}: HKD {price}")

            (w, h), _ = cv2.getTextSize(
                text=msgs[-1],
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=0.8,
                thickness=1,
            )

            w_max, h_max = max(w_max, w), max(h_max, h)

        lst_item = np.zeros_like(scene, np.uint8)
        cv2.rectangle(
            img=lst_item,
            pt1=(20, 10),
            pt2=(40+w_max, 10+(h_max+10)*len(msgs)),
            color=sv.Color.WHITE.as_bgr(),
            thickness=-1,
        )

        mask = lst_item.astype(bool)
        scene[mask] = cv2.addWeighted(
            src1=scene,
            src2=lst_item,
            alpha=0.1,
            beta=0.9,
            gamma=0,
        )[mask]

        for line, msg in enumerate(msgs):
            cv2.putText(
                img=scene,
                text=msg,
                org=(30, 30+line*30),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=0.8,
                color=sv.Color.BLACK.as_bgr(),
                thickness=1,
            )

        return scene, amt, qty

    def _show_summary(
            self,
            scene: np.ndarray,
            amt: defaultdict[str, int],
            qty: defaultdict[str, int],
        ) -> np.ndarray:
        """Overlay a transaction summary on top of the image."""
        h, w, _ = scene.shape
        x1, y1 = w // 4, h // 4
        x2, y2 = w * 3 // 4, h * 3 // 4

        scene = cv2.GaussianBlur(scene, (21, 21), 0)

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

        msgs = [
            f"Found {sum(qty.values())} items on the desk",
            f"Total HKD {sum(amt.values())} - Thank you",
        ]

        x = np.mean([x1, x2]).astype("int16")
        y = np.mean([y1, y2]).astype("int16")
        margin = (y-y1) // 3

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
                org=(x-w//2, y+margin*(-1)**(idx+1)),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=0.9,
                color=sv.Color.BLACK.as_bgr(),
                thickness=1,
            )

        return scene

    def scan(
            self,
            src: str|np.ndarray,
            summary: bool=False,
            save: bool=False
        ) -> tuple[np.ndarray, defaultdict[str, int]]:
        """Detect items on the current basket."""
        try:
            scene = sv.resize_image(
                image=cv2.imread(src) if isinstance(src, str) else src,
                resolution_wh=(self.imgsz, self.imgsz),
                keep_aspect_ratio=True,
            )

            result = self.predict(
                source=scene,
                imgsz=self.imgsz,
                conf=0.5,
                verbose=False,
            )[0]

            detections = sv.Detections.from_ultralytics(result)

            if not detections.is_empty():
                scene = self.annotator_mask.annotate(scene, detections)
                scene = self.annotator_polygon.annotate(scene, detections)
                scene = self.annotator_label.annotate(scene, detections)

            scene, amt, qty = self._summarise_basket(scene, detections)

            logging.info(
                f"Found {', '.join(f'{v} {k}' for k, v in qty.items())} "
                "items on the basket."
            )

            if summary:
                scene = self._show_summary(scene, amt, qty)

            if save:
                img_name = src.split("/")[-1] if isinstance(src, str) \
                    else f"{uuid.uuid4()}.jpg"
                cv2.imwrite(os.path.join(self.pth_output, img_name), scene)

                logging.info(f"Results saved to \033[1m{self.pth_output}\033[0m")
        except:
            scene, qty = None, defaultdict(int)

            logging.error(f"Error in scanning.", exc_info=True)

        return scene, qty

    def checkout(self, src: str|None=None) -> None:
        """Detect items placed on the desk during checkout."""
        cam = cv2.VideoCapture(0) if src is None else cv2.VideoCapture(src)
        if not cam.isOpened():
            logging.error("Failed to open camera or video source.")
            return

        width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        width, height = self.imgsz, int(self.imgsz / width * height)

        output_video = cv2.VideoWriter(
            filename=os.path.join(
                self.pth_output,
                f"transaction-{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4" \
                    if src is None else src.split("/")[-1],
            ),
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=20 if src is None else cam.get(cv2.CAP_PROP_FPS),
            frameSize=(width, height),
        )

        is_smy = False  # show summary if the number of products remains unchanged
        hist = [{}] * 24 * 3  # a queue stores history of 3-second frames

        if src is None:
            while True:
                flag, frame = cam.read()
                if not flag:
                    logging.warning("Failed to connect to camera.")

                frame, qty = self.scan(frame, is_smy)
                hist.pop(0)
                hist.append(qty)
                is_smy = bool(hist[0]) and hist[:30] == hist[-30:]

                frame = cv2.resize(frame, (width, height))
                output_video.write(frame)
                cv2.imshow("Camera", frame)

                if cv2.waitKey(1) == ord('q'):
                    break
        else:
            total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
            for _ in tqdm(range(total_frames)):
                flag, frame = cam.read()
                if flag:
                    frame, qty = self.scan(frame, is_smy)
                    hist.pop(0)
                    hist.append(qty)
                    is_smy = bool(hist[0]) and hist[:30] == hist[-30:]

                    frame = cv2.resize(frame, (width, height))
                    output_video.write(frame)

        cam.release()
        output_video.release()
        cv2.destroyAllWindows()

        logging.info(f"Results saved to \033[1m{self.pth_output}\033[0m")
