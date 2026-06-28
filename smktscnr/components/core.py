from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import uuid
from tqdm import tqdm

from .annotation import Cashier
from ..inference import InferenceEngine
from ..utils import get_logger

LOGGER = get_logger()


class SupermarketScanner:
    def __init__(self, weights: str, imgsz: int = 1_024):
        self.pth_output = Path("runs", "segment", "supermarketscanner")
        self.pth_output.mkdir(parents=True, exist_ok=True)

        self.imgsz = imgsz

        self.engine = InferenceEngine(weights)
        self.cashier = Cashier()

    def scan(
        self,
        src: str | np.ndarray,
        confidence: float = 0.5,
        summary: bool = False,
        save: bool = False,
    ) -> tuple[np.ndarray, defaultdict[str, int]]:
        """Detect items on the current basket."""
        try:
            scene = sv.resize_image(
                image=cv2.imread(src) if isinstance(src, str) else src,
                resolution_wh=(self.imgsz, self.imgsz),
                keep_aspect_ratio=True,
            )

            detections = self.engine.predict(
                source=scene,
                imgsz=self.imgsz,
                conf=confidence,
                verbose=False,
            )

            self.cashier.annotate(scene, detections)

            amt, qty = self.cashier.summarise_basket(scene, detections)

            LOGGER.info(
                f"Found {', '.join(f'{v} {k}' for k, v in qty.items())} "
                "items on the basket."
            )

            if summary:
                self.cashier.show_receipt(scene, amt, qty)

            if save:
                img_name = (
                    src.split("/")[-1]
                    if isinstance(src, str)
                    else f"{uuid.uuid4()}.jpg"
                )
                cv2.imwrite(str(self.pth_output / img_name), scene)

                LOGGER.info(
                    f"Results saved to \033[1m{self.pth_output}\033[0m"
                )
        except Exception:
            scene, qty = None, defaultdict(int)

            LOGGER.error("Error in scanning.", exc_info=True)

        return scene, qty

    def checkout(self, src: str | None = None) -> bool:
        """Detect items placed on the desk during checkout."""
        cam = cv2.VideoCapture(0) if src is None else cv2.VideoCapture(src)
        if not cam.isOpened():
            if src is None:
                LOGGER.error(
                    "Failed to open camera. Docker Desktop on macOS does not "
                    "expose host webcams to Linux containers. Run the kiosk "
                    "app locally for live camera access, or provide a video "
                    "file with --source or KIOSK_SOURCE."
                )
            else:
                LOGGER.error(f"Failed to open video source: {src}")
            return False

        width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        width, height = self.imgsz, int(self.imgsz / width * height)
        output_name = (
            f"transaction-{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
            if src is None
            else src.split("/")[-1]
        )

        output_video = cv2.VideoWriter(
            filename=str(self.pth_output / output_name),
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=20 if src is None else cam.get(cv2.CAP_PROP_FPS),
            frameSize=(width, height),
        )

        is_smy = False
        hist = [{}] * 24 * 3  # a queue stores history of 3-second frames

        try:
            if src is None:
                while True:
                    flag, frame = cam.read()
                    if not flag or frame is None:
                        LOGGER.warning("Failed to read frame from camera.")
                        continue

                    frame, qty = self.scan(frame, summary=is_smy)
                    if frame is None:
                        LOGGER.warning(
                            "Skipping camera frame because scanning failed."
                        )
                        continue

                    hist.pop(0)
                    hist.append(qty)
                    is_smy = bool(hist[0]) and hist[:30] == hist[-30:]

                    frame = cv2.resize(frame, (width, height))
                    output_video.write(frame)
                    cv2.imshow("Camera", frame)

                    if cv2.waitKey(1) == ord("q"):
                        break
            else:
                total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
                for _ in tqdm(range(total_frames)):
                    flag, frame = cam.read()
                    if not flag or frame is None:
                        LOGGER.warning(
                            "Failed to read frame from video source."
                        )
                        break

                    frame, qty = self.scan(frame, summary=is_smy)
                    if frame is None:
                        LOGGER.warning(
                            "Stopping video processing because "
                            "scanning failed."
                        )
                        break

                    hist.pop(0)
                    hist.append(qty)
                    is_smy = bool(hist[0]) and hist[:30] == hist[-30:]

                    frame = cv2.resize(frame, (width, height))
                    output_video.write(frame)

        except Exception:
            LOGGER.error("Error in scanning.", exc_info=True)
        finally:
            cam.release()
            output_video.release()
            cv2.destroyAllWindows()

        LOGGER.info(f"Results saved to \033[1m{self.pth_output}\033[0m")
        return True
