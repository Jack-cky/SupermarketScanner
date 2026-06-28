import cv2
import numpy as np
import supervision as sv

from hailo_platform import (
    ConfigureParams,
    Device,
    FormatType,
    HEF,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)

from ..utils import ProductCatalog


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis=axis, keepdims=True)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0,
        boxes[:, 3] - boxes[:, 1],
    )
    union = area1 + area2 - inter

    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.45,
) -> np.ndarray:
    keep = []

    for class_id in np.unique(class_ids):
        class_indices = np.where(class_ids == class_id)[0]
        order = class_indices[np.argsort(scores[class_indices])[::-1]]

        while order.size > 0:
            current = order[0]
            keep.append(current)
            if order.size == 1:
                break

            ious = box_iou(boxes[current], boxes[order[1:]])
            order = order[1:][ious <= iou_threshold]

    if len(keep) == 0:
        return np.empty((0,), dtype=np.int32)

    keep = np.array(keep, dtype=np.int32)
    return keep[np.argsort(scores[keep])[::-1]]


def normalise_output(
    output: np.ndarray,
    class_count: int | None = None,
) -> np.ndarray:
    if output.ndim == 3:
        output = np.expand_dims(output, axis=0)

    if output.ndim != 4:
        raise ValueError(f"Unsupported output shape: {output.shape}")

    if output.shape[-1] in (32, 64) or (
        class_count is not None and output.shape[-1] == class_count
    ):
        return output

    if output.shape[1] in (32, 64) or (
        class_count is not None and output.shape[1] == class_count
    ):
        return np.transpose(output, (0, 2, 3, 1))

    return output


def infer_score_channels(output_infos: list) -> int:
    channel_sizes = []

    for output_info in output_infos:
        shape = tuple(int(dim) for dim in output_info.shape)
        candidate = shape[-1]
        if candidate not in (32, 64):
            channel_sizes.append(candidate)
            continue

        candidate = shape[0]
        if candidate not in (32, 64):
            channel_sizes.append(candidate)

    scores = [channels for channels in channel_sizes if channels > 0]
    if not scores:
        output_shapes = {
            output_info.name: output_info.shape
            for output_info in output_infos
        }
        raise ValueError(
            f"Unable to infer score head channels from HEF outputs: "
            f"{output_shapes}"
        )

    return max(set(scores), key=scores.count)


class HailoModel:
    def __init__(self, weights: str):
        self.class_names = list(ProductCatalog.PRICES)
        self.expected_class_count = len(self.class_names)
        self._infer_vstreams_cls = InferVStreams
        self._hef = HEF(weights)

        device_ids = Device.scan()
        if not device_ids:
            raise RuntimeError("No Hailo device detected.")

        self._vdevice = VDevice(device_ids=device_ids)
        configure_params = ConfigureParams.create_from_hef(
            self._hef,
            interface=HailoStreamInterface.PCIe,
        )
        self._net_group = self._vdevice.configure(
            self._hef,
            configure_params,
        )[0]
        self._net_group_params = self._net_group.create_params()
        self._input_info = self._hef.get_input_vstream_infos()[0]
        self._output_infos = self._hef.get_output_vstream_infos()
        self.score_channels = infer_score_channels(self._output_infos)
        if self.score_channels != self.expected_class_count:
            raise ValueError(
                "Incompatible HEF model: "
                "expected "
                f"{self.expected_class_count} classes for the supermarket "
                "catalog, "
                f"but `{weights}` exposes {self.score_channels} class logits. "
                "This usually means the HEF was exported from the base COCO "
                "YOLOv8 segmentation model instead of the fine-tuned "
                "supermarket model. Re-export and compile the 8-class "
                "model."
            )
        self._input_params = InputVStreamParams.make_from_network_group(
            self._net_group,
            quantized=True,
        )
        self._output_params = OutputVStreamParams.make_from_network_group(
            self._net_group,
            quantized=False,
            format_type=FormatType.FLOAT32,
        )

    def predict(self, **kwargs):
        source = kwargs.get("source")
        conf = kwargs.get("conf", 0.5)

        if source is None:
            raise ValueError("`source` is required for Hailo inference.")

        scene = cv2.imread(source) if isinstance(source, str) else source
        if scene is None:
            raise ValueError(f"Unable to load input source: {source}")

        orig_h, orig_w = scene.shape[:2]
        model_h = int(self._input_info.shape[0])
        model_w = int(self._input_info.shape[1])
        resized = cv2.resize(scene, (model_w, model_h))
        input_data = np.expand_dims(
            np.asarray(resized),
            axis=0,
        ).astype(np.uint8)

        with self._net_group.activate(self._net_group_params):
            with self._infer_vstreams_cls(
                self._net_group,
                self._input_params,
                self._output_params,
            ) as inferer:
                results = inferer.infer(input_data)

        return self._postprocess(
            results,
            orig_h=orig_h,
            orig_w=orig_w,
            model_h=model_h,
            model_w=model_w,
            conf=conf,
        )

    def _postprocess(
        self,
        results: dict,
        *,
        orig_h: int,
        orig_w: int,
        model_h: int,
        model_w: int,
        conf: float,
    ) -> sv.Detections:
        bbox_outputs, score_outputs, coeff_outputs, proto_output = (
            self._collect_outputs(results)
        )
        decoded_boxes = self._decode_boxes(bbox_outputs)
        class_scores = np.concatenate(
            [out.reshape(1, -1, out.shape[-1]) for out in score_outputs],
            axis=1,
        )[0]
        mask_coeffs = np.concatenate(
            [out.reshape(1, -1, out.shape[-1]) for out in coeff_outputs],
            axis=1,
        )[0]

        candidates = self._filter_candidates(
            decoded_boxes,
            class_scores,
            mask_coeffs,
            conf=conf,
            model_h=model_h,
            model_w=model_w,
        )
        if candidates is None:
            return sv.Detections.empty()

        xyxy_boxes, scores, class_ids, mask_coeffs = candidates
        return self._build_detections(
            xyxy_boxes,
            scores,
            class_ids,
            mask_coeffs,
            proto_output[0],
            orig_h=orig_h,
            orig_w=orig_w,
            model_h=model_h,
            model_w=model_w,
        )

    def _collect_outputs(
        self,
        results: dict,
    ) -> tuple[
        list[np.ndarray],
        list[np.ndarray],
        list[np.ndarray],
        np.ndarray,
    ]:
        bbox_outputs = []
        score_outputs = []
        coeff_outputs = []
        proto_output = None
        proto_area = -1

        for output_info in self._output_infos:
            output = normalise_output(
                np.asarray(results[output_info.name]),
                class_count=self.score_channels,
            )

            channels = output.shape[-1]
            if channels == 64:
                bbox_outputs.append(output)
            elif channels == self.score_channels:
                score_outputs.append(output)
            elif channels == 32:
                area = output.shape[1] * output.shape[2]
                if area > proto_area:
                    if proto_output is not None:
                        coeff_outputs.append(proto_output)
                    proto_output = output
                    proto_area = area
                else:
                    coeff_outputs.append(output)

        if (
            len(bbox_outputs) != 3
            or len(score_outputs) != 3
            or len(coeff_outputs) != 3
            or proto_output is None
        ):
            shapes = {
                info.name: np.asarray(results[info.name]).shape
                for info in self._output_infos
            }
            raise ValueError(f"Unexpected segmentation outputs: {shapes}")

        bbox_outputs.sort(key=lambda x: x.shape[1] * x.shape[2])
        score_outputs.sort(key=lambda x: x.shape[1] * x.shape[2])
        coeff_outputs.sort(key=lambda x: x.shape[1] * x.shape[2])

        return bbox_outputs, score_outputs, coeff_outputs, proto_output

    def _decode_boxes(self, bbox_outputs: list[np.ndarray]) -> np.ndarray:
        reg_max = 15
        reg_range = np.arange(reg_max + 1, dtype=np.float32)
        decoded_boxes = []

        for bbox_output, stride in zip(bbox_outputs, (32, 16, 8), strict=True):
            _, grid_h, grid_w, channels = bbox_output.shape
            if channels != 4 * (reg_max + 1):
                raise ValueError(
                    f"Unexpected bbox output shape: {bbox_output.shape}"
                )

            grid_x = (np.arange(grid_w, dtype=np.float32) + 0.5) * stride
            grid_y = (np.arange(grid_h, dtype=np.float32) + 0.5) * stride
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            centers = np.stack(
                (grid_x, grid_y, grid_x, grid_y),
                axis=-1,
            ).reshape(1, grid_h * grid_w, 4)

            distances = bbox_output.reshape(1, grid_h * grid_w, 4, reg_max + 1)
            distances = (
                np.sum(softmax(distances, axis=-1) * reg_range, axis=-1)
                * stride
            )
            distances = np.concatenate(
                (-distances[:, :, :2], distances[:, :, 2:]),
                axis=-1,
            )
            xyxy = centers + distances
            xywh = np.stack(
                (
                    (xyxy[:, :, 0] + xyxy[:, :, 2]) / 2,
                    (xyxy[:, :, 1] + xyxy[:, :, 3]) / 2,
                    xyxy[:, :, 2] - xyxy[:, :, 0],
                    xyxy[:, :, 3] - xyxy[:, :, 1],
                ),
                axis=-1,
            )
            decoded_boxes.append(xywh)

        return np.concatenate(decoded_boxes, axis=1)[0]

    def _filter_candidates(
        self,
        decoded_boxes: np.ndarray,
        class_scores: np.ndarray,
        mask_coeffs: np.ndarray,
        *,
        conf: float,
        model_h: int,
        model_w: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
        keep = scores >= conf

        if not np.any(keep):
            return None

        decoded_boxes = decoded_boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        mask_coeffs = mask_coeffs[keep]

        xyxy_boxes = decoded_boxes.copy()
        xyxy_boxes[:, 0] = decoded_boxes[:, 0] - decoded_boxes[:, 2] / 2
        xyxy_boxes[:, 1] = decoded_boxes[:, 1] - decoded_boxes[:, 3] / 2
        xyxy_boxes[:, 2] = decoded_boxes[:, 0] + decoded_boxes[:, 2] / 2
        xyxy_boxes[:, 3] = decoded_boxes[:, 1] + decoded_boxes[:, 3] / 2
        xyxy_boxes[:, [0, 2]] = np.clip(xyxy_boxes[:, [0, 2]], 0, model_w)
        xyxy_boxes[:, [1, 3]] = np.clip(xyxy_boxes[:, [1, 3]], 0, model_h)

        valid = (xyxy_boxes[:, 2] > xyxy_boxes[:, 0]) & (
            xyxy_boxes[:, 3] > xyxy_boxes[:, 1]
        )
        if not np.any(valid):
            return None

        xyxy_boxes = xyxy_boxes[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]
        mask_coeffs = mask_coeffs[valid]

        keep_indices = nms(xyxy_boxes, scores, class_ids)
        if keep_indices.size == 0:
            return None

        xyxy_boxes = xyxy_boxes[keep_indices]
        scores = scores[keep_indices]
        class_ids = class_ids[keep_indices]
        mask_coeffs = mask_coeffs[keep_indices]

        return xyxy_boxes, scores, class_ids, mask_coeffs

    def _build_detections(
        self,
        xyxy_boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        mask_coeffs: np.ndarray,
        protos: np.ndarray,
        *,
        orig_h: int,
        orig_w: int,
        model_h: int,
        model_w: int,
    ) -> sv.Detections:
        proto_h, proto_w, proto_c = protos.shape
        mask_preds = sigmoid(mask_coeffs @ protos.reshape(-1, proto_c).T)
        mask_preds = mask_preds.reshape(-1, proto_h, proto_w)
        mask_preds = cv2.resize(
            np.transpose(mask_preds, (1, 2, 0)),
            (model_w, model_h),
            interpolation=cv2.INTER_LINEAR,
        )
        if mask_preds.ndim == 2:
            mask_preds = mask_preds[..., np.newaxis]
        mask_preds = np.transpose(mask_preds, (2, 0, 1))

        scale_x = orig_w / model_w
        scale_y = orig_h / model_h

        boxes = []
        masks = []
        scores_kept = []
        ids = []
        names = []

        for box, mask, score, class_id in zip(
            xyxy_boxes,
            mask_preds,
            scores,
            class_ids,
            strict=True,
        ):
            x1_m = int(np.clip(np.floor(box[0]), 0, model_w - 1))
            y1_m = int(np.clip(np.floor(box[1]), 0, model_h - 1))
            x2_m = int(np.clip(np.ceil(box[2]), 0, model_w))
            y2_m = int(np.clip(np.ceil(box[3]), 0, model_h))
            if x2_m <= x1_m or y2_m <= y1_m:
                continue

            mask[:y1_m, :] = 0
            mask[y2_m:, :] = 0
            mask[:, :x1_m] = 0
            mask[:, x2_m:] = 0

            full_mask = (
                cv2.resize(
                    mask,
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                > 0.5
            )
            if not np.any(full_mask):
                continue

            x1 = int(np.clip(box[0] * scale_x, 0, orig_w - 1))
            y1 = int(np.clip(box[1] * scale_y, 0, orig_h - 1))
            x2 = int(np.clip(box[2] * scale_x, 0, orig_w - 1))
            y2 = int(np.clip(box[3] * scale_y, 0, orig_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            masks.append(full_mask)
            scores_kept.append(float(score))
            ids.append(int(class_id))
            names.append(self.class_names[int(class_id)])

        if len(boxes) == 0:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32).reshape(-1, 4),
            mask=np.array(masks, dtype=bool),
            confidence=np.array(scores_kept, dtype=np.float32),
            class_id=np.array(ids, dtype=np.int32),
            data={"class_name": names},
        )
