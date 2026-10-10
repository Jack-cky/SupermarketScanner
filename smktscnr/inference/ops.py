import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


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
