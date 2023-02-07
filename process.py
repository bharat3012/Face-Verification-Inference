import cv2
import numpy as np
from numba import njit

@njit(cache=True)
def nms(dets, thresh=0.4):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def resize_image(image, max_size: list = None):

    if max_size is None:
        max_size = [640, 640]

    cw = max_size[0]
    ch = max_size[1]
    h, w, _ = image.shape

    scale_factor = min(cw / w, ch / h)
    # If image is too small, it may contain only single face, which leads to decreased detection accuracy,
    # so we reduce scale factor by some factor
    if scale_factor > 2:
        scale_factor = scale_factor * 0.7

    transformed_image = cv2.resize(image, (0, 0), fx=scale_factor,
                                             fy=scale_factor,
                                             interpolation=cv2.INTER_LINEAR)
    h, w, _ = transformed_image.shape

    if w < cw:
       transformed_image = cv2.copyMakeBorder(transformed_image, 0, 0, 0, cw - w,
                                                    cv2.BORDER_CONSTANT)
    if h < ch:
        transformed_image = cv2.copyMakeBorder(transformed_image, 0, ch - h, 0, 0,
                                                    cv2.BORDER_CONSTANT)

    return transformed_image, scale_factor


def decode(heatmap, scale, offset, landmark, size, threshold=0.1, nms_threshold=0.3):
    heatmap = np.squeeze(heatmap)
    scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
    offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
    c0, c1 = np.where(heatmap > threshold)

    boxes, lms = [], []

    if len(c0) > 0:
        for i in range(len(c0)):
            s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
            o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
            s = heatmap[c0[i], c1[i]]
            x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
            x1, y1 = min(x1, size[1]), min(y1, size[0])
            boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])
            
            #if lms
            lm = []
            for j in range(5):
                lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
            lms.append(lm)
            #if lms
            
        boxes = np.asarray(boxes, dtype=np.float32)
        keep = nms(boxes, nms_threshold)
        boxes = boxes[keep, :]
        
        #if lms
        lms = np.asarray(lms, dtype=np.float32)
        lms = lms[keep, :]
        lms = lms.reshape((-1, 5, 2))
        #if lms
        
    return boxes, lms


def postprocess(heatmap, lms, offset, scale, size, threshold, nms_threshold):

    dets, lms = decode(heatmap, scale, offset, lms, size, threshold=threshold, nms_threshold=nms_threshold)

    if len(dets) > 0:
        dets[:, 0:4:2], dets[:, 1:4:2] = dets[:, 0:4:2], dets[:, 1:4:2]
        lms[:, 0:10:2], lms[:, 1:10:2] = lms[:, 0:10:2], lms[:, 1:10:2]
    else:
        dets = np.empty(shape=[0, 5], dtype=np.float32)
        lms = np.empty(shape=[0, 10], dtype=np.float32)

    return dets, lms

# Translate bboxes and landmarks from resized to original image size
def reproject_points(dets, scale: float):
    if scale != 1.0:
        dets = dets / scale
    return dets

def draw_faces(image, bboxes, landmarks, scores, draw_landmarks=True, draw_scores=True, draw_sizes=True):
    for i, bbox in enumerate(bboxes):
        pt1 = tuple(bbox[0:2].astype(int))
        pt2 = tuple(bbox[2:4].astype(int))
        color = (0, 255, 0)
        x, y = pt1
        r, b = pt2
        w = r - x

        cv2.rectangle(image, pt1, pt2, color, 1)

        if draw_landmarks:
            lms = landmarks[i].astype(int)
            pt_size = int(w * 0.05)
            cv2.circle(image, (lms[0][0], lms[0][1]), 1, (0, 0, 255), pt_size)
            cv2.circle(image, (lms[1][0], lms[1][1]), 1, (0, 255, 255), pt_size)
            cv2.circle(image, (lms[2][0], lms[2][1]), 1, (255, 0, 255), pt_size)
            cv2.circle(image, (lms[3][0], lms[3][1]), 1, (0, 255, 0), pt_size)
            cv2.circle(image, (lms[4][0], lms[4][1]), 1, (255, 0, 0), pt_size)

        if draw_scores:
            text = f"{scores[i]:.3f}"
            pos = (x + 3, y - 5)
            textcolor = (0, 0, 0)
            thickness = 1
            border = int(thickness / 2)
            cv2.rectangle(image, (x - border, y - 21, w + thickness, 21), color, -1, 16)
            cv2.putText(image, text, pos, 0, 0.5, color, 3, 16)
            cv2.putText(image, text, pos, 0, 0.5, textcolor, 1, 16)

        if draw_sizes:
            text = f"w:{w}"
            pos = (x + 3, b - 5)
            cv2.putText(image, text, pos, 0, 0.5, (0, 0, 0), 3, 16)
            cv2.putText(image, text, pos, 0, 0.5, (0, 255, 0), 1, 16)

    total = f'faces: {len(bboxes)}'
    bottom = image.shape[0]
    cv2.putText(image, total, (5, bottom - 5), 0, 1, (0, 0, 0), 3, 16)
    cv2.putText(image, total, (5, bottom - 5), 0, 1, (0, 255, 0), 1, 16)

    return image


