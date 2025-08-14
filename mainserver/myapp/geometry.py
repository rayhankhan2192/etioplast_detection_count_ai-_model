import cv2
import numpy as np
from .config import Config

def mask_from_contour(image_shape, contour, dilate_iters=0):
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    if dilate_iters > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (Config.PARENT_KERNEL, Config.PARENT_KERNEL))
        mask = cv2.dilate(mask, kernel, iterations=dilate_iters)
    return mask

def points_inside_ratio(image_shape, contour, parent_mask):
    pts = contour[:, 0, :]
    yy = pts[:, 1].clip(0, image_shape[0]-1)
    xx = pts[:, 0].clip(0, image_shape[1]-1)
    inside = parent_mask[yy, xx] > 0
    return float(np.count_nonzero(inside)) / max(1, len(pts))

def overlap_ratio(child_mask, parent_mask):
    inter = cv2.bitwise_and(child_mask, parent_mask)
    inter_area = float(np.sum(inter > 0))
    child_area = float(np.sum(child_mask > 0))
    return (inter_area / child_area) if child_area > 0 else 0.0

def touches_border(image_shape, contour):
    h, w = image_shape
    m = Config.BORDER_MARGIN
    x, y, bw, bh = cv2.boundingRect(contour)
    if x <= m or y <= m or (x + bw) >= (w - m) or (y + bh) >= (h - m):
        return True
    if np.any(contour[:, 0, 0] <= m) or np.any(contour[:, 0, 1] <= m) or \
       np.any(contour[:, 0, 0] >= (w - m)) or np.any(contour[:, 0, 1] >= (h - m)):
        return True
    return False

def is_square_like(contour):
    area = cv2.contourArea(contour)
    if area <= 0:
        return False
    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = float(w * h) if w > 0 and h > 0 else 1.0
    ar = w / h if h > 0 else 999.0
    if not (Config.AR_MIN <= ar <= Config.AR_MAX):
        return False
    extent = area / bbox_area
    if extent < Config.EXTENT_MIN:
        return False
    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    rarea = float(rw * rh) if rw > 0 and rh > 0 else 1.0
    fill_ratio = area / rarea
    if fill_ratio < Config.RECT_FILL_MIN:
        return False
    peri = cv2.arcLength(contour, True)
    eps = Config.POLY_EPS_FRAC * peri
    approx = cv2.approxPolyDP(contour, eps, True)
    corners = len(approx)
    if not (Config.POLY_MIN <= corners <= Config.POLY_MAX):
        return False
    return True
