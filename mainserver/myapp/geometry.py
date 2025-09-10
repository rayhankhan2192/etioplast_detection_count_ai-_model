# v1.0.1
import os
import csv
import math
import cv2
import numpy as np
from skimage.morphology import skeletonize

from .config import Config

#geometry primitives
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

# measurements
class MeasurementCalculator:
    """All measurement conversions in one place (px/µm -> µm, µm²)"""
    def __init__(self, px_per_um: float):
        self.px_per_um = float(px_per_um) if px_per_um else float(Config.PX_PER_UM)
        if self.px_per_um <= 0:
            self.px_per_um = float(Config.PX_PER_UM)

    @property
    def um_per_px(self) -> float:
        return 1.0 / self.px_per_um

    @property
    def um2_per_px2(self) -> float:
        up = self.um_per_px
        return up * up

    def px_area_to_um2(self, area_px2: float) -> float:
        return float(area_px2) * self.um2_per_px2

    def px_len_to_um(self, length_px: float) -> float:
        return float(length_px) * self.um_per_px
    
    
def _binary_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask > 0))

def _equivalent_diameter_px(area_px2: float) -> float:
    if area_px2 <= 0:
        return 0.0
    return math.sqrt(4.0 * area_px2 / math.pi)

def _skeleton_length_px(mask: np.ndarray) -> int:
    sk = skeletonize((mask > 0).astype(bool))
    return int(np.count_nonzero(sk))

def write_measurements_csv(valid_etioplasts, px_per_um: float, save_dir: str, base_name: str):
    """
    Create per-etioplast CSV with all requested metrics (valid objects only).
    """
    calc = MeasurementCalculator(px_per_um)
    rows = []
    for et_idx, et in enumerate(valid_etioplasts, 1):
        full_et_mask = mask_from_contour(et.mask.shape, et.contour, dilate_iters=0)
        row = {
            'Image': base_name,
            'Etioplast_ID': f'Etioplast_{et_idx}',
            #'Etioplast_Area_um2': calc.px_area_to_um2(_binary_area(et.mask)),
            'Etioplast_Area_um2': calc.px_area_to_um2(_binary_area(full_et_mask)),
            'PLB_Area_um2': 0.0,
            'Prothylakoid_Number': 0,
            'Total_Prothylakoid_Length_um': 0.0,
            'Plastoglobule_Number': 0,
            'Plastoglobule_Diameter_Mean_um': 0.0,
            'Starch_Grain_Area_um2': 0.0,
        }
        pg_diams = []
        starch_area_px2 = 0
        plb_area_px2 = 0
        pro_len_px = 0

        for ch in et.children:
            if not getattr(ch, 'is_valid', False):
                continue
            if ch.class_id == 1:  # PLB
                plb_area_px2 += _binary_area(ch.mask)
            elif ch.class_id == 2:  # Prothylakoid
                row['Prothylakoid_Number'] += 1
                pro_len_px += _skeleton_length_px(ch.mask)
            elif ch.class_id == 3:  # Plastoglobule
                row['Plastoglobule_Number'] += 1
                area_px2 = _binary_area(ch.mask)
                d_px = _equivalent_diameter_px(area_px2)
                if d_px > 0:
                    pg_diams.append(d_px)
            elif ch.class_id == 4:  # Starch Grain
                starch_area_px2 += _binary_area(ch.mask)

        row['PLB_Area_um2'] = calc.px_area_to_um2(plb_area_px2)
        row['Total_Prothylakoid_Length_um'] = calc.px_len_to_um(pro_len_px)
        row['Starch_Grain_Area_um2'] = calc.px_area_to_um2(starch_area_px2)
        if pg_diams:
            row['Plastoglobule_Diameter_Mean_um'] = calc.px_len_to_um(float(np.mean(pg_diams)))
        rows.append(row)

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{base_name}_measurements.csv")
    import pandas as pd
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format='%.3f')
    return csv_path


def summarize_for_api(valid_etioplasts, px_per_um: float, decimals: int | None = None):
    if decimals is None:
        from .config import Config
        decimals = getattr(Config, "MEASURE_DECIMALS", 3)
    calc = MeasurementCalculator(px_per_um)
    um_per_px_str = f"{calc.um_per_px:.6f} µm/pixel"

    def r(v):  # round helper
        return round(float(v), decimals)

    # accumulators
    et_area_px2 = 0
    et_count = 0
    plb_area_px2 = 0
    plb_count = 0
    pro_count = 0
    pro_len_px = 0
    pg_count = 0
    pg_diams_px = []
    starch_count = 0
    starch_area_px2 = 0

    for et in valid_etioplasts:
        et_count += 1
        # et_area_px2 += _binary_area(et.mask)
        # sum full-border areas (ignore internal holes)
        full_et_mask = mask_from_contour(et.mask.shape, et.contour, dilate_iters=0)
        et_area_px2 += _binary_area(full_et_mask)

        for ch in et.children:
            if not getattr(ch, 'is_valid', False):
                continue
            if ch.class_id == 1:
                plb_count += 1
                plb_area_px2 += _binary_area(ch.mask)
            elif ch.class_id == 2:
                pro_count += 1
                pro_len_px += _skeleton_length_px(ch.mask)
            elif ch.class_id == 3:
                pg_count += 1
                a = _binary_area(ch.mask)
                d_px = _equivalent_diameter_px(a)
                if d_px > 0:
                    pg_diams_px.append(d_px)
            elif ch.class_id == 4:
                starch_count += 1
                starch_area_px2 += _binary_area(ch.mask)

    analysis = {
        "Etioplast": {
            "total_area_um2": r(calc.px_area_to_um2(et_area_px2)),
            "count": et_count
        },
        "PLB": {
            "total_area_um2": r(calc.px_area_to_um2(plb_area_px2)),
            "count": plb_count
        },
        "Prothylakoid": {
            "count": pro_count,
            "total_length_um": r(calc.px_len_to_um(pro_len_px))
        },
        "Plastoglobule": {
            "count": pg_count,
            "diameter_um": r(calc.px_len_to_um(float(np.mean(pg_diams_px)))) if pg_diams_px else 0.0
        },
        "StarchGrain": {
            "count": starch_count,
            "total_area_um2": r(calc.px_area_to_um2(starch_area_px2))
        }
    }
    return analysis, um_per_px_str
