# v1.0.1
import os
import cv2
import numpy as np
from .config import Config, CLASS_DEFINITIONS, logger
from .geometry import mask_from_contour, points_inside_ratio, overlap_ratio, touches_border, is_square_like

def strict_contains(self, child_obj, parent_obj):
    dil = Config.PARENT_DILATE_ITER if child_obj.class_id == 2 else Config.PARENT_DILATE_ITER
    parent_mask = mask_from_contour(self.image_shape, parent_obj.contour, dilate_iters=dil)

    ov_min = Config.CLASS_OVERLAP_MIN.get(child_obj.class_id, 0.5)
    pt_min = Config.CLASS_POINTS_INSIDE_MIN.get(child_obj.class_id, 0.5)

    ov = overlap_ratio(child_obj.mask, parent_mask)
    if ov < ov_min:
        return False

    pts = points_inside_ratio(self.image_shape, child_obj.contour, parent_mask)
    if pts < pt_min:
        return False

    return True

def build_hierarchy(self):
    logger.info("Building hierarchy with STRICT rules (valid-etioplast-only).")
    etioplasts = [o for o in self.detected_objects if o.class_id == 0]
    plbs = [o for o in self.detected_objects if o.class_id == 1]
    others = [o for o in self.detected_objects if o.class_id > 1]

    if not etioplasts:
        logger.warning("No Etioplasts detected by model!")
        return

    etioplasts.sort(key=lambda x: x.area, reverse=True)
    valid_list = []
    for idx, et in enumerate(etioplasts):
        logger.info(f"Check Etioplast {idx+1}/{len(etioplasts)} (conf: {et.confidence:.2f}, area: {et.area:.0f})")
        plbs_inside = [p for p in plbs if p.parent is None and strict_contains(self, p, et)]
        if len(plbs_inside) == 0:
            logger.warning("❌ Rejected: no PLB strictly inside")
            et.is_valid = False
            self.stats['rejected_etioplasts'] += 1
            self.stats['reject_no_plb'] += 1
            continue

        if touches_border(self.image_shape, et.contour):
            logger.warning("❌ Rejected: incomplete (touches image border)")
            et.is_valid = False
            self.stats['rejected_etioplasts'] += 1
            self.stats['reject_incomplete'] += 1
            continue

        if not is_square_like(et.contour):
            logger.warning("❌ Rejected: not square-like")
            et.is_valid = False
            self.stats['rejected_etioplasts'] += 1
            self.stats['reject_not_square_like'] += 1
            continue

        for p in plbs_inside:
            p.parent = et
            et.children.append(p)

        et.is_valid = True
        valid_list.append(et)

    for et in valid_list:
        for org in others:
            if org.parent is not None:
                continue
            if strict_contains(self, org, et):
                org.parent = et
                et.children.append(org)

    self.valid_etioplasts = valid_list
    self.stats['valid_etioplasts'] = len(self.valid_etioplasts)

    valid_children = 0
    rejected_children = 0
    for obj in self.detected_objects:
        if obj.class_id == 0:
            continue
        keep = False
        for et in self.valid_etioplasts:
            if strict_contains(self, obj, et):
                keep = True
                if obj.parent is None:
                    obj.parent = et
                    et.children.append(obj)
                break
        if keep:
            obj.is_valid = True
            valid_children += 1
        else:
            obj.is_valid = False
            if obj.parent is not None:
                try:
                    obj.parent.children.remove(obj)
                except ValueError:
                    pass
                obj.parent = None
            rejected_children += 1

    self.stats['valid_organelles'] = valid_children
    self.stats['rejected_organelles'] = rejected_children

    logger.info(f"Valid Etioplasts: {self.stats['valid_etioplasts']}; "
                f"Valid organelles: {self.stats['valid_organelles']}; "
                f"Rejected organelles: {self.stats['rejected_organelles']}")

def draw_visualization_improved(self, img):
    overlay = img.copy()
    for et in self.valid_etioplasts:
        cv2.drawContours(overlay, [et.contour], -1, et.get_color(), Config.LINE_THICKNESS)
        label = f"Etioplast ({len([c for c in et.children if c.is_valid])})"
        pos = (et.center[0], et.center[1] - 10)
        cv2.putText(overlay, label, pos, cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE,
                    et.get_color(), Config.FONT_THICKNESS, cv2.LINE_AA)
        for org in et.children:
            if not org.is_valid:
                continue
            cv2.drawContours(overlay, [org.contour], -1, org.get_color(), cv2.FILLED)
            pos = (org.center[0], org.center[1] + 10)
            cv2.putText(overlay, org.get_name(), pos, cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE,
                        org.get_color(), Config.FONT_THICKNESS, cv2.LINE_AA)
    return overlay

def save_outputs_mask(self, base_name):
    masks_dir = os.path.join(Config.SAVE_DIR, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    logger.info("💾 Saving individual masks...")

    combined_mask = np.zeros(self.image_shape, dtype=np.uint8)
    for idx, et in enumerate(self.valid_etioplasts, start=1):
        et_mask = np.zeros(self.image_shape, dtype=np.uint8)
        cv2.drawContours(et_mask, [et.contour], -1, 255, thickness=cv2.FILLED)
        cv2.imwrite(os.path.join(masks_dir, f"{base_name}_Etioplast_{idx}_mask.png"), et_mask)
        combined_mask = cv2.bitwise_or(combined_mask, et_mask)

        for j, org in enumerate([c for c in et.children if c.is_valid], start=1):
            om = np.zeros(self.image_shape, dtype=np.uint8)
            cv2.drawContours(om, [org.contour], -1, 255, thickness=cv2.FILLED)
            oname = org.get_name().replace(" ", "_")
            cv2.imwrite(os.path.join(masks_dir, f"{base_name}_Etioplast_{idx}_{oname}_{j}_mask.png"), om)
            combined_mask = cv2.bitwise_or(combined_mask, om)

    cv2.imwrite(os.path.join(masks_dir, f"{base_name}_combined_mask.png"), combined_mask)
