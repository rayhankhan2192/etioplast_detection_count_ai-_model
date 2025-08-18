# v1.0.1

import os
import logging
import cv2
import numpy as np
from skimage.transform import resize
from ultralytics import YOLO
from dotenv import load_dotenv

from .config import Config, CLASS_DEFINITIONS, DetectedObject

load_dotenv()
logger = logging.getLogger(__name__)

_YOLO = None
def get_model():
    global _YOLO
    if _YOLO is None:
        model_path = os.getenv("MODEL_PATH") or Config.MODEL_PATH
        logger.info(f"Loading YOLO model: {model_path}")
        _YOLO = YOLO(model_path)
    return _YOLO

def run_yolo_detection(self, image_path):
    logger.info(f"Running YOLO detection on: {os.path.basename(image_path)}")
    model = get_model()
    results = model.predict(
        source=image_path,
        imgsz=Config.IMG_SIZE,
        conf=Config.CONFIDENCE_THRESHOLD,
        iou=Config.IOU_THRESHOLD,
        save=False,
        verbose=False
    )
    if not results or results[0].masks is None:
        logger.warning("No detections found by YOLO")
        return None
    return results[0]

def extract_model_summary(self, yolo_result):
    if yolo_result.boxes is not None:
        class_ids = yolo_result.boxes.cls.cpu().numpy().astype(int)
        class_counts = {}
        for cid in class_ids:
            cname = CLASS_DEFINITIONS[cid]['name']
            class_counts[cname] = class_counts.get(cname, 0) + 1
        self.model_summary = class_counts
        self.stats['model_detections'] = class_counts
        logger.info("YOLO Model Detection Summary:")
        for k, v in class_counts.items():
            logger.info(f"  {k}: {v}")
        return True
    return False

def convert_yolo_to_objects(yolo_result, image_path, min_contour_area=None):
    img = cv2.imread(image_path)
    if img is None:
        return [], None
    orig_h, orig_w = img.shape[:2]
    image_shape = (orig_h, orig_w)

    masks = yolo_result.masks.data.cpu().numpy()
    class_ids = yolo_result.boxes.cls.cpu().numpy().astype(int)
    confidences = yolo_result.boxes.conf.cpu().numpy()
    boxes = yolo_result.boxes.xyxy.cpu().numpy()

    if min_contour_area is None:
        min_contour_area = Config.MIN_CONTOUR_AREA

    detected = []
    for i, (mask, cid, conf, box) in enumerate(zip(masks, class_ids, confidences, boxes)):
        if cid not in CLASS_DEFINITIONS:
            continue
        resized_mask = (resize(mask, (orig_h, orig_w), order=0, preserve_range=True, anti_aliasing=False)
                        .astype(np.uint8) * 255)
        contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        main_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(main_contour) < min_contour_area:
            continue

        imgsz = Config.IMG_SIZE
        scale_x, scale_y = orig_w / imgsz, orig_h / imgsz
        scaled_box = [int(box[0]*scale_x), int(box[1]*scale_y), int(box[2]*scale_x), int(box[3]*scale_y)]

        obj = DetectedObject(cid, main_contour, resized_mask, float(conf), scaled_box, i)
        obj.object_id = f"{CLASS_DEFINITIONS[cid]['name']}_{i+1}"
        detected.append(obj)

    logger.info(f"Converted {len(detected)} YOLO detections to objects")
    return detected, image_shape
