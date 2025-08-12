
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.files.storage import default_storage
from django.conf import settings
from ultralytics import YOLO
import os
import cv2
import numpy as np
import uuid
from .segmentation import SegmentationAnalyzer
from .generativeai import get_generative_response
from dotenv import load_dotenv
import logging
from skimage.transform import resize

# model_path = "/root/Rayhan/etioplast_detection_count_ai-_model/TrainedModel/best8v2.pt"
# model = YOLO(model_path)

load_dotenv()
model_path = os.getenv("MODEL_PATH")

class Config:
    MODEL_PATH = r'E:\Python\Research\Ethioplast\Etioplast New\Origial Data\Annotated Data Yolo\Model\modelv3_m100.pt'
    SOURCE_IMAGE = r'E:\Python\Research\Ethioplast\Etioplast New\Origial Data\Annotated Data Yolo\test copy\bean__24.png'
    SAVE_DIR = r'save'
    CONFIDENCE_THRESHOLD = 0.5
    MIN_CONTOUR_AREA = 30
    IOU_THRESHOLD = 0.3

    # STRICT containment (no fallbacks)
    # parent mask dilation: a tiny tolerance helps thin Prothylakoids without letting outsiders in
    PARENT_DILATE_ITER = 1         # keep small; 0–2
    PARENT_KERNEL = 3

    # class-specific overlap minima (intersection_area / child_area)
    CLASS_OVERLAP_MIN = {
        1: 0.60,  # PLB stricter
        2: 0.35,  # Prothylakoid tolerant
        3: 0.55,  # Plastoglobule
        4: 0.55,  # Starch Grain
    }
    # % of contour points that must lie inside parent
    CLASS_POINTS_INSIDE_MIN = {
        1: 0.60,
        2: 0.40,
        3: 0.55,
        4: 0.55,
    }

    # Etioplast completeness/shape
    BORDER_MARGIN = 4
    AR_MIN, AR_MAX = 0.75, 1.33
    EXTENT_MIN = 0.60
    RECT_FILL_MIN = 0.65
    POLY_EPS_FRAC = 0.02
    POLY_MIN, POLY_MAX = 3, 8

    ALPHA_BLEND = 0.6
    LINE_THICKNESS = 3
    ELLIPSE_THICKNESS = 4
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2

os.makedirs(Config.SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(Config.SAVE_DIR, 'masks'), exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASS_DEFINITIONS = {
    0: {'name': 'Etioplast', 'color': (255, 0, 0)},
    1: {'name': 'PLB', 'color': (0, 255, 0)},
    2: {'name': 'Prothylakoid', 'color': (0, 0, 255)},
    3: {'name': 'Plastoglobule', 'color': (255, 255, 0)},
    4: {'name': 'Starch Grain', 'color': (128, 0, 128)}
}

class DetectedObject:
    def __init__(self, class_id, contour, mask, confidence, bbox, yolo_detection_idx):
        self.class_id = class_id
        self.contour = contour
        self.mask = mask
        self.confidence = float(confidence)
        self.yolo_idx = yolo_detection_idx
        self.bbox = bbox  # [x1,y1,x2,y2]
        self.area = cv2.contourArea(contour)
        self.center = self._calculate_center()
        self.children = []
        self.parent = None
        self.is_valid = True
        self.object_id = None

    def _calculate_center(self):
        M = cv2.moments(self.contour)
        if M['m00'] != 0:
            return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        return (int((self.bbox[0] + self.bbox[2]) / 2), int((self.bbox[1] + self.bbox[3]) / 2))

    def get_name(self):
        return CLASS_DEFINITIONS[self.class_id]['name']

    def get_color(self):
        return CLASS_DEFINITIONS[self.class_id]['color']


class ModelFirstDetector:
    """YOLO-first with STRICT rules: Valid Etioplast = complete + square-like + ≥1 PLB (strictly inside).
    Only organelles strictly inside a VALID Etioplast are kept and saved.
    """
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.detected_objects = []
        self.valid_etioplasts = []
        self.model_summary = {}
        self.image_shape = None
        self.stats = {
            'model_detections': {},
            'processed_objects': 0,
            'valid_etioplasts': 0,
            'rejected_etioplasts': 0,
            'valid_organelles': 0,
            'rejected_organelles': 0,
            'reject_no_plb': 0,
            'reject_incomplete': 0,
            'reject_not_square_like': 0
        }

    # YOLO detection and conversion
    def run_yolo_detection(self, image_path):
        logger.info(f"Running YOLO detection on: {os.path.basename(image_path)}")
        results = self.model.predict(
            source=image_path,
            imgsz=640,
            conf=Config.CONFIDENCE_THRESHOLD,
            iou=Config.IOU_THRESHOLD,
            save=False,
            verbose=True
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
    

    def convert_yolo_to_objects(self, yolo_result, image_path):
        img = cv2.imread(image_path)
        if img is None:
            logger.error("Failed to read image")
            return []
        orig_h, orig_w = img.shape[:2]
        self.image_shape = (orig_h, orig_w)

        masks = yolo_result.masks.data.cpu().numpy()
        class_ids = yolo_result.boxes.cls.cpu().numpy().astype(int)
        confidences = yolo_result.boxes.conf.cpu().numpy()
        boxes = yolo_result.boxes.xyxy.cpu().numpy()

        detected_objects = []
        for i, (mask, class_id, confidence, box) in enumerate(zip(masks,class_ids, confidences, boxes)):
            resized_mask = resize(mask, (orig_h, orig_w), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)*255
            contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            main_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(main_contour) < Config.MIN_CONTOUR_AREA:
                continue
            scale_x = orig_w / 640
            scale_y = orig_h / 640

            scale_box = [
                int(box[0] * scale_x),
                int(box[1] * scale_y),
                int(box[2] * scale_x),
                int(box[3] * scale_y)
            ]
            detected_object = DetectedObject(class_id, main_contour, resized_mask, float(confidence), scale_box, i)
            detected_object.object_id = f"{CLASS_DEFINITIONS[class_id]['name']}_{i+1}"
            detected_objects.append(detected_object)
        logger.info(f"Converted {len(detected_objects)} YOLO detection to objects")
        self.detected_objects = detected_objects
        self.stats['processed_objects'] = len(detected_objects)
        return detected_objects
    



