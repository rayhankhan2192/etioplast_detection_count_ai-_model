import os
import logging
from django.conf import settings
# v1.0.1
class Config:
    # All outputs go under MEDIA_ROOT/detections
    SAVE_DIR = os.path.join(settings.MEDIA_ROOT, 'detections')
    MEASURE_DECIMALS = 3

    # Detection / NMS
    # 0.1 is often too low (lots of noise). 0.20–0.30 is a good balance.
    CONFIDENCE_THRESHOLD = 0.5
    MIN_CONTOUR_AREA = 30          # allow smaller objects to pass (was 30)
    IOU_THRESHOLD = 0.5            # be more willing to merge overlaps

    # Containment helpers
    PARENT_DILATE_ITER = 2         # grow parent a bit more so children fall inside
    PARENT_KERNEL = 3

    # Overlap / inside-point minima (lenient)
    # key: class_id => min overlap fraction
    # 1: PLB, 2: Prothylakoid, 3: Plastoglobule, 4: Starch Grain
    CLASS_OVERLAP_MIN = {
        1: 0.45,   # was 0.60
        2: 0.25,   # was 0.35
        3: 0.45,   # was 0.55
        4: 0.45    # was 0.55
    }
    CLASS_POINTS_INSIDE_MIN = {
        1: 0.50,   # was 0.60
        2: 0.30,   # was 0.40
        3: 0.45,   # was 0.55
        4: 0.45    # was 0.55
    }

    # Etioplast completeness/shape (widen windows)
    BORDER_MARGIN = 2              # was 4; allow closer-to-border parents
    AR_MIN, AR_MAX = 0.55, 1.80    # was 0.75–1.33; real shapes vary more
    EXTENT_MIN = 0.48              # was 0.60
    RECT_FILL_MIN = 0.55           # was 0.65
    POLY_EPS_FRAC = 0.03           # was 0.02; simplify contours a bit more
    POLY_MIN, POLY_MAX = 3, 12     # was 3–8; allow richer polygons

    # Drawing
    ALPHA_BLEND = 0.6
    LINE_THICKNESS = 3
    ELLIPSE_THICKNESS = 4
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2


    IMG_SIZE = 640

    # Default scale: pixels per micrometer (px/µm) — you can override per-image
    PX_PER_UM = 0

os.makedirs(Config.SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(Config.SAVE_DIR, 'masks'), exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASS_DEFINITIONS = {
    0: {'name': 'Etioplast', 'color': (255, 0, 0)},
    1: {'name': 'PLB', 'color': (0, 255, 0)},
    2: {'name': 'Prothylakoid', 'color': (0, 0, 255)},
    3: {'name': 'Plastoglobule', 'color': (255, 255, 0)},
    4: {'name': 'Starch Grain', 'color': (128, 0, 128)},
}

class DetectedObject:
    def __init__(self, class_id, contour, mask, confidence, bbox, yolo_detection_idx):
        import cv2
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
        import cv2
        M = cv2.moments(self.contour)
        if M['m00'] != 0:
            return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        return (int((self.bbox[0] + self.bbox[2]) / 2), int((self.bbox[1] + self.bbox[3]) / 2))

    def get_name(self):
        return CLASS_DEFINITIONS[self.class_id]['name']

    def get_color(self):
        return CLASS_DEFINITIONS[self.class_id]['color']
