
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