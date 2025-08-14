import os
import cv2
import json
from datetime import datetime

from .config import Config, CLASS_DEFINITIONS, logger
from .hierarchy_visualization import (
    build_hierarchy, draw_visualization_improved, save_outputs_mask, generate_report
)
from .detection import run_yolo_detection, extract_model_summary, convert_yolo_to_objects

def is_image_file(fname: str) -> bool:
    ext = os.path.splitext(fname)[1].lower()
    return ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

class Process:
    """Pipeline orchestrator (unchanged logic/prints)."""
    def __init__(self):
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

    def now_iso(self):
        return datetime.now().isoformat()

    def process_image(self, image_path):
        logger.info(f"🚀 Starting MODEL-FIRST processing for: {os.path.basename(image_path)}")
        print(f"Processing image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            logger.error("Failed to read image")
            return None
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # YOLO → masks/boxes
        yolo_result = run_yolo_detection(self, image_path)
        if yolo_result is None:
            return None

        # Summary
        if not extract_model_summary(self, yolo_result):
            logger.error("Failed to extract model summary")
            return None

        # Convert YOLO results to objects
        detected, image_shape = convert_yolo_to_objects(
            yolo_result, image_path, min_contour_area=Config.MIN_CONTOUR_AREA
        )
        self.detected_objects = detected
        self.image_shape = image_shape
        self.stats['processed_objects'] = len(self.detected_objects)

        # Build hierarchy & validate
        build_hierarchy(self)

        # Visuals
        overlay = draw_visualization_improved(self, img)
        blended = cv2.addWeighted(overlay, Config.ALPHA_BLEND, img, 1 - Config.ALPHA_BLEND, 0)

        img_name = os.path.basename(image_path)
        base_name = os.path.splitext(img_name)[0]

        # Save masks & images
        save_outputs_mask(self, base_name)
        cv2.imwrite(os.path.join(Config.SAVE_DIR, f"{base_name}_model_first_detection.png"), blended)
        cv2.imwrite(os.path.join(Config.SAVE_DIR, f"{base_name}_overlay.png"), overlay)

        contours_img = img.copy()
        for et in self.valid_etioplasts:
            cv2.drawContours(contours_img, [et.contour], -1, et.get_color(), 2)
            for ch in et.children:
                if ch.is_valid:
                    cv2.drawContours(contours_img, [ch.contour], -1, ch.get_color(), 2)
        cv2.imwrite(os.path.join(Config.SAVE_DIR, f"{base_name}_contours_only.png"), contours_img)

        self.print_summary()
        logger.info(f"✅ Done. Files -> {Config.SAVE_DIR}  |  Masks -> {Config.SAVE_DIR}/masks/")

        # JSON report
        report = generate_report(self, img_name)
        report_path = os.path.join(Config.SAVE_DIR, f"{base_name}_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        report['outputs'] = {
            'overlay': os.path.join(Config.SAVE_DIR, f"{base_name}_overlay.png"),
            'blended': os.path.join(Config.SAVE_DIR, f"{base_name}_model_first_detection.png"),
            'contours': os.path.join(Config.SAVE_DIR, f"{base_name}_contours_only.png"),
            'masks_dir': os.path.join(Config.SAVE_DIR, 'masks'),
            'report_json': report_path
        }
        return report

    def process_folder(self, source_dir: str):
        files = [f for f in os.listdir(source_dir) if is_image_file(f)]
        files.sort()
        if not files:
            logger.warning(f"No image files found in: {source_dir}")
            return {'processed': 0, 'results': []}

        logger.info(f"Found {len(files)} images in folder: {source_dir}")
        results = []
        for i, fname in enumerate(files, 1):
            path = os.path.join(source_dir, fname)
            logger.info(f"[{i}/{len(files)}] Processing {fname}")
            p = Process()
            out = p.process_image(path)
            results.append({'file': fname, 'result': out})
        return {'processed': len(files), 'results': results}

    def print_summary(self):
        print("\n" + "="*60)
        print("MODEL-FIRST DETECTION SUMMARY")
        print("="*60)
        print("YOLO MODEL DETECTIONS:")
        for class_name, count in self.model_summary.items():
            print(f"  {class_name}: {count}")

        print(f"\nPROCESSING RESULTS:")
        print(f"  Processed objects: {self.stats['processed_objects']}")
        print(f"  Valid Etioplasts: {self.stats['valid_etioplasts']}")
        print(f"  Rejected Etioplasts: {self.stats['rejected_etioplasts']}")
        print(f"    - No PLB: {self.stats['reject_no_plb']}")
        print(f"    - Incomplete (touch border): {self.stats['reject_incomplete']}")
        print(f"    - Not square-like: {self.stats['reject_not_square_like']}")
        print(f"  Valid organelles: {self.stats['valid_organelles']}")
        print(f"  Rejected organelles: {self.stats['rejected_organelles']}")

        if self.valid_etioplasts:
            print("\nVALID ETIOPLASTS DETAILS:")
            for i, et in enumerate(self.valid_etioplasts):
                oc = {}
                for ch in et.children:
                    if ch.is_valid:
                        oc[ch.get_name()] = oc.get(ch.get_name(), 0) + 1
                print(f"  Etioplast {i+1}: {dict(oc)}")
        print("="*60)
