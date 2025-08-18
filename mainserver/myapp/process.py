# v1.0.1
import os
import cv2
import json
from datetime import datetime

from .config import Config, CLASS_DEFINITIONS, logger
from .detection import run_yolo_detection, extract_model_summary, convert_yolo_to_objects
from .hierarchy_visualization import build_hierarchy, draw_visualization_improved, save_outputs_mask
from .geometry import write_measurements_csv, summarize_for_api

def is_image_file(fname: str) -> bool:
    ext = os.path.splitext(fname)[1].lower()
    return ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

class Process:
    def __init__(self, px_per_um: float = None):
        # scale
        self.px_per_um = float(px_per_um) if px_per_um else float(Config.PX_PER_UM)

        # runtime
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
        img = cv2.imread(image_path)
        if img is None:
            logger.error("Failed to read image")
            return None
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        yolo_result = run_yolo_detection(self, image_path)
        if yolo_result is None:
            return None

        if not extract_model_summary(self, yolo_result):
            logger.error("Failed to extract model summary")
            return None

        detected, image_shape = convert_yolo_to_objects(yolo_result, image_path, min_contour_area=Config.MIN_CONTOUR_AREA)
        self.detected_objects = detected
        self.image_shape = image_shape
        self.stats['processed_objects'] = len(self.detected_objects)

        build_hierarchy(self)

        overlay = draw_visualization_improved(self, img)
        blended = cv2.addWeighted(overlay, Config.ALPHA_BLEND, img, 1 - Config.ALPHA_BLEND, 0)

        img_name = os.path.basename(image_path)
        base_name = os.path.splitext(img_name)[0]

        # measurements CSV (per-etioplast)
        csv_path = write_measurements_csv(self.valid_etioplasts, self.px_per_um, Config.SAVE_DIR, base_name)

        # masks & outputs
        save_outputs_mask(self, base_name)
        blended_path  = os.path.join(Config.SAVE_DIR, f"{base_name}_model_first_detection.png")
        overlay_path  = os.path.join(Config.SAVE_DIR, f"{base_name}_overlay.png")
        contours_path = os.path.join(Config.SAVE_DIR, f"{base_name}_contours_only.png")
        cv2.imwrite(blended_path, blended)
        cv2.imwrite(overlay_path, overlay)
        contours_img = img.copy()
        for et in self.valid_etioplasts:
            cv2.drawContours(contours_img, [et.contour], -1, et.get_color(), 2)
            for ch in et.children:
                if ch.is_valid:
                    cv2.drawContours(contours_img, [ch.contour], -1, ch.get_color(), 2)
        cv2.imwrite(contours_path, contours_img)

        # basic printout
        self.print_summary()
        logger.info(f"✅ Done. Files -> {Config.SAVE_DIR}  |  Masks -> {Config.SAVE_DIR}/masks/")

        # report dict
        report = {
            'image': img_name,
            'timestamp': self.now_iso(),
            'model_summary': self.model_summary,
            'processing_stats': self.stats,
            'outputs': {
                'overlay': overlay_path,
                'blended': blended_path,
                'contours': contours_path,
                'masks_dir': os.path.join(Config.SAVE_DIR, 'masks'),
                'measurements_csv': csv_path,
            }
        }

        # compact analysis for API (and scale string)
        analysis, um_per_px_str = summarize_for_api(self.valid_etioplasts, self.px_per_um)
        report['analysis'] = analysis
        report['scale_used'] = um_per_px_str
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
            p = Process(px_per_um=self.px_per_um)
            # NOTE: caller may change SAVE_DIR before calling this
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
