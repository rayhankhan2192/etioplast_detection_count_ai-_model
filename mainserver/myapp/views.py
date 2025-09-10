# v1.0.1
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.text import get_valid_filename
from django.conf import settings
import uuid
from .config import logger, Config
from .process import Process, is_image_file
from .generativeai import get_generative_response

def _unique_dir(root: str, name: str) -> str:
    safe = get_valid_filename(name)
    base = os.path.join(root, safe)
    d = base
    i = 1
    while os.path.exists(d):
        d = f"{base}_{i}"
        i += 1
    os.makedirs(d, exist_ok=True)
    return d

def _as_media_url(path: str) -> str | None:
    try:
        rel = os.path.relpath(path, settings.MEDIA_ROOT)
    except ValueError:
        return None
    return settings.MEDIA_URL.rstrip('/') + '/' + rel.replace(os.sep, '/').lstrip('/')

def _as_media_url_abs(request, path: str) -> str | None:
    rel_url = _as_media_url(path)
    return request.build_absolute_uri(rel_url) if rel_url else None

def _get_px_per_um(request, fallback=None):
    """Read px_per_um from POST, else fallback, else Config.PX_PER_UM."""
    val = request.POST.get('px_per_um')
    if val not in (None, ''):
        try:
            return float(val)
        except ValueError:
            pass
    return float(fallback if fallback is not None else Config.PX_PER_UM)

@csrf_exempt
def analyze_summary_folder(request):
    """
    POST multipart/form-data:
      - file: single image (png/jpg/tif/webp)
      - files: multiple images (2–5 items)
      - px_per_um: optional (float)
    Returns JSON with results per image.
    """
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'detail': 'Use POST'}, status=405)

    try:
        px_per_um = _get_px_per_um(request)

        # ---- Prefer multi-upload if present ----
        uploaded_list = request.FILES.getlist('files')
        single = request.FILES.get('file')

        # Normalize into a list of (filename, fileobj)
        payload = []
        if uploaded_list:
            payload = [(f.name, f) for f in uploaded_list]
        elif single:
            payload = [(single.name, single)]
        else:
            return JsonResponse({'status': 'error', 'detail': 'No files uploaded. Use key "files" (multi) or "file" (single).'}, status=400)

        # Validate & cap count
        images = []
        for name, f in payload:
            if not is_image_file(name):
                continue
            images.append((name, f))

        if not images:
            return JsonResponse({'status': 'error', 'detail': 'No valid image files found.'}, status=400)

        if len(images) > 5:
            images = images[:5]  # enforce max 5

        # Temp session dir to save uploads before processing
        session_dir = os.path.join(settings.MEDIA_ROOT, 'uploads', uuid.uuid4().hex)
        os.makedirs(session_dir, exist_ok=True)

        results = []
        bundle_items = []

        for orig_name, fobj in images:
            # Save uploaded file to disk
            safe_name = orig_name.replace('\\', '/').split('/')[-1]
            save_path = os.path.join(session_dir, safe_name)
            with open(save_path, 'wb') as out:
                for chunk in fobj.chunks():
                    out.write(chunk)

            # Per-image output dir (for your pipeline artifacts)
            base_name, _ = os.path.splitext(safe_name)
            per_image_dir = _unique_dir(Config.SAVE_DIR, base_name)

            old_save_dir = Config.SAVE_DIR
            Config.SAVE_DIR = per_image_dir
            try:
                p = Process(px_per_um=px_per_um)
                report = p.process_image(save_path)
            finally:
                Config.SAVE_DIR = old_save_dir

            if not report:
                results.append({
                    'file': safe_name,
                    'save_dir_url': _as_media_url_abs(request, per_image_dir),
                    'error': 'Detection failed or no detections'
                })
                continue

            # Absolute URLs for outputs
            output_urls = {}
            if 'outputs' in report:
                for k, pth in report['outputs'].items():
                    output_urls[k] = _as_media_url_abs(request, pth)

            out_url = None
            if 'outputs' in report and 'blended' in report['outputs']:
                out_url = _as_media_url_abs(request, report['outputs']['blended'])

            # Optional: generative explanation
            explanation = None
            try:
                explanation = get_generative_response(report.get('analysis', {}))
            except Exception:
                logger.exception("Generative explanation failed")

            results.append({
                'file': safe_name,
                'analysis': report.get('analysis'),
                'scale_used': report.get('scale_used'),
                'output_image_url': out_url,
                'output_urls': output_urls,
                'explanation': explanation
            })

            # <<< NEW: remember where this image’s outputs were saved
            base_name, _ = os.path.splitext(safe_name)
            bundle_items.append({
                "per_image_dir": per_image_dir,
                "base_name": base_name,
            })

        mode = 'folder' if len(images) > 1 else 'file'
        # <<< NEW: build one master ZIP for all images in this request
        master_zip = _make_master_zip(request, bundle_items) if len(bundle_items) > 0 else None
        
        return JsonResponse({
            'status': 'ok', 
            'mode': mode, 
            'processed': len(results), 
            'results': results,
            # NEW:
            'master_zip_url': master_zip['master_url'] if master_zip else None,
            'master_zip_size_bytes': master_zip['size_bytes'] if master_zip else 0,
            #'master_zip_included': master_zip['included'] if master_zip else [],
            })

    except Exception as e:
        logger.exception("Folder processing failed")
        return JsonResponse({'status': 'error', 'detail': str(e)}, status=500)

# @csrf_exempt
# def analyze_summary_folder(request):
#     """
#     POST multipart/form-data:
#       - file: image file (png/jpg/tif/etc), OR
#       - source_dir: absolute server path to a folder of images
#       - px_per_um: optional (float)
#     Returns: JSON with output_urls + measurements summary + scale_used
#     """
#     if request.method != 'POST':
#         return JsonResponse({'status': 'error', 'detail': 'Use POST'}, status=405)
    
#     #Folder mode
#     source_dir = request.POST.get('source_dir')
#     if source_dir:
#         try:
#             if not os.path.getlist(source_dir):
#                 return JsonResponse({'status': 'error', 'detail': f'Not a directory: {source_dir}'}, status=400)

#             files = [f for f in os.listdir(source_dir) if is_image_file(f)]
#             files.sort()
#             if not files:
#                 return JsonResponse({'status': 'ok', 'mode': 'folder', 'processed': 0, 'results': []})

#             px_per_um = _get_px_per_um(request)
#             results = []

#             for fname in files:
#                 image_path = os.path.join(source_dir, fname)
#                 base_name, _ = os.path.splitext(fname)
#                 per_image_dir = _unique_dir(Config.SAVE_DIR, base_name)

#                 old_save_dir = Config.SAVE_DIR
#                 Config.SAVE_DIR = per_image_dir
#                 try:
#                     p = Process(px_per_um=px_per_um)
#                     report = p.process_image(image_path)
#                 finally:
#                     Config.SAVE_DIR = old_save_dir

#                 # Build per-file result
#                 if not report:
#                     results.append({
#                         'file': fname,
#                         'save_dir_url': _as_media_url_abs(request, per_image_dir),
#                         'error': 'Detection failed or no detections'
#                     })
#                     continue

#                 # Absolute URLs for outputs
#                 output_urls = {}
#                 if 'outputs' in report:
#                     for k, pth in report['outputs'].items():
#                         output_urls[k] = _as_media_url_abs(request, pth)

#                 # Choose blended as preview
#                 out_url = None
#                 if 'outputs' in report and 'blended' in report['outputs']:
#                     out_url = _as_media_url_abs(request, report['outputs']['blended'])

#                 # Optional: generative explanation (don’t break the API if it fails)
#                 explanation = None
#                 try:
#                     explanation = get_generative_response(report.get('analysis', {}))
#                 except Exception:
#                     logger.exception("Generative explanation failed")

#                 results.append({
#                     # 'file': fname,
#                     # 'save_dir_url': _as_media_url_abs(request, per_image_dir),
#                     'analysis': report.get('analysis'),
#                     'scale_used': report.get('scale_used'),
#                     'output_image_url': out_url,
#                     'output_urls': output_urls,
#                     'explanation': explanation
#                 })

#             return JsonResponse({'status': 'ok', 'mode': 'folder', 'processed': len(results), 'results': results})

#         except Exception as e:
#             logger.exception("Folder processing failed")
#             return JsonResponse({'status': 'error', 'detail': str(e)}, status=500)


@csrf_exempt
def analyze_summary_file(request):
    """
    Same processing as analyze_upload (single upload), but returns only the compact JSON:
    {
        "analysis": {...},
        "scale_used": "0.006944 µm/pixel",
        "output_image_url": "http://127.0.0.1:8000/media/detections/<folder>/<file>_model_first_detection.png"
    }
    """
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'detail': 'Use POST'}, status=405)

    if 'file' not in request.FILES:
        return JsonResponse({'status': 'error', 'detail': 'No file provided'}, status=400)

    up = request.FILES['file']
    try:
        px_per_um = _get_px_per_um(request)

        original_name = get_valid_filename(os.path.basename(up.name))
        base_name, _ = os.path.splitext(original_name)

        per_image_dir = _unique_dir(Config.SAVE_DIR, base_name)
        upload_path = os.path.join(per_image_dir, original_name)
        with open(upload_path, "wb+") as dst:
            for chunk in up.chunks():
                dst.write(chunk)

        old_save_dir = Config.SAVE_DIR
        Config.SAVE_DIR = per_image_dir
        try:
            p = Process(px_per_um=px_per_um)
            report = p.process_image(upload_path)
        finally:
            Config.SAVE_DIR = old_save_dir

        if report is None:
            return JsonResponse({'status': 'error', 'detail': 'Detection failed or no detections'}, status=200)
        
        output_urls = {}
        if 'outputs' in report:
            for k, pth in report['outputs'].items():
                output_urls[k] = _as_media_url_abs(request, pth)

        # choose the blended output as the preview
        out_url = None
        if 'outputs' in report and 'blended' in report['outputs']:
            out_url = _as_media_url_abs(request, report['outputs']['blended'])
        explanation = None
        try:
            explanation = get_generative_response(report.get('analysis', {}))
        except Exception:
            logger.exception("Generative explanation failed")
        
                # --- create a per-image ZIP (master zip for this image) ---
        zip_path = os.path.join(per_image_dir, f"{base_name}.zip")
        try:
            if os.path.exists(zip_path):
                os.remove(zip_path)
        except Exception:
            pass

        # Expected output paths
        blended_path  = report.get('outputs', {}).get('blended',  os.path.join(per_image_dir, f"{base_name}_model_first_detection.png"))
        contours_path = report.get('outputs', {}).get('contours', os.path.join(per_image_dir, f"{base_name}_contours_only.png"))
        overlay_path  = report.get('outputs', {}).get('overlay',  os.path.join(per_image_dir, f"{base_name}_overlay.png"))
        csv_path      = os.path.join(per_image_dir, f"{base_name}_measurements.csv")
        masks_dir     = os.path.join(per_image_dir, "masks")

        included = []
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # put top-level files at zip root
            for ap in (blended_path, contours_path, overlay_path, csv_path):
                if ap and os.path.isfile(ap):
                    zf.write(ap, os.path.basename(ap))
                    included.append(os.path.basename(ap))
            # add masks/ folder (if exists)
            if os.path.isdir(masks_dir):
                for root, _, files in os.walk(masks_dir):
                    for fn in files:
                        fullp = os.path.join(root, fn)
                        rel = os.path.relpath(fullp, masks_dir)
                        arc = os.path.join("masks", rel).replace("\\", "/")
                        zf.write(fullp, arc)
                        included.append(arc)

        master_zip_url = _as_media_url_abs(request, zip_path)
        master_zip_size = os.path.getsize(zip_path) if os.path.exists(zip_path) else 0

        return JsonResponse({
            "analysis": report.get('analysis'),
            "scale_used": report.get('scale_used'),
            "output_image_url": out_url,
            "output_urls": output_urls,
            "explanation": explanation,
            # New zip fields:
            "master_zip_url": master_zip_url,
            # "master_zip_size_bytes": master_zip_size,
            # "master_zip_included": included,
        })

    except Exception as e:
        logger.exception("Analyze summary failed")
        return JsonResponse({'status': 'error', 'detail': str(e)}, status=500)
    



# ...... ZIP -------

import zipfile
import datetime
# ... keep your other imports

def _zip_add_file(zf: zipfile.ZipFile, abs_path: str, arcname: str, included: list):
    if abs_path and os.path.isfile(abs_path):
        zf.write(abs_path, arcname)
        included.append(arcname)

def _make_master_zip(request, bundle_items: list[dict]) -> dict:
    """
    bundle_items: list of dicts with keys:
      - per_image_dir (abs path)
      - base_name (file base without ext)
    Will create a single master zip that contains per-image outputs under subfolders:
      <base>/<base>_model_first_detection.png
      <base>/<base>_contours_only.png
      <base>/<base>_overlay.png         # optional, remove if not needed
      <base>/<base>_measurements.csv
    """
    bundles_root = os.path.join(settings.MEDIA_ROOT, 'detections', '_batches')
    os.makedirs(bundles_root, exist_ok=True)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    master_name = f"master_{stamp}_{uuid.uuid4().hex[:6]}.zip"
    master_path = os.path.join(bundles_root, master_name)

    included = []
    with zipfile.ZipFile(master_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in bundle_items:
            per_image_dir = item["per_image_dir"]
            base_name     = item["base_name"]

            blended  = os.path.join(per_image_dir, f"{base_name}_model_first_detection.png")
            contours = os.path.join(per_image_dir, f"{base_name}_contours_only.png")
            overlay  = os.path.join(per_image_dir, f"{base_name}_overlay.png")         # optional
            csv_file = os.path.join(per_image_dir, f"{base_name}_measurements.csv")

            # put each image’s files under a subfolder named <base_name> inside the ZIP
            _zip_add_file(zf, blended,  f"{base_name}/{os.path.basename(blended)}", included)
            _zip_add_file(zf, contours, f"{base_name}/{os.path.basename(contours)}", included)
            _zip_add_file(zf, overlay,  f"{base_name}/{os.path.basename(overlay)}", included)   # optional
            _zip_add_file(zf, csv_file, f"{base_name}/{os.path.basename(csv_file)}", included)

    return {
        "master_path": master_path,
        "master_url": _as_media_url_abs(request, master_path),
        "included": included,
        "size_bytes": os.path.getsize(master_path) if os.path.exists(master_path) else 0,
    }
