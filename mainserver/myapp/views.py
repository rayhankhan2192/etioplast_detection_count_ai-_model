# detector/views.py
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.text import get_valid_filename
from django.conf import settings

from .config import logger, Config
from .process import Process, is_image_file

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
    """
    Convert an absolute filesystem path under MEDIA_ROOT to a MEDIA_URL.
    Returns None if the path is not inside MEDIA_ROOT.
    """
    try:
        rel = os.path.relpath(path, settings.MEDIA_ROOT)
    except ValueError:
        return None
    # Always forward slashes for URLs
    rel_url = rel.replace(os.sep, '/')
    return settings.MEDIA_URL.rstrip('/') + '/' + rel_url.lstrip('/')

@csrf_exempt
def analyze_upload(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'detail': 'Use POST'}, status=405)

    # Folder mode
    source_dir = request.POST.get('source_dir')
    if source_dir:
        try:
            if not os.path.isdir(source_dir):
                return JsonResponse({'status': 'error', 'detail': f'Not a directory: {source_dir}'}, status=400)

            files = [f for f in os.listdir(source_dir) if is_image_file(f)]
            files.sort()
            if not files:
                return JsonResponse({'status': 'ok', 'mode': 'folder', 'processed': 0, 'results': []})

            results = []
            for fname in files:
                image_path = os.path.join(source_dir, fname)
                base_name, _ = os.path.splitext(fname)
                per_image_dir = _unique_dir(Config.SAVE_DIR, base_name)

                old_save_dir = Config.SAVE_DIR
                Config.SAVE_DIR = per_image_dir
                try:
                    p = Process()
                    report = p.process_image(image_path)
                finally:
                    Config.SAVE_DIR = old_save_dir

                # Convert report outputs to URLs
                output_urls = {}
                if report and 'outputs' in report:
                    for k, pth in report['outputs'].items():
                        output_urls[k] = _as_media_url(pth)

                results.append({
                    'file': fname,
                    'save_dir': per_image_dir,
                    'save_dir_url': _as_media_url(per_image_dir),  # may not list; useful as base
                    'report': report,
                    'output_urls': output_urls
                })

            return JsonResponse({'status': 'ok', 'mode': 'folder', 'processed': len(results), 'results': results})
        except Exception as e:
            logger.exception("Folder processing failed")
            return JsonResponse({'status': 'error', 'detail': str(e)}, status=500)

    #Single upload
    if 'file' not in request.FILES:
        return JsonResponse({'status': 'error', 'detail': 'No file provided'}, status=400)

    up = request.FILES['file']
    try:
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
            p = Process()
            report = p.process_image(upload_path)
        finally:
            Config.SAVE_DIR = old_save_dir

        if report is None:
            return JsonResponse({'status': 'error', 'detail': 'Detection failed or no detections'}, status=200)

        # Build URLs for client
        output_urls = {}
        if 'outputs' in report:
            for k, pth in report['outputs'].items():
                output_urls[k] = _as_media_url(pth)

        return JsonResponse({
            'status': 'ok',
            'mode': 'single',
            'upload_path': upload_path,
            'upload_url': _as_media_url(upload_path),
            'save_dir': per_image_dir,
            'save_dir_url': _as_media_url(per_image_dir),  # base folder URL
            'report': report,
            'output_urls': output_urls
        })
    except Exception as e:
        logger.exception("Analyze failed")
        return JsonResponse({'status': 'error', 'detail': str(e)}, status=500)
