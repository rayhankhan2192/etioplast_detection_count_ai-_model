import os
import tempfile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .config import logger
from .process import Process

@csrf_exempt
def analyze_upload(request):
    """
    POST multipart/form-data:
      - file: image file (png/jpg/tif/etc), OR
      - source_dir: absolute server path to a folder of images (optional alternative)

    Returns JSON report + output file paths.
    """
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'detail': 'Use POST'}, status=405)

    # Option 1: process a server folder path (advanced)
    source_dir = request.POST.get('source_dir')
    if source_dir:
        try:
            p = Process()
            result = p.process_folder(source_dir)
            return JsonResponse({'status': 'ok', 'mode': 'folder', 'result': result})
        except Exception as e:
            logger.exception("Folder processing failed")
            return JsonResponse({'status': 'error', 'detail': str(e)}, status=500)

    # Option 2: uploaded single image
    if 'file' not in request.FILES:
        return JsonResponse({'status': 'error', 'detail': 'No file provided'}, status=400)

    up = request.FILES['file']
    try:
        # Save upload to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as tmp:
            for chunk in up.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        p = Process()
        report = p.process_image(tmp_path)
        if report is None:
            return JsonResponse({'status': 'error', 'detail': 'Detection failed or no detections'}, status=200)

        return JsonResponse({'status': 'ok', 'mode': 'single', 'report': report})
    except Exception as e:
        logger.exception("Analyze failed")
        return JsonResponse({'status': 'error', 'detail': str(e)}, status=500)
    finally:
        # cleanup temp file
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
