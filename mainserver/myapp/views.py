# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.parsers import MultiPartParser, FormParser
# from django.conf import settings
# from django.core.files.storage import default_storage
# from ultralytics import YOLO
# import numpy as np
# import cv2
# import os
# import matplotlib.pyplot as plt
# from dotenv import load_dotenv


# load_dotenv()
# model_path = os.getenv("MODEL_PATH")
# model = YOLO(model_path)

# class SegmentView(APIView):
#     parser_classes = (MultiPartParser, FormParser)
#     class_names = ["Etioplast", "PLBs", "Class_2", "Prothylakoids"]
#     colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]

#     def post(self, request, format=None):
#         image_file = request.FILES.get("image")
#         if not image_file:
#             return Response({"error": "No image provided"}, status=400)

#         # Save uploaded file
#         file_path = default_storage.save("tmp/" + image_file.name, image_file)
#         abs_path = os.path.join(settings.MEDIA_ROOT, file_path)

#         image = cv2.imread(abs_path)
#         results = self.model(image)

#         mask_image = image.copy()
#         masks = results[0].masks.xy
#         classes = results[0].boxes.cls.cpu().numpy().astype(int)

#         for seg, cls in zip(masks, classes):
#             polygon = np.array(seg, dtype=np.int32)
#             color = self.colors[cls % len(self.colors)]
#             label = self.class_names[cls % len(self.class_names)]

#             cv2.polylines(mask_image, [polygon], isClosed=True, color=color, thickness=12)
#             center = np.mean(polygon, axis=0).astype(int)
#             text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
#             text_x = center[0] - text_size[0] // 2
#             text_y = center[1] + text_size[1] // 2

#             cv2.putText(mask_image, label, (text_x, text_y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6, cv2.LINE_AA)
#             cv2.putText(mask_image, label, (text_x, text_y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)

#         # Save output
#         output_filename = "segmented_" + os.path.basename(file_path)
#         output_path = os.path.join(settings.MEDIA_ROOT, "outputs", output_filename)
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         cv2.imwrite(output_path, mask_image)

#         return Response({"message": "Segmentation completed", "output_url": f"/media/outputs/{output_filename}"})




# views.py

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

from dotenv import load_dotenv


load_dotenv()
model_path = os.getenv("MODEL_PATH")


class YOLOSegmentAnalyzeView(APIView):
    model = YOLO(model_path)
    parser_classes = (MultiPartParser, FormParser)
    class_names = ['Etioplast', 'PLB', 'Plastoglobule', 'Prothylakoid']
    colors = [
        (255, 0, 0),     # Etioplast
        (0, 255, 0),     # PLBs
        (0, 0, 255),     # Plastoglobule
        (0, 255, 255),   # Prothylakoids
    ]

    def post(self, request):
        image_file = request.FILES.get("image")
        print("img:",image_file)
        if not image_file:
            return Response({"error": "No image uploaded."}, status=400)

        # Save uploaded image
        upload_path = default_storage.save(f"uploads/{uuid.uuid4()}_{image_file.name}", image_file)
        image_full_path = os.path.join(settings.MEDIA_ROOT, upload_path)
        image = cv2.imread(image_full_path)

        # Run YOLOv8
        results = self.model(image)
        masks = results[0].masks.xy
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        # Analyze results
        analyzer = SegmentationAnalyzer(
            masks=masks,
            classes=classes,
            class_names=self.class_names,
            scale_bar_microns=0.5,
            scale_bar_pixels=72
        )
        result_data, micron_per_pixel = analyzer.analyze()

        # Draw detection output
        drawn_image = image.copy()
        for seg, cls_idx in zip(masks, classes):
            polygon = np.array(seg, dtype=np.int32)
            label = self.class_names[cls_idx]
            color = self.colors[cls_idx % len(self.colors)]

            cv2.polylines(drawn_image, [polygon], isClosed=True, color=color, thickness=8)

            # Center of polygon
            center = np.mean(polygon, axis=0).astype(int)
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            tx, ty = center[0] - text_size[0] // 2, center[1] + text_size[1] // 2

            cv2.putText(drawn_image, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)  # black outline
            cv2.putText(drawn_image, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)      # label text

        # Save output image
        output_filename = f"output_{uuid.uuid4().hex}.png"
        output_path = os.path.join(settings.MEDIA_ROOT, "outputs", output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, drawn_image)

        # Public image URL (optional: use settings.MEDIA_URL in prod)
        output_url = request.build_absolute_uri(f"/media/outputs/{output_filename}")

        return Response({
            "analysis": result_data,
            "scale_used": f"{micron_per_pixel} µm/pixel",
            "output_image_url": output_url
        })
