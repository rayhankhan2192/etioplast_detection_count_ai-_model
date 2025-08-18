# v1.0.1
from django.urls import path
from .views import analyze_summary_file, analyze_summary_folder

urlpatterns = [
    path('analyze-folder/', analyze_summary_folder, name='analyze-folder'),
    path('analyze-file/', analyze_summary_file, name='analyze-filee'),
]

# # if settings.DEBUG:
# #     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


