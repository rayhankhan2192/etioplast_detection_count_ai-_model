from django.urls import path
from .views import analyze_upload, analyze_summary

urlpatterns = [
    path('analyze/', analyze_upload, name='analyze_upload'),
    path('detect-analyze/', analyze_upload, name='analyze_upload_alias'),
    path('analyze-summary/', analyze_summary, name='analyze_summary'),
]

# # if settings.DEBUG:
# #     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


