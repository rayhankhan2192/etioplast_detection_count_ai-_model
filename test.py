# import sys, requests

# API_URL = "http://127.0.0.1:8000/api/analyze-bulk/"  # <-- fixed
# image_path = sys.argv[1]
# px_per_um  = sys.argv[2]  # e.g. "850"

# with open(image_path, "rb") as f:
#     r = requests.post(API_URL, files={"file": (image_path.split("\\")[-1], f)},
#                       data={"px_per_um": px_per_um}, timeout=300)
# print("HTTP", r.status_code)
# print(r.text)

# test_folder.py
import sys, requests

API_URL = "http://127.0.0.1:8000/api/analyze/"  # folder mode endpoint
source_dir = sys.argv[1]                         # e.g. r"E:\...\CanDetect"
px_per_um  = sys.argv[2]                         # e.g. "850"

resp = requests.post(
    API_URL,
    data={
        "source_dir": source_dir,
        "px_per_um": px_per_um,
        # optional:
        # "max_files": "5",                       # limit to first N images (1..5)
        # "filenames": "img1.png,img2.jpg",       # process only these names in the folder
    },
    timeout=600,
)
print("HTTP", resp.status_code)
print(resp.text)
