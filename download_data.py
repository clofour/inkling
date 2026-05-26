from shared import DATA_DIR, CATEGORIES
import os.path as path
import requests

ENDPOINT = "https://storage.googleapis.com"
BUCKET_NAME = "quickdraw_dataset"
NUMPY_BITMAP_PATH = "full/numpy_bitmap"
API_URL = f"{ENDPOINT}/{BUCKET_NAME}/{NUMPY_BITMAP_PATH}"

for category in CATEGORIES:
    category_file_path = f"{DATA_DIR}/{category}.npy"
    if not path.exists(category_file_path):
        print(f"Downloading {category} data")

        url = f"{API_URL}/{category}.npy"
        request = requests.get(url, stream=True)

        with open(category_file_path, "wb") as file:
            for chunk in request.iter_content(chunk_size=1024):
                file.write(chunk)
