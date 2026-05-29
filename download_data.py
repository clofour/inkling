from shared import RAW_DATA_DIR, CATEGORIES
import os.path as path
import requests

ENDPOINT = "https://storage.googleapis.com"
BUCKET_NAME = "quickdraw_dataset"
NUMPY_BITMAP_PATH = "full/raw"
API_URL = f"{ENDPOINT}/{BUCKET_NAME}/{NUMPY_BITMAP_PATH}"

for category in CATEGORIES:
    category_file_path = f"{RAW_DATA_DIR}/{category}.ndjson"
    if not path.exists(category_file_path):
        print(f"Downloading {category} data")

        url = f"{API_URL}/{category}.ndjson"
        request = requests.get(url, stream=True)

        with open(category_file_path, "wb") as file:
            for chunk in request.iter_content(chunk_size=1024):
                file.write(chunk)
