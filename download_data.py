"""
Download script for MaleCNS v1.0 connectome datasets from Google Cloud Storage.
"""

import os
import urllib.request

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = [
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
]

def download_data():
    os.makedirs("data", exist_ok=True)
    for filename in FILES:
        url = BASE_URL + filename
        dest = os.path.join("data", filename)
        if os.path.exists(dest):
            print(f"[EXISTS] {dest} ({os.path.getsize(dest)/(1024*1024):.2f} MB)")
            continue
        print(f"Downloading {url} -> {dest}...")
        urllib.request.urlretrieve(url, dest)
        print(f"[DONE] {dest}")

if __name__ == "__main__":
    download_data()
