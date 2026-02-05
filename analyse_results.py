import os
from azure.storage.blob import BlobServiceClient
import os
from azure.storage.blob import BlobServiceClient
CONNECTION_STRING = (
    "DefaultEndpointsProtocol=https;"
    "AccountName=rightmoveukstorage;"
    "AccountKey=c5XeBrrI3x80immhsG7KWTkD8oF+OAyw6sOQRB1rCYPkWqUafDquu+tqLFDX/1K0DlULEd1gw/fC+AStTrIYdA==;"
    "EndpointSuffix=core.windows.net"
)
import os
from azure.storage.blob import BlobServiceClient

CONTAINER_NAME = "glasgow-rent"
PREFIX = "02-01-2026/rightmove_images/"
LOCAL_BASE_DIR = "./download"

blob_service = BlobServiceClient.from_connection_string(CONNECTION_STRING)
container_client = blob_service.get_container_client(CONTAINER_NAME)

print(f"📥 Downloading blobs from {CONTAINER_NAME}/{PREFIX}")

count = 0
for blob in container_client.list_blobs(name_starts_with=PREFIX):
    if blob.name.endswith("/"):
        continue

    local_path = os.path.join(LOCAL_BASE_DIR, blob.name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    with open(local_path, "wb") as f:
        f.write(container_client.get_blob_client(blob.name).download_blob().readall())

    print(f"✅ {blob.name}")
    count += 1

print(f"🎉 Downloaded {count} files")
