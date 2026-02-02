import os
import time
import json
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict

# =========================================================
# CONFIGURATION
# =========================================================
CONTAINERS = ["Glasgow-rent", "Glasgow-sale", "London-rent", "London-sale"]
WAIT_SECONDS = 3600  # 1 hour wait before processing
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
STORAGE_CONN_STR = os.environ["AZURE_STORAGE_CONNECTION_STRING"]

# =========================================================
# DEVICE & CLIP SETUP
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {DEVICE}")

print("📦 Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval().half()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("✅ CLIP model loaded")

# Labels
object_labels = [
    "a photo of a single bed", "a photo of a double bed", "a photo of a bed",
    "a photo of a bathroom basin", "a photo of a bath", "a photo of a refrigerator",
    "a photo of a duvet", "a photo of a pillow", "a photo of a television",
    "a photo of a wardrobe", "a photo of a Single Ended Bath", "a photo of a stove vent hood",
    "a photo of a sofa", "a photo of a shower head", "a photo of a heater radiator",
    "a photo of a kitchen stove", "a photo of a kitchen cabinets", "a photo of a toilet seat",
    "a photo of a kitchen sink", "a photo of a dinning table and chairs", "a photo of stairs",
    "a photo of a garden", "a photo of a car", "a photo of a tree", "a photo of a cabinets",
    "a photo of a door", "a photo of a tiles", "a photo of a carpet", "a photo of a wood floor",
    "a photo of an electric socket", "a photo of an alarm", "a photo of a camera",
    "a photo of a celing light", "a photo of a shop", "a photo of a window",
    "a photo of curtains", "a photo of an unknown room", "a photo of a floorplan"
]

cleanliness_prompts = [
    "This is a very clean room, everything is tidy and organized.",
    "This is a somewhat messy room, there are a few items scattered.",
    "This is a dirty room, cluttered and untidy with visible mess."
]

# =========================================================
# HELPER FUNCTIONS
# =========================================================
blob_service = BlobServiceClient.from_connection_string(STORAGE_CONN_STR)

@torch.no_grad()
def classify_image(image):
    obj_inputs = processor(text=object_labels, images=image, return_tensors="pt", padding=True).to(DEVICE)
    clean_inputs = processor(text=cleanliness_prompts, images=image, return_tensors="pt", padding=True).to(DEVICE)

    obj_outputs = model(**obj_inputs)
    clean_outputs = model(**clean_inputs)

    obj_probs = obj_outputs.logits_per_image.softmax(dim=1)
    obj_pred = torch.argmax(obj_probs, dim=1).item()
    obj_conf = float(obj_probs[0, obj_pred].item())
    best_label = object_labels[obj_pred] if obj_conf >= 0.4 else "unknown"

    clean_probs = clean_outputs.logits_per_image.softmax(dim=1)
    clean_pred = torch.argmax(clean_probs, dim=1).item()
    cleanliness_level = clean_pred + 1
    cleanliness_reason = cleanliness_prompts[clean_pred]

    return best_label, obj_conf, cleanliness_level, cleanliness_reason

def list_date_folders(container_client):
    """Return all date folders in the container."""
    blobs = container_client.list_blobs(name_starts_with="")
    folders = set()
    for b in blobs:
        parts = b.name.split("/")
        if len(parts) >= 2:  # e.g., "02-02-2026/rightmove_images/..."
            folders.add(parts[0])
    return sorted(folders)

def process_property(container_client, date_folder, property_id):
    prefix = f"{date_folder}/rightmove_images/{property_id}/"
    blobs = [b.name for b in container_client.list_blobs(name_starts_with=prefix)
             if b.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]

    for i in range(0, len(blobs), BATCH_SIZE):
        batch = blobs[i:i+BATCH_SIZE]
        images = []
        paths = []
        for blob_name in batch:
            json_path = blob_name.rsplit(".", 1)[0] + ".json"
            blob_client = container_client.get_blob_client(blob_name)
            # Skip if JSON exists
            if container_client.get_blob_client(json_path).exists():
                continue
            try:
                data = blob_client.download_blob().readall()
                image = Image.open(io.BytesIO(data)).convert("RGB")
                images.append(image)
                paths.append(blob_name)
            except Exception as e:
                print(f"⚠️ Failed to read {blob_name}: {e}")
                continue

        # Classify batch
        for j, img in enumerate(images):
            label, conf, clean_level, reason = classify_image(img)
            json_path = paths[j].rsplit(".",1)[0]+".json"
            result = {
                "image_name": paths[j].split("/")[-1],
                "primary_label": label,
                "primary_confidence": conf,
                "cleanliness_level": clean_level,
                "cleanliness_reason": reason
            }
            container_client.get_blob_client(json_path).upload_blob(json.dumps(result, indent=4), overwrite=True)
            print(f"✅ Processed {paths[j]} -> {json_path}")

def process_container(container_name):
    print(f"\n📦 Processing container: {container_name}")
    container_client = blob_service.get_container_client(container_name)
    date_folders = list_date_folders(container_client)
    if not date_folders:
        print("⚠️ No date folders found")
        return
    latest_date = date_folders[-1]
    print(f"⏱ Waiting {WAIT_SECONDS/60:.0f} minutes for uploads in {latest_date}...")
    time.sleep(WAIT_SECONDS)

    # List properties
    property_prefix = f"{latest_date}/rightmove_images/"
    blobs = container_client.list_blobs(name_starts_with=property_prefix)
    property_ids = set()
    for b in blobs:
        parts = b.name.split("/")
        if len(parts) >= 4:
            property_ids.add(parts[3])

    print(f"🏠 Found {len(property_ids)} properties in {latest_date}")
    for prop_id in property_ids:
        process_property(container_client, latest_date, prop_id)

# =========================================================
# ENTRYPOINT
# =========================================================
def main():
    for container in CONTAINERS:
        process_container(container)
    print("\n🎉 All containers processed!")

if __name__=="__main__":
    main()
