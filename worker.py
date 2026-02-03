import os
import json
import io
import time
import torch
from PIL import Image
from azure.storage.queue import QueueClient
from azure.storage.blob import BlobServiceClient
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel
from PIL import ImageDraw, ImageFont
import os
import io
import json
import time
from collections import defaultdict
from urllib.parse import urlparse
from azure.storage.queue import QueueClient

import torch
from PIL import Image, ImageDraw, ImageFont
from azure.storage.queue import QueueClient
from azure.storage.blob import BlobServiceClient
from transformers import CLIPProcessor, CLIPModel
import os
from azure.storage.queue import QueueClient
from azure.storage.blob import BlobServiceClient


import os
from azure.storage.queue import QueueClient
from azure.storage.blob import BlobServiceClient
# =========================================================
# 3. LABELS & CLEANLINESS
# =========================================================
object_labels = [
    "a photo of a single bed", "a photo of a double bed", "a photo of a bed",
    "a photo of a bathroom basin", "a photo of a bath", "a photo of a refrigerator",
    "a photo of a bathroom", "a photo of a duvet", "a photo of a pillow",
    "a photo of a television", "a photo of a wardrobe", "a photo of a single ended bath",
    "a photo of a stove vent hood", "a photo of a sofa", "a photo of a shower head",
    "a photo of a heater radiator", "a photo of a kitchen stove", "a photo of a kitchen",
    "a photo of a toilet seat", "a photo of a kitchen sink", "a photo of a dining table and chairs",
    "a photo of stairs", "a photo of a garden", "a photo of a car", "a photo of a tree",
    "a photo of cabinets", "a photo of a door", "a photo of tiles", "a photo of a carpet",
    "a photo of a wood floor", "a photo of an electric socket", "a photo of an alarm",
    "a photo of a camera", "a photo of a ceiling light", "a photo of a shop",
    "a photo of a window", "a photo of curtains", "a photo of an unknown room",
    "a photo of a living room", "a photo of a floorplan"
]

cleanliness_prompts = [
    "This is a very clean room, everything is tidy and organized.",
    "This is a somewhat messy room, there are a few items scattered.",
    "This is a dirty room, cluttered and untidy with visible mess."
]

STORAGE_CONN_STR = os.environ["AZURE_STORAGE_CONNECTION_STRING"]

queue_client = QueueClient.from_connection_string(STORAGE_CONN_STR, "rightmove-images-queue")
blob_service = BlobServiceClient.from_connection_string(STORAGE_CONN_STR)

# =========================================================
# 1. GPU / DEVICE SETUP
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {DEVICE}")
# =========================================================
# 2. LOAD CLIP MODEL
# =========================================================
print("📦 Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
model = model.half()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print("✅ CLIP loaded")
# =========================================================
# 5. HELPER FUNCTIONS
# =========================================================
def parse_blob_url(blob_url: str):
    parsed = urlparse(blob_url)
    path = parsed.path.lstrip("/")
    container, blob = path.split("/", 1)
    return container, blob

@torch.no_grad()
def classify_room_and_cleanliness(image, object_conf_threshold=0.4):
    """Classify image for object and cleanliness"""
    obj_inputs = processor(text=object_labels, images=[image], return_tensors="pt", padding=True).to(DEVICE)
    clean_inputs = processor(text=cleanliness_prompts, images=[image], return_tensors="pt", padding=True).to(DEVICE)

    obj_outputs = model(**obj_inputs)
    clean_outputs = model(**clean_inputs)

    obj_probs = obj_outputs.logits_per_image.softmax(dim=1)
    obj_pred = torch.argmax(obj_probs, dim=1).item()
    obj_conf = float(obj_probs[0, obj_pred].item())
    best_label = object_labels[obj_pred] if obj_conf >= object_conf_threshold else "unknown"

    clean_probs = clean_outputs.logits_per_image.softmax(dim=1)
    clean_pred = torch.argmax(clean_probs, dim=1).item()
    cleanliness_level = clean_pred + 1
    cleanliness_reason = cleanliness_prompts[clean_pred]

    return best_label, obj_conf, cleanliness_level, cleanliness_reason

def classify_room(image):
    inputs = processor(text=object_labels, images=[image], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return object_labels[pred], float(probs[0, pred].item())

def classify_room_from_objects_multi_label(objects_dict):
    detected = {k.lower().strip(): v for k, v in objects_dict.items()}
    categories = {
        "floorplan": ["a floorplan"],
        "bedroom": ["a bed", "a single bed", "a double bed", "a duvet", "a pillow", "a wardrobe", "a bedroom"],
        "bathroom": ["a bath", "a bathroom basin", "a shower head", "a toilet seat", "a single ended bath", "a bathroom"],
        "living/sitting room": ["a sofa", "a dining table and chairs", "a living room"],
        "outdoor": ["a garden", "a tree", "a car"],
        "kitchen": ["a refrigerator", "a kitchen stove", "a stove vent hood", "a kitchen", "a kitchen sink"]
    }

    matched = []
    for room, objs in categories.items():
        if any(obj.lower().strip() in detected for obj in objs):
            matched.append(room)

    if not matched:
        return "unknown"
    return " & ".join(matched) if len(matched) > 1 else matched[0]

def detect_objects_multiscale_annotate(image_input, n_scales=4, overlap_ratio=0.1, confidence_threshold=0.7):
    """Detect objects in multiple scales and annotate the image"""
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise ValueError("Input must be a file path or PIL.Image")

    W, H = image.size
    min_dim = min(W, H)
    scales = sorted(set(int(min_dim / (2 ** i)) for i in range(n_scales, 0, -1) if int(min_dim / (2 ** i)) > 60))

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()

    detections = []
    label_counts = defaultdict(int)

    for crop_size in scales:
        stride_x = int(crop_size * (1 - overlap_ratio))
        stride_y = int(crop_size * (1 - overlap_ratio))
        x_steps = list(range(0, W - crop_size, stride_x)) + [W - crop_size]
        y_steps = list(range(0, H - crop_size, stride_y)) + [H - crop_size]

        for top in y_steps:
            for left in x_steps:
                crop = image.crop((left, top, left + crop_size, top + crop_size))
                label, conf = classify_room(crop)
                if conf > confidence_threshold and label != "a photo of an unknown room":
                    cx = left + crop_size // 2
                    cy = top + crop_size // 2
                    label_clean = label.split("a photo of ")[-1].strip()
                    detections.append((cx, cy, label_clean, conf))
                    label_counts[label_clean] += 1

    # Draw annotations
    for cx, cy, label_clean, conf in detections:
        text = f"{label_clean} ({conf*100:.1f}%)"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.rectangle([(cx-tw//2-6, cy-th//2-6), (cx+tw//2+6, cy+th//2+6)], fill=(0,0,0,160))
        draw.text((cx-tw//2, cy-th//2), text, fill="white", font=font)

    return annotated, dict(label_counts)

def process_single_blob(container_name, blob_name):
    blob_client = blob_service.get_blob_client(container_name, blob_name)
    json_name = blob_name.rsplit(".", 1)[0] + ".json"
    json_client = blob_service.get_blob_client(container_name, json_name)

    # Skip if JSON exists
    try:
        json_client.get_blob_properties()
        print(f"Skipping {blob_name}, JSON exists")
        return
    except:
        pass

    # Download blob
    try:
        data = blob_client.download_blob().readall()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        print(f"⚠️ Could not open blob as image {blob_name}: {e}")
        return

    # Process
    try:
        annotated_img, labels_dict = detect_objects_multiscale_annotate(image)
    except Exception as e:
        print(f"⚠️ Error in object detection for {blob_name}: {e}")
        labels_dict = {}

    try:
        label, conf, clean_level, reason = classify_room_and_cleanliness(image)
    except Exception as e:
        print(f"⚠️ Error classifying {blob_name}: {e}")
        label, conf, clean_level, reason = "unknown", 1.0, "unknown", "unknown"

    try:
        room_type = classify_room_from_objects_multi_label(labels_dict)
    except Exception as e:
        print(f"⚠️ Error classifying room type for {blob_name}: {e}")
        room_type = "unknown"

    result = {
        "image_name": blob_name,
        "objects_detected": labels_dict,
        "primary_label": label,
        "primary_confidence": conf,
        "cleanliness_level": clean_level,
        "cleanliness_reason": reason,
        "room_type": room_type
    }

    try:
        json_client.upload_blob(json.dumps(result, indent=4), overwrite=True)
        print(f"✅ Processed {blob_name}")
    except Exception as e:
        print(f"⚠️ Failed to upload JSON for {blob_name}: {e}")

def safe_delete(msg):
    try:
        queue_client.delete_message(msg.id, msg.pop_receipt)
    except Exception as e:
        print(f"⚠️ Could not delete message: {e}")

# =========================================================
# 6. MAIN WORKER LOOP
# =========================================================
def main():
    print("📥 Worker started, polling queue...")
    image_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    while True:
        try:
            messages = list(queue_client.receive_messages(messages_per_page=32))
        except Exception as e:
            print(f"⚠️ Error receiving messages: {e}")
            time.sleep(5)
            continue

        if not messages:
            time.sleep(1)
            continue

        for msg in messages:
            try:
                data = json.loads(msg.content)
                blob_url = data.get("url")
                container_name = data.get("container")
                prefix = data.get("prefix")

                if blob_url:
                    container_name, blob_name = parse_blob_url(blob_url)
                    process_single_blob(container_name, blob_name)
                elif container_name and prefix:
                    container_client = blob_service.get_container_client(container_name)
                    for blob in container_client.list_blobs(name_starts_with=prefix):
                        if not blob.name.lower().endswith(image_exts):
                            continue
                        process_single_blob(container_name, blob.name)
                else:
                    print(f"⚠️ Message missing URL and container/prefix: {msg.content}")

            except Exception as e:
                print(f"⚠️ Error processing message: {e}")

            safe_delete(msg)
        time.sleep(1)

if __name__ == "__main__":
    main()