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
# 4. HELPER FUNCTIONS
# =========================================================
@torch.no_grad()
def classify_room_and_cleanliness(image, object_conf_threshold=0.4):
    obj_inputs = processor(text=object_labels, images=image,
                           return_tensors="pt", padding=True).to(DEVICE)
    clean_inputs = processor(text=cleanliness_prompts, images=image,
                             return_tensors="pt", padding=True).to(DEVICE)

    obj_outputs = model(**obj_inputs)
    clean_outputs = model(**clean_inputs)

    # Object detection
    obj_probs = obj_outputs.logits_per_image.softmax(dim=1)
    obj_pred = torch.argmax(obj_probs, dim=1).item()
    obj_conf = float(obj_probs[0, obj_pred].item())
    best_label = object_labels[obj_pred] if obj_conf >= object_conf_threshold else "unknown"

    # Cleanliness detection
    clean_probs = clean_outputs.logits_per_image.softmax(dim=1)
    clean_pred = torch.argmax(clean_probs, dim=1).item()
    cleanliness_level = clean_pred + 1
    cleanliness_reason = cleanliness_prompts[clean_pred]

    return best_label, obj_conf, cleanliness_level, cleanliness_reason

def classify_room(image):
    inputs = processor(text=object_labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return object_labels[pred], float(probs[0, pred].item()), inputs

def classify_room_from_objects(objects_dict):
    """Infer room type from detected objects."""
    detected = {k.lower(): v for k, v in objects_dict.items()}

    # Rules
    kitchen_objects = ["a refrigerator", "a kitchen stove", "a stove vent hood", "a kitchen cabinets", "a kitchen sink"]
    bedroom_objects = ["a bed", "a single bed", "a double bed", "a duvet", "a pillow", "a wardrobe"]
    bathroom_objects = ["a bath", "a bathroom basin", "a shower head", "a toilet seat", "a single ended bath"]
    livingroom_objects = ["a sofa", "a dinning table and chairs"]#, "a heater radiator", "a television"
    outdoor_objects = ["a garden", "a tree", "a car"]
    floorplan_objects = ["a floorplan"]

    def has_any(category):
        return any(obj in detected for obj in category)
    if has_any(floorplan_objects): return "floorplan"
    if has_any(kitchen_objects): return "kitchen"
    if has_any(bedroom_objects): return "bedroom"
    if has_any(bathroom_objects): return "bathroom"
    if has_any(livingroom_objects): return "living room"
    if has_any(outdoor_objects): return "outdoor"
    return "unknown"

def crop_image_with_overlap(image_path, crop_size=500, overlap_ratio=0.7, include_edges=True):
    """Crop an image into overlapping patches."""
    image = Image.open(image_path)
    width, height = image.size
    stride = int(crop_size * (1 - overlap_ratio))
    x_steps = list(range(0, width - crop_size + 1, stride))
    y_steps = list(range(0, height - crop_size + 1, stride))
    if include_edges:
        if x_steps[-1] + crop_size < width: x_steps.append(width - crop_size)
        if y_steps[-1] + crop_size < height: y_steps.append(height - crop_size)
    crops = [image.crop((left, top, left + crop_size, top + crop_size))
             for top in y_steps for left in x_steps]
    return crops


def detect_objects_multiscale_annotate(
    image_path,
    n_scales=1,
    overlap_ratio=0.1,
    confidence_threshold=0.7,
    crop_batch_size=16  # 🔥 adjust: 16 (T4), 32 (A10)
):
    """Multi-scale object detection with batched CLIP inference."""
    image = image_path#Image.open(image_path).convert("RGB")
    W, H = image.size
    min_dim = min(W, H)

    scales = sorted(set([
        int(min_dim / (2 ** i))
        for i in range(n_scales, 0, -1)
        if int(min_dim / (2 ** i)) > 60
    ]))

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

        x_steps = list(range(0, W - crop_size, stride_x))
        y_steps = list(range(0, H - crop_size, stride_y))
        if x_steps[-1] + crop_size < W:
            x_steps.append(W - crop_size)
        if y_steps[-1] + crop_size < H:
            y_steps.append(H - crop_size)

        crops = []
        positions = []

        for top in y_steps:
            for left in x_steps:
                crops.append(image.crop((left, top, left + crop_size, top + crop_size)))
                positions.append((left, top))

        # 🔥 Batched CLIP inference
        for i in range(0, len(crops), crop_batch_size):
            batch_imgs = crops[i:i + crop_batch_size]
            batch_pos = positions[i:i + crop_batch_size]

            inputs = processor(
                text=object_labels,
                images=batch_imgs,
                return_tensors="pt",
                padding=True
            ).to(DEVICE)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)

            for j, (left, top) in enumerate(batch_pos):
                conf, idx = torch.max(probs[j], dim=0)
                conf = float(conf)
                label = object_labels[idx]

                if conf > confidence_threshold and label != "a photo of an unknown room":
                    cx = left + crop_size // 2
                    cy = top + crop_size // 2
                    label_clean = label.split("a photo of")[-1].strip()

                    detections.append((cx, cy, label_clean, conf))
                    label_counts[label_clean] += 1

    # Annotation (unchanged)
    for (cx, cy, label_clean, conf) in detections:
        text = f"{label_clean} ({conf*100:.1f}%)"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        draw.rectangle(
            [(cx - tw//2 - 6, cy - th//2 - 6),
             (cx + tw//2 + 6, cy + th//2 + 6)],
            fill=(0, 0, 0, 160)
        )
        draw.text(
            (cx - tw//2, cy - th//2),
            text,
            fill="white",
            font=font
        )

    return annotated, dict(label_counts)


# =========================================================
# 3. LABELS AND CLEANLINESS PROMPTS
# =========================================================
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
    "a photo of curtains", "a photo of an unknown room",
    "a photo of a floorplan"
]

cleanliness_prompts = [
    "This is a very clean room, everything is tidy and organized.",
    "This is a somewhat messy room, there are a few items scattered.",
    "This is a dirty room, cluttered and untidy with visible mess."
]

# =========================================================
# 4. QUEUE & BLOB CLIENT SETUP
# =========================================================
STORAGE_CONN_STR = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
queue_client = QueueClient.from_connection_string(STORAGE_CONN_STR, "rightmove-images-queue")
blob_service = BlobServiceClient.from_connection_string(STORAGE_CONN_STR)
from urllib.parse import urlparse
def parse_blob_url(blob_url: str):
    parsed = urlparse(blob_url)
    path = parsed.path.lstrip("/")   # london-rent/01-02-2026/...
    container, blob = path.split("/", 1)
    return container, blob

# =========================================================
# 5. PROCESS SINGLE BLOB
# =========================================================
def process_blob(container_name, blob_name):
    json_name = blob_name.rsplit(".", 1)[0] + ".json"
    blob_client = blob_service.get_blob_client(container_name, blob_name)
    json_client = blob_service.get_blob_client(container_name, json_name)

    # Skip if JSON already exists
    try:
        json_client.get_blob_properties()
        print(f"Skipping {blob_name}, JSON exists")
        return
    except:
        pass

    data = blob_client.download_blob().readall()
    image = Image.open(io.BytesIO(data)).convert("RGB")

    # === Your existing CLIP processing functions ===
    # Multi-scale detection
    annotated_img, labels_dict = detect_objects_multiscale_annotate(image)
    # Room & cleanliness
    label, conf, clean_level, reason = "unknown", 1.0, "unknown", "unknown"
    try:
        label, conf, clean_level, reason = classify_room_and_cleanliness(image)
    except Exception as e:
        print(f"⚠️ Error classifying {blob_name}: {e}")
    # Room type from detected objects
    room_type = classify_room_from_objects(labels_dict)

    # Build JSON result
    result = {
        "image_name": blob_name,
        "objects_detected": labels_dict,  # placeholder if you want to integrate multi-scale later
        "primary_label": label,
        "primary_confidence": conf,
        "cleanliness_level": clean_level,
        "cleanliness_reason": reason,
        "room_type": room_type
    }

    # Upload JSON
    json_client.upload_blob(json.dumps(result, indent=4), overwrite=True)
    print(f"✅ Processed {blob_name}")

# =========================================================
# 6. WORKER LOOP
# =========================================================
def main():
    print("📥 Worker started, polling queue...")
    while True:
        messages = queue_client.receive_messages(messages_per_page=32)
        for msg in messages:
            try:
                data = json.loads(msg.content)
            except Exception as e:
                print(f"⚠️ Failed to parse JSON: {e}, content: {msg.content}")
                try:
                    queue_client.delete_message(msg.id, msg.pop_receipt)
                except Exception as e2:
                    print(f"⚠️ Could not delete message: {e2}")
                continue

            blob_url = data.get("url")
            container_name = data.get("container")
            prefix = data.get("prefix")

            if not (blob_url or (container_name and prefix)):
                print(f"⚠️ Message missing URL and container/prefix: {msg.content}")
                try:
                    queue_client.delete_message(msg.id, msg.pop_receipt)
                except Exception as e:
                    print(f"⚠️ Could not delete message: {e}")
                continue

            if blob_url:
                # Single blob
                container_name, blob_name = parse_blob_url(blob_url)
                process_blob(container_name, blob_name)
            else:
                # Process all blobs under the prefix
                prefix = prefix.lstrip("/").rstrip("/")  # remove accidental slashes
                container_client = blob_service.get_container_client(container_name)
                blobs = list(container_client.list_blobs(name_starts_with=prefix))

                if not blobs:
                    print(f"⚠️ No blobs found under prefix '{prefix}' in container '{container_name}'")
                for blob in blobs:
                    # Only process files, skip "folders"
                    if not blob.name.endswith("/"):
                        process_blob(container_name, blob.name)
            try:
                queue_client.delete_message(msg.id, msg.pop_receipt)
            except Exception as e:
                print(f"⚠️ Could not delete message (possibly already gone): {e}")

        time.sleep(1)


if __name__ == "__main__":
    main()
