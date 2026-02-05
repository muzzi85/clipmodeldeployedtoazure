import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import json
from PIL import Image
import matplotlib.pyplot as plt

# Existing object labels (shortened for clarity)
labels = [
    "a photo of a single bed",
    "a photo of a double bed",
    "a photo of a bed",
    "a photo of a bedroom",
    "a photo of a bathroom basin",
    "a photo of a bath",
    "a photo of a bathroom",
    "a photo of a refrigerator",
    "a photo of a duvet",
    "a photo of a pillow",
    "a photo of a television",
    "a photo of a wardrobe",
    "a photo of a washing machine",
    "a photo of a Single Ended Bath",
    "a photo of a stove vent hood",
    "a photo of a sofa",
    "a photo of a leather sofa",
    "a photo of a living room",
    "a photo of a shower head",
    "a photo of a heater radiator",
    "a photo of a kitchen stove",
    "a photo of a kitchen",
    "a photo of a toilet seat or loo",
    "a photo of a kitchen sink",
    "a photo of a dinning table and chairs",
    "a photo of stairs",
    "a photo of a garden",
    "a photo of a car",
    "a photo of a tree",
    "a photo of a door",
    "a photo of a tiles",
    "a photo of a carpet",
    "a photo of a wood floor",
    "a photo of an electric socket",
    "a photo of an alarm",
    "a photo of a camera",
    "a photo of a celing light",
    "a photo of a shop",
    "a photo of a window",
    "a photo of curtains",
    "a photo of an unknown room",
    "a photo of a building",
    "a photo of a sea", 
    "a photo of seaview",
    "a photo of a single bed",
    "a photo of a double bed",
    "a photo of a bed",
    "a photo of a bedroom",
    "a photo of a bathroom basin",
    "a photo of a bath",
    "a photo of a bathroom",
    "a photo of a refrigerator",
    "a photo of a duvet",
    "a photo of a pillow",
    "a photo of a television",
    "a photo of a wardrobe",
    "a photo of a stove vent hood",
    "a photo of a sofa",
    "a photo of a tub"
    "a photo of a shower head",
    "a photo of a heater radiator",
    "a photo of a kitchen stove",
    "a photo of a cooking oven",
    "a photo of a kitchen",
    "a photo of a kitchen sink",
    "a photo of a dinning table and chairs",
    "a photo of stairs",
    "a photo of a garden",
    "a photo of a car",
    "a photo of a tree",
    "a photo of a door",
    "a photo of a tiles",
    "a photo of a carpet",
    "a photo of a wood floor",
    "a photo of an electric socket",
    "a photo of an alarm",
    "a photo of a camera",
    "a photo of a celing light",
    "a photo of a shop",
    "a photo of a window",
    "a photo of curtains",
    "a photo of an unknown room",
    "a photo of a gym",
    "a photo of a training machine",
    "a photo of a 2D house layout",
    "a photo of a real estate floor plan with dimensions",
    "a photo of a residential blueprint"
]

# 1️⃣ Model + device setup (ADD THIS)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2️⃣ Precompute text embeddings ONCE (ADD THIS)
with torch.no_grad():
    text_inputs = processor(text=labels, return_tensors="pt", padding=True).to(device)
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# 3️⃣ Batched CLIP inference (NEW FUNCTION)
def classify_room_batch(images, batch_size=32):
    results = []

    logit_scale = model.logit_scale.exp()

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]

        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = logit_scale * (image_features @ text_features.T)
            probs = logits.softmax(dim=1)

            confs, preds = probs.max(dim=1)

        for j in range(len(batch)):
            results.append((labels[preds[j].item()], confs[j].item()))

    return results

# 4️⃣ 🚀 FAST multiscale detector (REPLACEMENT)
from collections import defaultdict
from PIL import ImageDraw, ImageFont

def detect_objects_multiscale_annotate(
    image_path,
    n_scales=4,
    overlap_ratio=0.1,
    confidence_threshold=0.35
):
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path.convert("RGB")

    W, H = image.size
    min_dim = min(W, H)

    scales = sorted(
        set(int(min_dim / (2 ** i)) for i in range(n_scales, 0, -1) if int(min_dim / (2 ** i)) > 96)
    )

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()

    detections = []
    label_counts = defaultdict(int)

    for crop_size in scales:
        stride = int(crop_size * (1 - overlap_ratio))
        x_steps = list(range(0, W - crop_size, stride)) + [W - crop_size]
        y_steps = list(range(0, H - crop_size, stride)) + [H - crop_size]

        crops = []
        meta = []

        for top in y_steps:
            for left in x_steps:
                crop = image.crop((left, top, left + crop_size, top + crop_size))
                crops.append(crop)
                meta.append((left, top, crop_size))

        if not crops:
            continue
        crops.append(image)
        results = classify_room_batch(crops)
        x=1
        for (label, conf), (left, top, size) in zip(results, meta):
            label_clean = label.replace("a photo of ", "").strip()
            cx = left + size // 2
            cy = top + size // 2

            # Relaxed bedroom
            if conf > confidence_threshold - 0.1 and label_clean in {
                "a bed", "a single bed", "a double bed", "a duvet", "a bedroom", "a pillow"
            }:
                detections.append((cx, cy, label_clean, conf))
                label_counts[label_clean] += 1
                continue

            # Relaxed kitchen
            if conf > confidence_threshold - 0.4 and label_clean in {
                "kitchen", "kitchen stove", "stove vent hood", "cooking oven"
            }:
                detections.append((cx, cy, label_clean, conf))
                label_counts[label_clean] += 1
                continue

            # General strict
            if conf > confidence_threshold and label_clean != "unknown room":
                detections.append((cx, cy, label_clean, conf))
                label_counts[label_clean] += 1

    # Draw results
    for cx, cy, label_clean, conf in detections:
        text = f"{label_clean} ({conf*100:.1f}%)"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        draw.rectangle(
            [(cx - tw//2 - 5, cy - th//2 - 5), (cx + tw//2 + 5, cy + th//2 + 5)],
            fill=(0, 0, 0)
        )
        draw.text((cx - tw//2, cy - th//2), text, fill="white", font=font)

    return annotated, dict(label_counts)
# New: cleanliness prompts
cleanliness_prompts = [
    "This is a very clean room, everything is tidy and organized.",   # 1
    "This is a somewhat messy room, there are a few items scattered.", # 2
    "This is a dirty room, cluttered and untidy with visible mess."    # 3
]

def classify_room_and_cleanliness(
    image,
    object_conf_threshold=0.2
):
    device = next(model.parameters()).device  # 🔑 get model device

    obj_inputs = processor(
        text=labels,
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(device)

    clean_inputs = processor(
        text=cleanliness_prompts,
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        obj_outputs = model(**obj_inputs)
        clean_outputs = model(**clean_inputs)

        # --------------------------
        # OBJECT CLASSIFICATION
        # --------------------------
        obj_probs = obj_outputs.logits_per_image.softmax(dim=1)
        obj_pred = obj_probs.argmax(dim=1).item()
        obj_conf = float(obj_probs[0, obj_pred])
        best_label = labels[obj_pred]

        if obj_conf < object_conf_threshold:
            best_label = "unknown"

        # --------------------------
        # CLEANLINESS CLASSIFICATION
        # --------------------------
        clean_probs = clean_outputs.logits_per_image.softmax(dim=1)
        clean_pred = clean_probs.argmax(dim=1).item()

        cleanliness_level = clean_pred + 1
        cleanliness_reason = cleanliness_prompts[clean_pred]

    return best_label, obj_conf, cleanliness_level, cleanliness_reason

## 5

def classify_room_from_objects_multi_label_count(objects_dict, min_score=3):
    """
    Infer room type from detected objects using object counts.

    Args:
        objects_dict (dict): {"object_label": count, ...}
        min_score (int): minimum total score to confidently assign a room

    Returns:
        str: room type or "unknown"
    """
    # Normalize keys
    detected = {k.lower().strip(): v for k, v in objects_dict.items()}
    categories = {
        "floorplan": ["a floorplan","a 2D house layout", "a real estate floor plan with dimensions", "a residential blueprint"],
        "bedroom": ["a bed", "a single bed", "a double bed", "a duvet", "a bedroom"],
        "bathroom": ["a bath", "a bathroom basin", "a shower head", "a toilet seat or loo", "a single ended bath", "a bathroom"],
        "living/sitting room": ["a sofa", "a dinning table and chairs","a living room"],
        "outdoor": ["a garden", "a tree", "a car", "a building", "a sea", "a seaview"],
        "kitchen/cooking area": ["a refrigerator", "a kitchen stove", "a stove vent hood", "a kitchen", "a kitchen sink","a kitchen","a cooking oven"],
        "gym": ["a gym", "a training machine"],
        "laundry": ["a washing machine"]
    }

    # Compute scores per room
    room_scores = {room: 0 for room in categories}
    for room, objs in categories.items():
        for obj in objs:
            if obj in detected:
                room_scores[room] += detected[obj]  # count objects for weight

    # Find the room with highest score
    # best_room = max(room_scores, key=room_scores.get)
    # best_score = room_scores[best_room]

    if room_scores.get("floorplan", 0) >= 5:
            return "floorplan"

     # ---------------------------
    # Rooms that pass threshold
    # ---------------------------
    valid_rooms = [
        room for room, score in room_scores.items()
        if score >= min_score and room != "floorplan"
    ]
    if valid_rooms:
        if room_scores.get("bedroom", 0) == 1:
            return "bedroom"
    else:
        return "unknown"

    if len(valid_rooms) == 1:
        return valid_rooms[0]

    # ---------------------------
    # Mixed room case
    # ---------------------------
    return " & ".join(sorted(valid_rooms))

# ------------------------------------------------------------
def process_all_properties(root_dir):
    """
    Walk through the main directory and process all images
    inside each property_* folder.
    """
    for property_folder in os.listdir(root_dir):
        full_property_path = os.path.join(root_dir, property_folder)

        # Only process directories like property_XXXXXX
        if not os.path.isdir(full_property_path):
            continue
        
        print(f"\n📁 Processing folder: {property_folder}")

        # Loop through all images inside the folder
        for file in os.listdir(full_property_path):
            if not file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            image_path = os.path.join(full_property_path, file)

            # -----------------------------
            # CHECK IMAGE SIZE
            # -----------------------------
            try:
                img = Image.open(image_path)
                w, h = img.size
            except:
                print(f"  ⚠️ Failed to open: {file}")
                continue

            if w < 300 or h < 400:
                print(f"  ⏩ Skipping {file} (too small: {w}x{h})")
                continue

            print(f"  🖼️ Processing image: {file}")

            annotated_image, labels_dict = detect_objects_multiscale_annotate(image_path)

            image = img.convert("RGB")
            label, conf, clean_level, reason = classify_room_and_cleanliness(
                image,
                object_conf_threshold=0.9
            )
            if conf is not None:
                if float(str(conf)) < 0.9:
                    label="unknown"
                    clean_level = "None"
                    reason = "None"
            if label=="unknown":
                clean_level = "None"
                reason = "None"
                conf = "None"
            if conf is None:
                label="unknown"
                clean_level = "None"
                reason = "None"
                conf = "None"


                #continue
            room_type = classify_room_from_objects_multi_label_count(labels_dict)

            # Keep only items with value >= 2
            labels_dict = {k: v for k, v in labels_dict.items() if v >= 2}
            # ------------------------------------------------------------
            # CREATE JSON OUTPUT
            # ------------------------------------------------------------
            json_output = {
                "image_name": file,
                "objects_detected": labels_dict,
                "primary_label": label,
                "primary_confidence": conf,
                "cleanliness_level": clean_level,
                "cleanliness_reason": reason,
                "room_type": room_type
            }

            # ------------------------------------------------------------
            # SAVE JSON NEXT TO IMAGE
            # ------------------------------------------------------------
            json_path = os.path.join(
                full_property_path, file.rsplit(".",1)[0] + ".json"
            )
            with open(json_path, "w") as f:
                json.dump(json_output, f, indent=4)

            print(f"  ✅ Saved JSON: {json_path}")

    print("\n🎉 All images processed!")


ROOT_IMAGE_DIR = "/mnt/c/Muzzi work/DAS4Whales-main/houseclassification/rightmove_images_Glasgow_09_12_2025"
ROOT_IMAGE_DIR = "/mnt/c/Muzzi work/DAS4Whales-main/houseclassification/rightmove_data_20_Glasgow/images/test"
#ROOT_IMAGE_DIR = "/mnt/c/Muzzi work/test/s"
ROOT_IMAGE_DIR = "/mnt/c/Muzzi work/test"
process_all_properties(ROOT_IMAGE_DIR)

