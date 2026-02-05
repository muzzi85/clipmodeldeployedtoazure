import os
import json
import torch
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1️⃣ OBJECT GROUP DEFINITIONS
# ------------------------------------------------------------
OBJECT_GROUPS = {
    "bed": [
        "a bed",
        "a single bed",
        "a double bed",
        "a bedroom bed",
        "a duvet",
        "a pillow and duvet",
        "a bedside table"
    ],
    "gym": [
        "a gym",
        "a training machine",
        "a treadmill",
        "a weight machine"
    ],
    "sofa": [
        "a sofa",
        "a leather sofa",
        "a couch",
        "living room seating furniture",
    ],
    "kitchen_appliance": [
        "a kitchen stove",
        "a cooking oven",
        "a stove vent hood",
    ],
    "laundry_appliance": [
        "a washing machine",
    ],
    "bathroom_fixture": [
        "a bath",
        "a bathtub",
        "a shower head",
        "a toilet",
        "a bathroom basin",
    ],
    "dining": [
        "a dining table and chairs",
        "a dining area",
    ],
    "floor": [
        "a tiled floor",
        "a wooden floor",
        "a carpeted floor",
    ],
    "window": [
        "a window",
        "curtains on a window",
    ],
    "lighting": [
        "a ceiling light",
        "a lamp",
    ],
    "outdoor": [
        "a garden",
        "a tree",
        "a car",
        "a building exterior",
        "a sea view",
    ],
    "floorplan": [
        "a floor plan",
        "a residential blueprint",
        "a 2D house layout",
    ],
}

## just storing function

from collections import defaultdict

def extract_frequent_phrases(detections, min_conf=0.5, min_count=2):
    phrase_counts = defaultdict(int)

    for _, _, _, conf, phrase in detections:
        if conf >= min_conf:
            phrase_counts[phrase] += 1

    # Keep only phrases appearing more than once
    frequent_phrases = {
        phrase: count
        for phrase, count in phrase_counts.items()
        if count >= min_count
    }

    return frequent_phrases

# ------------------------------------------------------------
# 2️⃣ BUILD CLIP PROMPTS
# ------------------------------------------------------------
labels = []
label_to_group = {}

for group, phrases in OBJECT_GROUPS.items():
    for p in phrases:
        prompt = f"a photo of {p}"
        labels.append(prompt)
        label_to_group[prompt] = group

# ------------------------------------------------------------
# 3️⃣ MODEL SETUP
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device).eval()

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

# ------------------------------------------------------------
# 4️⃣ PRECOMPUTE TEXT EMBEDDINGS
# ------------------------------------------------------------
with torch.no_grad():
    text_inputs = processor(
        text=labels,
        return_tensors="pt",
        padding=True
    ).to(device)

    text_features = model.get_text_features(**text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# ------------------------------------------------------------
# 5️⃣ BATCH CLASSIFIER
# ------------------------------------------------------------
def classify_room_batch(images, batch_size=32):
    results = []
    logit_scale = model.logit_scale.exp()

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]

        inputs = processor(
            images=batch,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = logit_scale * image_features @ text_features.T
            probs = logits.softmax(dim=1)

            confs, preds = probs.max(dim=1)

        for j in range(len(batch)):
            phrase = labels[preds[j].item()]
            group = label_to_group[phrase]
            results.append((group, phrase, confs[j].item()))

    return results
def detect_objects_multiscale_annotate(
    image_path,
    n_scales=4,
    overlap_ratio=0.1,
    min_confidence=0.25
):
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    min_dim = min(W, H)

    scales = sorted(
        set(
            int(min_dim / (2 ** i))
            for i in range(n_scales, 0, -1)
            if int(min_dim / (2 ** i)) > 96
        )
    )

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    detections = []

    group_scores = defaultdict(float)   # 🔑 confidence-weighted
    group_counts = defaultdict(int)     # for diagnostics only

    for crop_size in scales:
        stride = int(crop_size * (1 - overlap_ratio))
        x_steps = list(range(0, W - crop_size, stride)) + [W - crop_size]
        y_steps = list(range(0, H - crop_size, stride)) + [H - crop_size]

        crops, meta = [], []

        for top in y_steps:
            for left in x_steps:
                crop = image.crop((left, top, left + crop_size, top + crop_size))
                crops.append(crop)
                meta.append((left, top, crop_size))

        if not crops:
            continue

        results = classify_room_batch(crops)
        x = 1
        for (group, phrase, conf), (left, top, size) in zip(results, meta):
            if conf < min_confidence:
                continue

            cx = left + size // 2
            cy = top + size // 2

            detections.append((cx, cy, group, conf, phrase))

            # 🔥 This is the key change
            group_scores[group] += conf
            group_counts[group] += 1

    # Optional: draw only stronger signals for clarity
    for cx, cy, group, conf, phrase in detections:
        if conf < 0.5:
            continue

        text = f"{group} ({conf*100:.1f}%)"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        draw.rectangle(
            [(cx - tw//2 - 4, cy - th//2 - 4),
             (cx + tw//2 + 4, cy + th//2 + 4)],
            fill=(0, 0, 0)
        )
        draw.text(
            (cx - tw//2, cy - th//2),
            text,
            fill="white",
            font=font
        )

    return detections, dict(group_scores), dict(group_counts)


# ------------------------------------------------------------
# 7️⃣ CLEANLINESS CLASSIFICATION
# ------------------------------------------------------------
cleanliness_prompts = [
    "This is a very clean room, everything is tidy and organized.",
    "This is a somewhat messy room, there are a few items scattered.",
    "This is a dirty room, cluttered and untidy with visible mess."
]

def classify_cleanliness(image):
    inputs = processor(
        text=cleanliness_prompts,
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
        idx = probs.argmax(dim=1).item()

    return idx + 1, cleanliness_prompts[idx]
# ------------------------------------------------------------
# 8️⃣ ROOM INFERENCE FROM GROUP SCORES (CONFIDENCE-AWARE)
# ------------------------------------------------------------
def classify_room_from_objects(
    group_scores,
    group_counts,
    min_score=3.0,
    dominance_ratio=2
):
    """
    Infer room type using confidence-weighted group scores.

    Args:
        group_scores (dict): {group_name: sum_confidence}
        group_counts (dict): {group_name: count}
        min_score (float): minimum total confidence to consider a room
        dominance_ratio (float): how much a room must dominate others

    Returns:
        str: inferred room type
    """

    categories = {
        "floorplan": ["floorplan"],
        "bedroom": ["bed"],
        "bathroom": ["bathroom_fixture"],
        "kitchen/cooking area": ["kitchen_appliance"],
        "sitting/living room": ["sofa", "dining"],
        "outdoor": ["outdoor"],
        "gym": ["gym"],
        "laundry": ["laundry_appliance"],
    }

    # Aggregate room scores
    room_scores = {}
    room_counts = {}

    for room, groups in categories.items():
        room_scores[room] = sum(group_scores.get(g, 0.0) for g in groups)
        room_counts[room] = sum(group_counts.get(g, 0) for g in groups)

    # 🧠 Hard override: floorplan dominates visually
    if room_scores.get("floorplan", 0.0) >= 4.0:
        return "floorplan"

    # Keep rooms with meaningful evidence
    valid_rooms = {
        room: score
        for room, score in room_scores.items()
        if score >= min_score and room != "floorplan"
    }

    if not valid_rooms:
        return "unknown"

    # Sort rooms by confidence score
    sorted_rooms = sorted(
        valid_rooms.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best_room, best_score = sorted_rooms[0]

    # Single strong winner
    if len(sorted_rooms) == 1:
        return best_room

    second_score = sorted_rooms[1][1]

    # 🏆 Dominance check
    if best_score >= dominance_ratio * second_score:
        return best_room

    # Otherwise → mixed room
    return " & ".join(room for room, _ in sorted_rooms)

# ------------------------------------------------------------
# 9️⃣ PROCESS DIRECTORY
# ------------------------------------------------------------
def process_all_properties(root_dir):
    for folder in os.listdir(root_dir):
        path = os.path.join(root_dir, folder)
        if not os.path.isdir(path):
            continue

        print(f"\n📁 {folder}")

        for file in os.listdir(path):
            if not file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            print(f"\n📁 {file}")

            image_path = os.path.join(path, file)
            image = Image.open(image_path).convert("RGB")
            objects, group_scores, group_counts = detect_objects_multiscale_annotate(image_path)
            room = classify_room_from_objects(group_scores, group_counts)
            clean_level, clean_reason = classify_cleanliness(image)

            objects = extract_frequent_phrases(objects)
            out = {
                "image_name": file,
                "objects_detected": objects,
                "room_type": room,
                "cleanliness_level": clean_level,
                "cleanliness_reason": clean_reason
            }

            json_path = os.path.join(
                path, file.rsplit(".", 1)[0] + ".json"
            )
            with open(json_path, "w") as f:
                json.dump(out, f, indent=4)

            print(f"  ✅ {file}")

    print("\n🎉 Done!")

# ------------------------------------------------------------
# 🔟 RUN
# ------------------------------------------------------------
ROOT_IMAGE_DIR = "/mnt/c/Muzzi work/test"
ROOT_IMAGE_DIR = "/mnt/c/Muzzi work/test/s"

process_all_properties(ROOT_IMAGE_DIR)
