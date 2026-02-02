import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# 1. Load CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

### test for room object detection

from PIL import Image

def crop_image_with_overlap(image_path, crop_size=500, overlap_ratio=0.7, include_edges=True):
    """
    Crop an image into multiple overlapping patches.
    
    Args:
        image_path (str): Path to the input image.
        crop_size (int): Size of the square crops (default: 256).
        overlap_ratio (float): Overlap ratio between 0 and 1 (default: 0.25).
        include_edges (bool): Whether to include edge patches even if not a full stride away.
        
    Returns:
        List[Image.Image]: List of cropped PIL Image objects.
    """
    image = Image.open(image_path)
    width, height = image.size
    stride = int(crop_size * (1 - overlap_ratio))
    
    # Calculate crop positions
    x_steps = list(range(0, width - crop_size + 1, stride))
    y_steps = list(range(0, height - crop_size + 1, stride))

    if include_edges:
        if x_steps[-1] + crop_size < width:
            x_steps.append(width - crop_size)
        if y_steps[-1] + crop_size < height:
            y_steps.append(height - crop_size)

    # Crop and collect images
    crops = []
    for top in y_steps:
        for left in x_steps:
            box = (left, top, left + crop_size, top + crop_size)
            crops.append(image.crop(box))

    return crops


### multi crop sizes

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from collections import defaultdict

def detect_objects_multiscale_annotate(
    image_path, 
    n_scales=4, 
    overlap_ratio=0.1, 
    confidence_threshold=0.7
):
    """
    Multi-scale object detection + annotation using adaptive crops.
    Also returns a JSON-style dictionary of unique labels detected.
    
    Returns:
        annotated_image (PIL.Image)
        label_summary (dict)
    """
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    min_dim = min(W, H)
    
    # Automatically determine scales (progressively smaller)
    scales = [int(min_dim / (2 ** (i))) for i in range(n_scales, 0, -1)]
    scales = sorted(set(s for s in scales if s > 60))
    print(f"🧩 Scales: {scales}")

    # Prepare drawing surface
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()

    detections = []                # (cx, cy, label, conf)
    label_counts = defaultdict(int)  # unique label → count

    for crop_size in scales:
        stride_x = int(crop_size * (1 - overlap_ratio))
        stride_y = int(crop_size * (1 - overlap_ratio))

        x_steps = list(range(0, W - crop_size, stride_x))
        y_steps = list(range(0, H - crop_size, stride_y))
        if x_steps[-1] + crop_size < W:
            x_steps.append(W - crop_size)
        if y_steps[-1] + crop_size < H:
            y_steps.append(H - crop_size)

        print(f"🔹 Scale {crop_size}px → {len(x_steps)*len(y_steps)} crops")

        for top in y_steps:
            for left in x_steps:
                box = (left, top, left + crop_size, top + crop_size)
                crop = image.crop(box)
                label, conf, _ = classify_room(crop)

                if conf > confidence_threshold and label != "a photo of an unknown room":
                    center_x = left + crop_size // 2
                    center_y = top + crop_size // 2
                    
                    # clean the label
                    label_clean = label.split("a photo of ")[-1].strip()

                    detections.append((center_x, center_y, label_clean, conf))

                    # Count unique label once per detection
                    label_counts[label_clean] += 1

    # Draw detections on combined image
    for (cx, cy, label_clean, conf) in detections:
        text = f"{label_clean} ({conf*100:.1f}%)"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        draw.rectangle(
            [(cx - tw//2 - 6, cy - th//2 - 6), (cx + tw//2 + 6, cy + th//2 + 6)],
            fill=(0, 0, 0, 160)
        )
        draw.text((cx - tw//2, cy - th//2), text, fill="white", font=font)

    print(f"✅ Total detections: {len(detections)}")
    print(f"🧾 Unique label summary: {dict(label_counts)}")

    return annotated, dict(label_counts)


### cleaness

# Existing object labels (shortened for clarity)
object_labels = [
  "a photo of a single bed",
    "a photo of a double bed",
    "a photo of a bed",
    "a photo of a bathroom basin",
    "a photo of a bath",
    "a photo of a refrigerator",
    "a photo of a duvet",
    "a photo of a pillow",
    "a photo of a television",
    "a photo of a wardrobe",
    "a photo of a Single Ended Bath",
    "a photo of a stove vent hood",
    "a photo of a sofa",
    "a photo of a shower head",
    "a photo of a heater radiator",
    "a photo of a kitchen stove",
    "a photo of a kitchen cabinets",
    "a photo of a toilet seat ",
    "a photo of a kitchen sink",
    "a photo of a dinning table and chairs",
    "a photo of stairs",
    "a photo of a garden",
    "a photo of a car",
    "a photo of a tree",
    "a photo of a cabinets",
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
    "a photo of an unknown room"
]
# New: cleanliness prompts
cleanliness_prompts = [
    "This is a very clean room, everything is tidy and organized.",   # 1
    "This is a somewhat messy room, there are a few items scattered.", # 2
    "This is a dirty room, cluttered and untidy with visible mess."    # 3
]


def classify_room_and_cleanliness(
    image,
    object_conf_threshold=0.40   # default = 40% confidence required
):
    """
    Classify room object and estimate cleanliness (1=clean, 3=dirty).

    Args:
        image: PIL image
        object_conf_threshold (float): minimum confidence required to accept object label

    Returns:
        - best_label: str (object detected or 'unknown')
        - obj_conf: float (confidence)
        - cleanliness_level: int (1–3)
        - cleanliness_reason: str
    """
    # Process for both object detection and cleanliness
    obj_inputs = processor(text=object_labels, images=image,
                           return_tensors="pt", padding=True)
    clean_inputs = processor(text=cleanliness_prompts, images=image,
                             return_tensors="pt", padding=True)

    with torch.no_grad():
        obj_outputs = model(**obj_inputs)
        clean_outputs = model(**clean_inputs)

        # --------------------------
        # OBJECT DETECTION
        # --------------------------
        obj_probs = obj_outputs.logits_per_image.softmax(dim=1)
        obj_pred = torch.argmax(obj_probs, dim=1).item()
        obj_conf = obj_probs[0, obj_pred].item()
        best_label = object_labels[obj_pred]

        # Apply threshold
        if obj_conf < object_conf_threshold:
            best_label = "unknown"
            obj_conf = float(obj_conf)   # still show actual confidence
        else:
            obj_conf = float(obj_conf)

        # --------------------------
        # CLEANLINESS CLASSIFICATION
        # --------------------------
        clean_probs = clean_outputs.logits_per_image.softmax(dim=1)
        clean_pred = torch.argmax(clean_probs, dim=1).item()
        cleanliness_level = clean_pred + 1
        cleanliness_reason = cleanliness_prompts[clean_pred]

    return best_label, obj_conf, cleanliness_level, cleanliness_reason



import os
import json
from PIL import Image
import matplotlib.pyplot as plt


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

            # ------------------------------------------------------------
            # RUN DETECTION + CLEANLINESS
            # ------------------------------------------------------------
            annotated_image, labels_dict = detect_objects_multiscale_annotate(
                image_path,
                n_scales=4,
                overlap_ratio=0.15,
                confidence_threshold=0.85
            )

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

            # ------------------------------------------------------------
            # CREATE JSON OUTPUT
            # ------------------------------------------------------------
            json_output = {
                "image_name": file,
                "objects_detected": labels_dict,
                "primary_label": label,
                "primary_confidence": conf,
                "cleanliness_level": clean_level,
                "cleanliness_reason": reason
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



## classify rooms as well as detecting objects

labels = [
    "a photo of a single bed",
    "a photo of a double bed",
    "a photo of a bed",
    "a photo of a bathroom basin",
    "a photo of a bath",
    "a photo of a refrigerator",
    "a photo of a duvet",
    "a photo of a pillow",
    "a photo of a television",
    "a photo of a wardrobe",
    "a photo of a Single Ended Bath",
    "a photo of a stove vent hood",
    "a photo of a sofa",
    "a photo of a living room",
    "a photo of a shower head",
    "a photo of a heater radiator",
    "a photo of a kitchen stove",
    "a photo of a kitchen cabinets",
    "a photo of a toilet seat ",
    "a photo of a kitchen sink",
    "a photo of a dinning table and chairs",
    "a photo of stairs",
    "a photo of a garden",
    "a photo of a car",
    "a photo of a tree",
    "a photo of a cabinets",
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
    "a photo of an unknown room"
]
# 3. Inference function
def classify_room(imagee):
    #image = Image.open(image_path).convert("RGB")
    inputs = processor(text=labels, images=imagee, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return labels[pred], probs[0][pred].item(),inputs

def classify_room_from_objects(objects_dict):


    """
    Decide the room type based ONLY on detected objects from the CLIP detector.
    objects_dict = { "object label": confidence, ... }
    """

    # Normalize keys (lowercase, strip)
    detected = objects_dict#{k.lower().strip() for k in objects_dict.keys()}

    # ---------------------------
    # ROOM RULES
    # ---------------------------
    # ---------------------------

    # Kitchen rules
    kitchen_objects = [
        "a photo of a refrigerator",
        "a photo of a kitchen stove",
        "a photo of a stove vent hood",
        "a photo of a kitchen cabinets",
        "a photo of a kitchen sink",
    ]
    # Bedroom rules
    bedroom_objects = [
        "a photo of a bed",
        "a photo of a single bed",
        "a photo of a double bed",
        "a photo of a duvet",
        "a photo of a pillow",
        "a photo of a wardrobe",
    ]

    # Bathroom rules
    bathroom_objects = [
        "a photo of a bath",
        "a photo of a bathroom basin",
        "a photo of a shower head",
        "a photo of a toilet seat",
        "a photo of a single ended bath",
    ]

    # Living room rules
    livingroom_objects = [
        "a photo of a sofa",
        "a photo of a television",
        "a photo of a dinning table and chairs",
        "a photo of a heater radiator",
        "a photo of a living room",
    ]

    # Outdoor rules
    outdoor_objects = [
        "a photo of a garden",
        "a photo of a tree",
        "a photo of a car",
    ]

    # Helper for rule matching
    def has_any(category_objects):
        return any(obj.lower() in detected for obj in category_objects)

    # ---------------------------
    # DECISION LOGIC
    # ---------------------------
    if has_any(kitchen_objects):
        return "kitchen"

    if has_any(bedroom_objects):
        return "bedroom"

    if has_any(bathroom_objects):
        return "bathroom"

    if has_any(livingroom_objects):
        return "living room"

    if has_any(outdoor_objects):
        return "outdoor"

    return "unknown"

    """
    Infer final room type based purely on detected objects.
    labels_dict is: { "object_label": confidence, ... }
    """
    # Normalize keys for matching
    detected_objects = [obj.lower() for obj in labels_dict.keys()]

    # Count hits per room
    scores = {room: 0 for room in ROOM_RULES}

    for room, rule in ROOM_RULES.items():
        for keyword in rule["keywords"]:
            for obj in detected_objects:
                if keyword in obj:
                    scores[room] += 1

    # Pick the room with highest score
    best_room = max(scores, key=scores.get)

    # Ensure minimum object evidence
    if scores[best_room] >= ROOM_RULES[best_room]["min_hits"]:
        return best_room
    else:
        return "unknown"


def process_all_properties(root_dir):
    """
    Walk through the main directory and process all images
    inside each property_* folder.
    """
    for property_folder in os.listdir(root_dir):
        full_property_path = os.path.join(root_dir, property_folder)

        if not os.path.isdir(full_property_path):
            continue
        
        print(f"\n📁 Processing folder: {property_folder}")

        for file in os.listdir(full_property_path):
            if not file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            image_path = os.path.join(full_property_path, file)

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


            ## detection on full image

            labels_dict, conf_O, _ = classify_room(img)

            if conf_O < 0.3:
                labels_dict = ""
            # ------------------------------------------------------------
            # DETECTION
            # ------------------------------------------------------------
            # annotated_image, labels_dict = detect_objects_multiscale_annotate(
            #     image_path,
            #     n_scales=4,
            #     overlap_ratio=0.15,
            #     confidence_threshold=0.85
            # )


            # CLIP classification
            image = img.convert("RGB")
            label, conf, clean_level, reason = classify_room_and_cleanliness(
                image, object_conf_threshold=0.9
            )

            # Clean fallback logic
            if conf is None or (str(conf).replace(".", "").isdigit() and float(conf) < 0.9):
                label = "unknown"
                clean_level = "None"
                reason = "None"
                conf = "None"

            # ------------------------------------------------------------
            # ROOM TYPE FROM OBJECTS
            # ------------------------------------------------------------
            room_type = classify_room_from_objects(labels_dict)

            # ------------------------------------------------------------
            # JSON OUTPUT
            # ------------------------------------------------------------
            json_output = {
                "image_name": file,
                "objects_detected": labels_dict,
                "primary_label": label,
                "primary_confidence": conf,
                "cleanliness_level": clean_level,
                "cleanliness_reason": reason,
                "room_type": room_type   # <-------- NEW KEY
            }

            # Save JSON
            json_path = os.path.join(full_property_path, file.rsplit(".",1)[0] + ".json")
            with open(json_path, "w") as f:
                json.dump(json_output, f, indent=4)

            print(f"  ✅ Saved JSON: {json_path}")

    print("\n🎉 All images processed!")


ROOT_IMAGE_DIR = "/mnt/c/Muzzi work/DAS4Whales-main/houseclassification/rightmove_images_Glasgow_09_12_2025"
process_all_properties(ROOT_IMAGE_DIR)

