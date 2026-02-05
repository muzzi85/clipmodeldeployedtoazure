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
    confidence_threshold=0.65
):
    """
    Multi-scale object detection + annotation using adaptive crops.
    Also returns a JSON-style dictionary of unique labels detected.
    
    Returns:
        annotated_image (PIL.Image)
        label_summary (dict)
    """
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    elif isinstance(image_path, Image.Image):
        image = image_path.convert("RGB")
    else:
        raise ValueError("Input must be a file path or PIL.Image")
        #image = image_path
    print("imageitself", image)
    print("image size",image.size )

    W, H = image.size
    min_dim = min(W, H)
    
    # Automatically determine scales (progressively smaller)
    scales = sorted(set(int(min_dim / (2 ** i)) for i in range(n_scales, 0, -1) if int(min_dim / (2 ** i)) > 60))

    print(f"🧩 Scales: {scales}")

    # Prepare drawing surface
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

                # ## bedroom case
                if conf > (confidence_threshold-0.4) and (label == "a bed" or label == "a single bed" or label == "a double bed" or label =="a duvet" or label =="a bedroom"):
                    center_x = left + crop_size // 2
                    center_y = top + crop_size // 2
                    
                    # clean the label
                    label_clean = label.split("a photo of ")[-1].strip()

                    detections.append((center_x, center_y, label_clean, conf))

                    # Count unique label once per detection
                    label_counts[label_clean] += 1
                
                #  ## kitchen case
                if conf > (confidence_threshold-0.4) and label == "a kitchen stove" or label == "a stove vent hood" or label == "a cooking oven" or label =="a kitchen":
                    center_x = left + crop_size // 2
                    center_y = top + crop_size // 2
                    
                    # clean the label
                    label_clean = label.split("a photo of ")[-1].strip()

                    detections.append((center_x, center_y, label_clean, conf))

                    # Count unique label once per detection
                    label_counts[label_clean] += 1

                ## general case
                if conf > confidence_threshold and label != "a photo of an unknown room":
                    print("general case")
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

    label, conf = classify_room(image)
    label_clean = label.split("a photo of ")[-1].strip()
    if conf > 0.5:
        label_counts[label_clean] += 1
        detections.append((0, 0, label_clean, conf))
    if conf > 0.3 and label_clean=="a photo of a kitchen":
        label_counts[label_clean] += 1
        detections.append((0, 0, label_clean, conf))
    if conf > 0.3 and (label_clean=="a single bed" or label_clean=="a bed" or label_clean=="a double bed"):
        label_counts[label_clean] += 1
        detections.append((0, 0, label_clean, conf))
    print(f"✅ detections: {detections}")
    print(f"✅ Total detections: {len(detections)}")
    print(f"🧾 Unique label summary: {dict(label_counts)}")
    return annotated, dict(label_counts)

### cleaness

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
    "a photo of a sink tap",
    "a photo of a sofa",
    "a photo of a tub"
    "a photo of a shower head",
    "a photo of a heater radiator",
    "a photo of a kitchen stove",
    "a photo of a kitchen cabinets and sink",
    "a photo of a cooking oven",
    "a photo of a kitchen",
    "a photo of a toilet seat ",
    "a photo of a kitchen sink",
    "a photo of a dinning table and chairs",
    "a photo of stairs",
    "a photo of a garden",
    "a photo of a car",
    "a photo of a tree",
    "a photo of a cabinet and sink",
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

# Remove duplicates while preserving order
labels = list(dict.fromkeys(labels))

# labels = [
#     "a photo of a single bed", "a photo of a double bed", "a photo of a bed",
#     "a photo of a bathroom basin", "a photo of a bath", "a photo of a refrigerator",
#     "a photo of a bathroom", "a photo of a duvet", "a photo of a pillow",
#     "a photo of a television", "a photo of a wardrobe", "a photo of a single ended bath",
#     "a photo of a stove vent hood", "a photo of a sofa", "a photo of a shower head",
#     "a photo of a heater radiator", "a photo of a kitchen stove", "a photo of a kitchen",
#     "a photo of a toilet seat", "a photo of a kitchen sink", "a photo of a dining table and chairs",
#     "a photo of stairs", "a photo of a garden", "a photo of a car", "a photo of a tree",
#     "a photo of cabinets", "a photo of a door", "a photo of tiles", "a photo of a carpet",
#     "a photo of a wood floor", "a photo of an electric socket", "a photo of an alarm",
#     "a photo of a camera", "a photo of a ceiling light", "a photo of a shop",
#     "a photo of a window", "a photo of curtains", "a photo of an unknown room",
#     "a photo of a living room", "a photo of a floorplan","a photo of a gym","a photo of a gym tools","a photo of a gym machine"
# ]

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
    obj_inputs = processor(text=labels, images=image,
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
        best_label = labels[obj_pred]

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

## classify rooms as well as detecting objects
def has_any(category_objects):
    return any(obj.lower() in detected for obj in category_objects)
# Kitchen rules
kitchen_objects = [
"a photo of a kitchen cabinets",
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

# 3. Inference function
def classify_room(imagee):
    #image = Image.open(image_path).convert("RGB")
    inputs = processor(text=labels, images=[imagee], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return labels[pred], probs[0][pred].item()

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

    def has_any(category_objects):
        return any(obj.lower() in detected for obj in category_objects)

    # Helper for rule matching

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
def classify_room_from_objects_multi_label(objects_dict):
    """Infer room type(s) from detected objects. Returns multiple types if necessary."""
    # Lowercase keys and strip spaces
    detected = {k.lower().strip(): v for k, v in objects_dict.items()}

    categories = {
        "floorplan": ["a floorplan"],
        "bedroom": ["a bed", "a single bed", "a double bed", "a duvet", "a bedroom"],
        "bathroom": ["a bath", "a bathroom basin", "a shower head", "a toilet seat", "a single ended bath", "a bathroom", "a tub"],
        "living/sitting room": ["a sofa", "a leather sofa", "a dinning table and chairs","a living room"],
        "outdoor": ["a garden", "a tree", "a car"],
        "kitchen": ["a refrigerator", "a kitchen stove", "a stove vent hood", "a kitchen", "a kitchen sink","a kitchen", "a cabinet and sink"],
        "gym": ["a gym", "a training machine"]

    }

    matched = []

    for room, objects in categories.items():
        # Normalize category objects
        normalized_objects = [obj.lower().strip() for obj in objects]
        if any(obj in detected for obj in normalized_objects):
            matched.append(room)

    if not matched:
        return "unknown"

    return " & ".join(matched) if len(matched) > 1 else matched[0]
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
        "bathroom": ["a bath", "a bathroom basin", "a shower head", "a toilet seat", "a single ended bath", "a bathroom"],
        "living/sitting room": ["a sofa", "a dinning table and chairs","a living room"],
        "outdoor": ["a garden", "a tree", "a car", "a building", "a sea", "a seaview"],
        "kitchen/cooking area": ["a refrigerator", "a kitchen stove", "a stove vent hood", "a kitchen", "a kitchen sink","a kitchen", "a cabinet and sink","a kitchen cabinet and sink","a sink tap","a cooking oven"],
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

    if room_scores.get("floorplan", 0) >= 1:
            return "floorplan"

     # ---------------------------
    # Rooms that pass threshold
    # ---------------------------
    valid_rooms = [
        room for room, score in room_scores.items()
        if score >= min_score and room != "floorplan"
    ]
    if not valid_rooms:
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

            # ------------------------------------------------------------
            # RUN DETECTION + CLEANLINESS
            # ------------------------------------------------------------
            # annotated_image, labels_dict = detect_objects_multiscale_annotate(
            #     image_path,
            #     n_scales=4,
            #     overlap_ratio=0.1,
            #     confidence_threshold=0.7
            # )
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

# def process_all_properties(root_dir):
#     """
#     Walk through the main directory and process all images
#     inside each property_* folder.
#     """
#     for property_folder in os.listdir(root_dir):
#         full_property_path = os.path.join(root_dir, property_folder)

#         if not os.path.isdir(full_property_path):
#             continue
        
#         print(f"\n📁 Processing folder: {property_folder}")

#         for file in os.listdir(full_property_path):
#             if not file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
#                 continue

#             image_path = os.path.join(full_property_path, file)

#             try:
#                 img = Image.open(image_path)
#                 w, h = img.size
#             except:
#                 print(f"  ⚠️ Failed to open: {file}")
#                 continue

#             if w < 300 or h < 400:
#                 print(f"  ⏩ Skipping {file} (too small: {w}x{h})")
#                 continue

#             print(f"  🖼️ Processing image: {file}")


#             ## detection on full image

#             labels_dict, conf_O, _ = classify_room(img)

#             if conf_O < 0.3:
#                 labels_dict = ""
#             # ------------------------------------------------------------
#             # DETECTION
#             # ------------------------------------------------------------
#             # annotated_image, labels_dict = detect_objects_multiscale_annotate(
#             #     image_path,
#             #     n_scales=4,
#             #     overlap_ratio=0.15,
#             #     confidence_threshold=0.85
#             # )


#             # CLIP classification
#             image = img.convert("RGB")
#             label, conf, clean_level, reason = classify_room_and_cleanliness(
#                 image, object_conf_threshold=0.9
#             )

#             # Clean fallback logic
#             if conf is None or (str(conf).replace(".", "").isdigit() and float(conf) < 0.9):
#                 label = "unknown"
#                 clean_level = "None"
#                 reason = "None"
#                 conf = "None"

#             # ------------------------------------------------------------
#             # ROOM TYPE FROM OBJECTS
#             # ------------------------------------------------------------
#             # room_type = classify_room_from_objects(labels_dict)
#             room_type = classify_room_from_objects_multi_label(labels_dict)

#             # ------------------------------------------------------------
#             # JSON OUTPUT
#             # ------------------------------------------------------------
#             json_output = {
#                 "image_name": file,
#                 "objects_detected": labels_dict,
#                 "primary_label": label,
#                 "primary_confidence": conf,
#                 "cleanliness_level": clean_level,
#                 "cleanliness_reason": reason,
#                 "room_type": room_type   # <-------- NEW KEY
#             }

#             # Save JSON
#             json_path = os.path.join(full_property_path, file.rsplit(".",1)[0] + ".json")
#             with open(json_path, "w") as f:
#                 json.dump(json_output, f, indent=4)

#             print(f"  ✅ Saved JSON: {json_path}")

#     print("\n🎉 All images processed!")


ROOT_IMAGE_DIR = "/mnt/c/Muzzi work/DAS4Whales-main/houseclassification/rightmove_images_Glasgow_09_12_2025"
ROOT_IMAGE_DIR = "/mnt/c/Muzzi work/DAS4Whales-main/houseclassification/rightmove_data_20_Glasgow/images/test"
ROOT_IMAGE_DIR = "/mnt/c/Muzzi work/test/s"
#ROOT_IMAGE_DIR = "/mnt/c/Muzzi work/test"

process_all_properties(ROOT_IMAGE_DIR)

