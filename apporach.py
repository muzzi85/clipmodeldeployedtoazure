import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# 1. Load CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

labels = [
    "a photo of a single bed",
    "a photo of a double bed",
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
    "a photo of an unknown room",
    "a photo of a floorplan"

]
# 3. Inference function
def classify_room(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return labels[pred], probs[0][pred].item(),inputs

# Test
image_path = "/home/muzzi/houseroomsclassification/rightmove2/bathroom/fllor.jpg"

room, confidence, vit_output = classify_room(image_path)
if confidence > 0.5:
    print(f"Predicted: {room} ({confidence*100:.2f}%)")
else:
    d="unknown"
    print(f"Predicted: {room+d} ({confidence*100:.2f}%)")

#print("vit_output",vit_output)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread(image_path)
imgplot = plt.imshow(img)
plt.show()



### partition the image and run trained CLIP

import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# 1. Load CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

labels = [
    "a photo of a single bed",
    "a photo of a double bed",
    "a photo of a bathroom basin",
    "a photo of a bath",
    "a photo of a refergerator",
    "a photo of a duvet",
    "a photo of a pillow",
    "a photo of a television",
    "a photo of a washing machine",
    "a photo of a microwave",
    "a photo of a kattle",
    "a photo of a sink tap",
    "a photo of a fire alarm",
    "a photo of a oven",
    "a photo of a river",
    "a photo of a ship",
    "a photo of a building",
    "a photo of a waredrobe",
    "a photo of a door",
    "a photo of a window",
    "a photo of a wooden floor",
    "a photo of a carpet floor",
    "a photo of a tiles flooring",
    "a photo of a Single Ended Bath",
    "a photo of a stove vent hood",
    "a photo of a stove",
    "a photo of a sofa",
    "a photo of a shower head",
    "a photo of a heater radiator",
    "a photo of a kitchen stove",
    "a photo of a kitchen cabinets",
    "a photo of a toilet seat ",
    "a photo of a kitchen sink",
    "a photo of a dinning table and chairs",
    "a photo of stairs",
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



# Test of cropping
from PIL import Image
image_path = "/home/muzzi/houseroomsclassification/rightmove2/bathroom/173825_711443_IMG_04_0000.jpeg"
imagee = Image.open(image_path).convert("RGB")

print(imagee.size)
print(type(imagee))
## generate splits per 100 pixel

array_of_split = []

for x in range(0, imagee.size[0]-100, 100):
    for y in range(0, imagee.size[1]-100, 100):
        sample_split = ()
img_left_area = (0, 0, 400, 600)
img_right_area = (400, 400, 500, 600)

img_left = imagee.crop(img_left_area)
img_right = imagee.crop(img_right_area)

plt.imshow(img_left)
plt.figure()
plt.imshow(img_right)
plt.figure()
plt.imshow(imagee)

room, confidence, vit_output = classify_room(img_right)
if confidence > 0.5:
    print(f"Predicted: {room} ({confidence*100:.2f}%)")
else:
    d="unknown"
    print(f"Predicted: {room+d} ({confidence*100:.2f}%)")

#print("vit_output",vit_output)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread(image_path)
imgplot = plt.imshow(img)
plt.show()


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
image_path = "/home/muzzi/houseroomsclassification/rightmove2/bathroom/173825_711443_IMG_10_0000.jpeg" # nothing
image_path = "/home/muzzi/houseroomsclassification/rightmove/bathroom/187742_LOWHILL280_IMG_00_0000.jpeg" # bedroom with tbale and chair 
image_path = "/home/muzzi/houseroomsclassification/rightmove/bathroom/187742_LOWHILL280_IMG_04_0000.jpeg" # little details of living room, just one sofa and little table
image_path = "/home/muzzi/houseroomsclassification/rightmove/bathroom/187742_LOWHILL280_IMG_47_0000.jpeg" # kitchen details and living room sofa in one picture
image_path = "/home/muzzi/houseroomsclassification/rightmove/bathroom/187742_LOWHILL280_IMG_51_0000.jpeg" # small kitchen
image_path = "/home/muzzi/houseroomsclassification/rightmove/bathroom/187742_LOWHILL280_IMG_48_0000.jpeg" # bedroom with small chair
image_path = "/home/muzzi/houseroomsclassification/rightmove/bathroom/187742_LOWHILL280_IMG_08_0000.jpeg" # washing machines
image_path = "/home/muzzi/houseroomsclassification/rightmove2/bathroom/173825_711443_IMG_02_0000.jpeg" # empty kitchen area with little stove and little sink and tap
image_path = "/home/muzzi/houseroomsclassification/rightmove1/73586_house_tobacco_warehouse_stanley_dock_regent_rd_liverpool_l3_0bl_2L_IMG_02_0000.jpeg" # empty kitchen area with little stove and little sink and tap
image_path = "/home/muzzi/houseroomsclassification/rightmove1/73586_house_tobacco_warehouse_stanley_dock_regent_rd_liverpool_l3_0bl_2L_IMG_04_0000.jpeg" # nothing just buildings from top
image_path = "/home/muzzi/houseroomsclassification/rightmove1/73586_house_tobacco_warehouse_stanley_dock_regent_rd_liverpool_l3_0bl_2L_IMG_06_0000.jpeg" # stairs and kitchen cabinets from distance
image_path = "/home/muzzi/houseroomsclassification/rightmove3/4955_MAG240625_IMG_07_0000.jpeg" # kitchen cabinets from close distance
image_path = "/home/muzzi/houseroomsclassification/rightmove3/4955_MAG240625_IMG_12_0000.jpeg" # bathroom from close distance
image_path = "/home/muzzi/houseroomsclassification/rightmove3/54487_DOR1002175_IMG_07_0000.jpeg" # garden outside a house
image_path = "/home/muzzi/houseroomsclassification/rightmove3/54487_DOR1002175_IMG_06_0000.jpeg" # full bathroom content
image_path = "/home/muzzi/houseroomsclassification/rightmove3/54487_DOR1002175_IMG_01_0000.jpeg" # corridor between rooms 
image_path = "/home/muzzi/houseroomsclassification/rightmove3/54487_DOR1002175_IMG_03_0000.jpeg" # kitchen with stove and sink
image_path = "/home/muzzi/houseroomsclassification/rightmove3/54487_DOR1002175_IMG_04_0000.jpeg" # empty room and carpet flooring
image_path = "/home/muzzi/houseroomsclassification/rightmove4/52974_7440321_IMG_01_0000.jpeg" # view in front of a building with seaview
#image_path = "/home/muzzi/houseroomsclassification/rightmove4/224933_A16_IMG_06_0000 (1).jpeg" # modern bedroom with chair


cropss = crop_image_with_overlap(image_path)

## start the game

## multi cripping rate functionality

## biggest rate
cropss = crop_image_with_overlap(image_path, crop_size=600, overlap_ratio=0.9)
cropss[0].size
print(len(cropss))
for i in range(len(cropss)):
   
    room, confidence, vit_output = classify_room(cropss[i])
    if confidence > 0.5 and room != "a photo of an unknown room":# and room != "a photo of a kitchen cabinets"
        plt.figure()
        plt.imshow(cropss[i])

        print(f"Predicted: {room} ({confidence*100:.2f}%)")
        plt.title(room)
    else:
        d="unknown"
        #print(f"Predicted: {room+d} ({confidence*100:.2f}%)")
        #plt.title(d)

## big rate
cropss = crop_image_with_overlap(image_path, crop_size=500, overlap_ratio=0.9)
cropss[0].size
print(len(cropss))
for i in range(len(cropss)):
   
    room, confidence, vit_output = classify_room(cropss[i])
    if confidence > 0.8 and room != "a photo of an unknown room":# and room != "a photo of a kitchen cabinets"
        plt.figure()
        plt.imshow(cropss[i])

        print(f"Predicted: {room} ({confidence*100:.2f}%)")
        plt.title(room)
    else:
        d="unknown"
        #print(f"Predicted: {room+d} ({confidence*100:.2f}%)")
        #plt.title(d)

## medium rate
cropss = crop_image_with_overlap(image_path, crop_size=256, overlap_ratio=0.5)
cropss[0].size
print(len(cropss))
for i in range(len(cropss)):
   
    room, confidence, vit_output = classify_room(cropss[i])
    if confidence > 0.9 and room != "a photo of an unknown room" :#and room != "a photo of a kitchen cabinets"
        plt.figure()
        plt.imshow(cropss[i])

        print(f"Predicted: {room} ({confidence*100:.2f}%)")
        plt.title(room)
    else:
        d="unknown"
        #print(f"Predicted: {room+d} ({confidence*100:.2f}%)")
        #plt.title(d)

## small rate
cropss = crop_image_with_overlap(image_path, crop_size=125, overlap_ratio=0.3)
cropss[0].size
print(len(cropss))
for i in range(len(cropss)):
    room, confidence, vit_output = classify_room(cropss[i])
    if confidence > 0.8 and room != "a photo of an unknown room" :#and room != "a photo of a kitchen cabinets"
        plt.figure()
        plt.imshow(cropss[i])

        print(f"Predicted: {room} ({confidence*100:.2f}%)")
        plt.title(room)
    else:
        d="unknown"
        #print(f"Predicted: {room+d} ({confidence*100:.2f}%)")
        #plt.title(d)

## very small rate
cropss = crop_image_with_overlap(image_path, crop_size=75, overlap_ratio=0.2)
cropss[0].size
print(len(cropss))
for i in range(len(cropss)):
    room, confidence, vit_output = classify_room(cropss[i])
    if confidence > 0.5 and room != "a photo of an unknown room" and room != "a photo of a kitchen cabinets" and room != "a photo of a tiles flooring" and room != "a photo of a wooden floor":
        plt.figure()
        plt.imshow(cropss[i])

        print(f"Predicted: {room} ({confidence*100:.2f}%)")
        plt.title(room)
    else:
        d="unknown"
        #print(f"Predicted: {room+d} ({confidence*100:.2f}%)")
        #plt.title(d)

### detect interest areas

import cv2
import numpy as np
import numpy as np
from PIL import ImageDraw
def find_high_entropy_regions(image, num_regions=5, crop_ratio=0.15):
    gray = np.array(image.convert("L"))
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    score_map = cv2.convertScaleAbs(np.abs(lap))
    h, w = gray.shape
    crop_w, crop_h = int(w*crop_ratio), int(h*crop_ratio)

    regions = []
    for _ in range(num_regions):
        y, x = np.unravel_index(np.argmax(score_map), score_map.shape)
        x1, y1 = max(0, x-crop_w//2), max(0, y-crop_h//2)
        x2, y2 = min(w, x+crop_w//2), min(h, y+crop_h//2)
        regions.append((x1,y1,x2,y2))
        score_map[y1:y2, x1:x2] = 0  # suppress region

    return regions

# Load image
image = Image.open(image_path).convert("RGB")

# Find high-entropy regions
regions = find_high_entropy_regions(image, num_regions=15, crop_ratio=0.15)

# Draw rectangles on the image
annotated = image.copy()
draw = ImageDraw.Draw(annotated)
for i, (x1, y1, x2, y2) in enumerate(regions):
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    draw.text((x1, y1-15), f"Region {i+1}", fill="red")

# Visualize
plt.figure(figsize=(10, 6))
plt.imshow(annotated)
plt.axis("off")
plt.title("High-Entropy Regions")
plt.show()


### Kitchen only run classification into the regions
labels = [
    "a photo of a refrigerator",
    "a photo of a oven",
    "a photo of a shower head",
    "a photo of a heater radiator",
    "a photo of a stove",
    "a photo of a kitchen sink",
]
# 3. Inference function
def classify_room_kitchen(image):
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return labels[pred], probs[0][pred].item(),inputs

# Assume `image` is your PIL.Image and `regions` is from find_high_entropy_regions
for i, (x1, y1, x2, y2) in enumerate(regions):
    # Crop the high-entropy region
    crop = image.crop((x1, y1, x2, y2))
    
    # Classify the region
    label, confidence, _ = classify_room_kitchen(crop)
    
    # Optional: filter low-confidence or unknown labels
    if confidence > 0.27 and label != "a photo of an unknown room":
        print(f"Region {i+1}: {label} ({confidence*100:.1f}%)")
        # Show crop
        plt.figure()
        plt.imshow(crop)
        plt.axis("off")
        plt.title(f"{label} ({confidence*100:.1f}%)")
        plt.show()
    else:
        print(f"Region {i+1}: Unknown / Low confidence")



### one function that does :

# Takes an input image.

# Crops it into overlapping patches.

# Runs each crop through your classify_room function.

# Draws the predicted label on the top of each crop (if confidence is above a threshold).

# Recombines all cropped images back into the original image size.

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
def detect_objects_and_annotate(image_path, crop_size=600, overlap_ratio=0.9, confidence_threshold=0.8):
    """
    Crop an image, detect objects in each crop, annotate with cartoon-style text in the center, and recombine into full image.
    
    Args:
        image_path (str): Path to input image.
        crop_size (int): Size of square crops.
        overlap_ratio (float): Overlap ratio between crops (0 to 1).
        confidence_threshold (float): Only annotate objects above this confidence.
        
    Returns:
        PIL.Image: Annotated recombined image.
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    stride = int(crop_size * (1 - overlap_ratio))
    
    # Calculate crop positions
    x_steps = list(range(0, width - crop_size + 1, stride))
    y_steps = list(range(0, height - crop_size + 1, stride))
    if x_steps[-1] + crop_size < width:
        x_steps.append(width - crop_size)
    if y_steps[-1] + crop_size < height:
        y_steps.append(height - crop_size)

    # Create a blank image to paste annotated crops
    recombined = Image.new("RGB", (width, height))
    
    # Load a fun font (make sure this TTF file exists in your system or project folder)
    try:
        font = ImageFont.truetype("ComicSansMS.ttf", 36)  # Bigger and fun font
    except:
        font = ImageFont.load_default()
    
    for top in y_steps:
        for left in x_steps:
            box = (left, top, left + crop_size, top + crop_size)
            crop = image.crop(box)
            
            # Classify crop
            label, conf, _ = classify_room(crop)
            
            draw = ImageDraw.Draw(crop)
            
            if conf > confidence_threshold and label != "a photo of an unknown room":
                text = f"{label} ({conf*100:.1f}%)"
                
                # Calculate text size
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Calculate coordinates to center text
                x_text = (crop_size - text_width) // 2
                y_text = (crop_size - text_height) // 2
                
                # Draw semi-transparent rectangle behind text
                padding = 10
                draw.rectangle(
                    [x_text - padding//2, y_text - padding//2, x_text + text_width + padding//2, y_text + text_height + padding//2],
                    fill=(0, 0, 0, 150)
                )
                
                # Cartoon-style outline
                outline_range = [-2, -1, 0, 1, 2]
                for dx in outline_range:
                    for dy in outline_range:
                        if dx != 0 or dy != 0:
                            draw.text((x_text + dx, y_text + dy), text, font=font, fill="black")
                
                # Draw main text in bright color
                draw.text((x_text, y_text), text, font=font, fill="yellow")
            
            # Paste crop back into recombined image
            recombined.paste(crop, (left, top))
    
    return recombined


# Example usage:
annotated_image = detect_objects_and_annotate("/home/muzzi/houseroomsclassification/rightmove3/54487_DOR1002175_IMG_06_0000.jpeg", crop_size=125, overlap_ratio=0.2)
plt.figure(figsize=(12, 12))
plt.imshow(annotated_image)
plt.axis('off')
plt.show()

### multi crop sizes

from PIL import Image, ImageDraw, ImageFont

def detect_objects_multiscale_annotate(
    image_path, 
    n_scales=4, 
    overlap_ratio=0.1, 
    confidence_threshold=0.7
):
    """
    Multi-scale object detection + annotation using adaptive crops.
    
    Args:
        image_path (str): Path to input image.
        n_scales (int): Number of crop scales (default: 4).
        overlap_ratio (float): Overlap between crops (0–1).
        confidence_threshold (float): Minimum confidence for labeling.
    
    Returns:
        PIL.Image: Annotated combined image.
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

    detections = []  # store (x, y, label, conf)

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
                    detections.append((center_x, center_y, label, conf))
    
    # Draw detections on combined image
    for (cx, cy, label, conf) in detections:
        text = f"{label.split('a photo of ')[-1]} ({conf*100:.1f}%)"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # Draw background rectangle
        draw.rectangle(
            [(cx - tw//2 - 6, cy - th//2 - 6), (cx + tw//2 + 6, cy + th//2 + 6)],
            fill=(0, 0, 0, 160)
        )
        # Draw text centered
        draw.text((cx - tw//2, cy - th//2), text, fill="white", font=font)

    print(f"✅ Total detections: {len(detections)}")
    return annotated


image_path = "/home/muzzi/houseroomsclassification/rightmove1/73586_house_tobacco_warehouse_stanley_dock_regent_rd_liverpool_l3_0bl_2L_IMG_01_0000.jpeg"

annotated_img = detect_objects_multiscale_annotate(
    image_path,
    n_scales=16,
    overlap_ratio=0.15,
    confidence_threshold=0.7
)

plt.imshow(annotated_img)


### cleaness

# Existing object labels (shortened for clarity)
object_labels = [
    "a photo of a single bed",
    "a photo of a double bed",
    "a photo of a bathroom basin",
    "a photo of a bath",
    "a photo of a refergerator",
    "a photo of a duvet",
    "a photo of a pillow",
    "a photo of a television",
    "a photo of a washing machine",
    "a photo of a microwave",
    "a photo of a kattle",
    "a photo of a sink tap",
    "a photo of a fire alarm",
    "a photo of a oven",
    "a photo of a river",
    "a photo of a ship",
    "a photo of a building",
    "a photo of a waredrobe",
    "a photo of a door",
    "a photo of a window",
    "a photo of a wooden floor",
    "a photo of a carpet floor",
    "a photo of a tiles flooring",
    "a photo of a Single Ended Bath",
    "a photo of a stove vent hood",
    "a photo of a stove",
    "a photo of a sofa",
    "a photo of a shower head",
    "a photo of a heater radiator",
    "a photo of a kitchen stove",
    "a photo of a kitchen cabinets",
    "a photo of a toilet seat ",
    "a photo of a kitchen sink",
    "a photo of a dinning table and chairs",
    "a photo of stairs",
    "a photo of an unknown room"
]
# New: cleanliness prompts
cleanliness_prompts = [
    "This is a very clean room, everything is tidy and organized.",   # 1
    "This is a somewhat messy room, there are a few items scattered.", # 2
    "This is a dirty room, cluttered and untidy with visible mess."    # 3
]


def classify_room_and_cleanliness(image):
    """
    Classify room object and estimate cleanliness (1=clean, 3=dirty).
    Returns:
        - best_label: str (object detected)
        - obj_conf: float (confidence for label)
        - cleanliness_level: int (1–3)
        - cleanliness_reason: str
    """
    # Process for both object detection and cleanliness
    obj_inputs = processor(text=object_labels, images=image, return_tensors="pt", padding=True)
    clean_inputs = processor(text=cleanliness_prompts, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        obj_outputs = model(**obj_inputs)
        clean_outputs = model(**clean_inputs)

        # Object detection
        obj_probs = obj_outputs.logits_per_image.softmax(dim=1)
        obj_pred = torch.argmax(obj_probs, dim=1).item()
        best_label = object_labels[obj_pred]
        obj_conf = obj_probs[0, obj_pred].item()

        # Cleanliness
        clean_probs = clean_outputs.logits_per_image.softmax(dim=1)
        clean_pred = torch.argmax(clean_probs, dim=1).item()
        cleanliness_level = clean_pred + 1  # 1 to 3
        cleanliness_reason = cleanliness_prompts[clean_pred]
    
    return best_label, obj_conf, cleanliness_level, cleanliness_reason


image = Image.open(image_path).convert("RGB")
label, conf, clean_level, reason = classify_room_and_cleanliness(image)

print(f"Object: {label} ({conf*100:.1f}%)")
print(f"Cleanliness: {clean_level}/3")
print(f"Reason: {reason}")

plt.imshow(image)


## if kitchen was definied run this 

from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch
from PIL import Image

processor1 = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model1 = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

texts = [["stove", "sink", "oven", "refrigerator", "table"]]
inputs = processor1(text=texts, images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model1(**inputs)

target_sizes = torch.Tensor([image.size[::-1]])
results = processor1.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score > 0.01:
        print(texts[0][label], score.item(), box.tolist())

## if kitchen was definied run this 

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch


def check_stove_presence(image):
    """
    Estimate whether a stove is visible in a kitchen image using CLIP similarity.
    Returns a tuple (has_stove, confidence_score)
    """
    prompts = ["a stove is visible in the kitchen", "no stove is visible"]
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # (1, 2)
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    has_stove = probs[0] > probs[1]
    confidence = probs[0] if has_stove else probs[1]

    return has_stove, confidence, probs

# Example usage:
has_stove, confidence, probs = check_stove_presence(image)
print("Stove visible:", has_stove, "Confidence:", confidence)


### stove location

import numpy as np
from PIL import ImageDraw

def highlight_stove_location(image, crop_size=128, overlap=0.5, threshold=0.6):
    """
    Scans image with sliding window using CLIP to find the stove location.
    Returns annotated image and stove confidence map.
    """
    w, h = image.size
    stride = int(crop_size * (1 - overlap))
    
    prompts = ["a stove is visible", "no stove is visible"]

    # Storage for confidence heatmap
    heatmap = np.zeros((h, w), dtype=np.float32)
    countmap = np.zeros((h, w), dtype=np.float32)

    for top in range(0, h - crop_size + 1, stride):
        for left in range(0, w - crop_size + 1, stride):
            crop = image.crop((left, top, left + crop_size, top + crop_size))
            inputs = processor(text=prompts, images=crop, return_tensors="pt", padding=True)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)[0]
                stove_prob = probs[0].item()  # probability of "stove visible"
            
            # Fill the heatmap region
            heatmap[top:top+crop_size, left:left+crop_size] += stove_prob
            countmap[top:top+crop_size, left:left+crop_size] += 1

    # Average overlapping areas
    heatmap /= np.maximum(countmap, 1e-5)

    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-5)

    # Find the region of highest probability
    max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    y, x = max_idx

    # Draw bounding box
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    box = (x - crop_size//2, y - crop_size//2, x + crop_size//2, y + crop_size//2)
    draw.rectangle(box, outline="red", width=4)
    draw.text((x - crop_size//2, y - crop_size//2 - 20), "🔥 Stove detected", fill="red")

    return annotated, heatmap


from PIL import Image
import matplotlib.pyplot as plt

# Run stove localization
annotated_image, heatmap = highlight_stove_location(image, crop_size=128, overlap=0.5, threshold=0.6)

# Show annotated image with red bounding box
plt.figure(figsize=(10,6))
plt.imshow(annotated_image)
plt.axis("off")
plt.title("Stove detection via CLIP sliding window")
plt.show()

# Optional: visualize stove heatmap
plt.figure(figsize=(10,6))
plt.imshow(heatmap, cmap="hot")
plt.colorbar(label="Stove probability")
plt.title("Stove probability heatmap")
plt.show()

