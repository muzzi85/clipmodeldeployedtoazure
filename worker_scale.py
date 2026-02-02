import os
import json
import time
import base64
from io import BytesIO
from collections import defaultdict
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from azure.storage.queue import QueueClient
from azure.storage.blob import BlobServiceClient

# =========================================================
# 1. ENV + GPU SETUP
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {DEVICE}")

QUEUE_NAME = os.environ["QUEUE_NAME"]
STORAGE_ACCOUNT_URL = os.environ["STORAGE_ACCOUNT_URL"]
STORAGE_CONNECTION_STRING = os.environ["STORAGE_CONNECTION_STRING"]

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", 2))

# =========================================================
# 2. LOAD CLIP ONCE (CRITICAL)
# =========================================================
print("📦 Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print("✅ CLIP loaded")

# =========================================================
# 3. LABEL PROMPTS
# =========================================================
OBJECT_LABELS = [
    "a photo of a single bed", "a photo of a double bed", "a photo of a bed",
    "a photo of a bathroom basin", "a photo of a bath", "a photo of a refrigerator",
    "a photo of a sofa", "a photo of a television", "a photo of a toilet seat",
    "a photo of a kitchen sink", "a photo of a kitchen stove",
    "a photo of stairs", "a photo of a garden", "a photo of a window",
    "a photo of curtains", "a photo of an unknown room"
]

CLEAN_PROMPTS = [
    "This is a very clean room.",
    "This is a somewhat messy room.",
    "This is a dirty room."
]

# =========================================================
# 4. AZURE CLIENTS
# =========================================================
queue_client = QueueClient.from_connection_string(
    STORAGE_CONNECTION_STRING, QUEUE_NAME
)

blob_service = BlobServiceClient.from_connection_string(
    STORAGE_CONNECTION_STRING
)

# =========================================================
# 5. CLIP BATCH INFERENCE
# =========================================================
@torch.no_grad()
def classify_batch(images):
    inputs = processor(
        text=OBJECT_LABELS,
        images=images,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)

    results = []
    for i in range(len(images)):
        idx = torch.argmax(probs[i]).item()
        results.append({
            "label": OBJECT_LABELS[idx],
            "confidence": float(probs[i, idx])
        })
    return results


@torch.no_grad()
def classify_cleanliness(images):
    inputs = processor(
        text=CLEAN_PROMPTS,
        images=images,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)

    results = []
    for i in range(len(images)):
        idx = torch.argmax(probs[i]).item()
        results.append({
            "level": idx + 1,
            "reason": CLEAN_PROMPTS[idx]
        })
    return results

# =========================================================
# 6. MAIN WORK LOOP
# =========================================================
def main():
    print("🟢 GPU worker started")

    while True:
        messages = queue_client.receive_messages(
            messages_per_page=BATCH_SIZE,
            visibility_timeout=300
        )

        batch_msgs = []
        batch_images = []
        blob_refs = []

        for msg in messages:
            body = json.loads(
                base64.b64decode(msg.content).decode("utf-8")
            )

            container = body["container"]
            blob_path = body["blob_path"]

            blob_client = blob_service.get_blob_client(
                container=container,
                blob=blob_path
            )

            try:
                data = blob_client.download_blob().readall()
                img = Image.open(BytesIO(data)).convert("RGB")
            except Exception as e:
                print(f"⚠️ Failed to load {blob_path}: {e}")
                queue_client.delete_message(msg)
                continue

            batch_msgs.append(msg)
            batch_images.append(img)
            blob_refs.append((container, blob_path))

        if not batch_images:
            time.sleep(POLL_SECONDS)
            continue

        # 🔥 GPU BATCH INFERENCE
        obj_results = classify_batch(batch_images)
        clean_results = classify_cleanliness(batch_images)

        # ✍️ WRITE JSON BACK NEXT TO IMAGE
        for (container, blob_path), obj, clean in zip(
            blob_refs, obj_results, clean_results
        ):
            json_blob = blob_path.rsplit(".", 1)[0] + ".json"

            result = {
                "primary_label": obj["label"],
                "confidence": obj["confidence"],
                "cleanliness_level": clean["level"],
                "cleanliness_reason": clean["reason"]
            }

            blob_service.get_blob_client(
                container=container,
                blob=json_blob
            ).upload_blob(
                json.dumps(result, indent=2),
                overwrite=True
            )

        # ✅ ACK QUEUE MESSAGES
        for msg in batch_msgs:
            queue_client.delete_message(msg)

        print(f"✅ Processed batch of {len(batch_msgs)} images")


if __name__ == "__main__":
    main()
