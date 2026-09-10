from fastapi import FastAPI
from pydantic import BaseModel
import os

from supabase import create_client

import requests
from PIL import Image
from io import BytesIO

import torch
from torchvision import models
from torchvision import transforms

from rembg import remove

# =========================
# Supabase
# =========================

supabase = create_client(
    os.getenv("SUPABASE_URLs"),
    os.getenv("SUPABASE_PRIMARY_KEY")
)

# =========================
# FastAPI
# =========================

app = FastAPI()

# =========================
# Modelo EfficientNet
# =========================

model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)

model.classifier = torch.nn.Identity()
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# Request model
# =========================

class ImageRequest(BaseModel):
    image_id: str
    image_url: str


# =========================
# Health Check
# =========================

@app.get("/")
def root():
    return {"status": "ok"}

# =========================
# Variables Test
# =========================

@app.get("/env-test")
def env_test():
    return {
        "url_exists": os.getenv("SUPABASE_URLs") is not None,
        "key_exists": os.getenv("SUPABASE_PRIMARY_KEY") is not None
    }

# =========================
# Supabase Test
# =========================

@app.get("/supabase-test")
def supabase_test():

    result = (
        supabase
        .table("ingresos")
        .select("id")
        .limit(1)
        .execute()
    )

    return {
        "success": True,
        "rows": len(result.data)
    }

# =========================
# Generate Embedding
# =========================

@app.post("/generate-embedding")
def generate_embedding(data: ImageRequest):

    print(f"Procesando: {data.image_id}")

    response = requests.get(data.image_url)

    if response.status_code != 200:
        return {
            "saved": False,
            "error": "Image download failed"
        }

    # =========================
    # Remove background
    # =========================

    image_without_bg = remove(
        response.content
    )

    image = Image.open(
        BytesIO(image_without_bg)
    ).convert("RGB")

    # =========================
    # Crop transparent borders
    # =========================

    bbox = image.getbbox()

    if bbox:
        image = image.crop(bbox)

    # =========================
    # Prepare image
    # =========================

    image_tensor = transform(
        image
    ).unsqueeze(0)

    # =========================
    # Generate embedding
    # =========================

    with torch.no_grad():
        embedding = model(image_tensor)

    vector = embedding[0].numpy().tolist()

    print(
        f"Embedding generado: {len(vector)} dimensiones"
    )

    vector_string = (
        "[" +
        ",".join(map(str, vector))
        + "]"
    )

    # =========================
    # Save to Supabase
    # =========================

    (
        supabase
        .table("ingresos")
        .update({
            "embedding": vector_string,
            "processed": True
        })
        .eq("id", data.image_id)
        .execute()
    )

    return {
        "saved": True,
        "dimensions": len(vector)
    }

# =========================
# Process all pending
# =========================

@app.post("/process-pending")
def process_pending():

    rows = (
        supabase
        .table("ingresos")
        .select("id,image_url")
        .eq("processed", False)
        .execute()
    )

    total = 0

    for row in rows.data:

        try:

            response = requests.post(
                "http://127.0.0.1:8000/generate-embedding",
                json={
                    "image_id": row["id"],
                    "image_url": row["image_url"]
                }
            )

            if response.status_code == 200:
                total += 1

        except Exception as e:
            print(e)

    return {
        "processed": total
    }
