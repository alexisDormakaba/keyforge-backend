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


# =========================
# CONFIG
# =========================

SUPABASE_URL = os.getenv("SUPABASE_URLs")
SUPABASE_KEY = os.getenv("SUPABASE_PRIMARY_KEY")

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

app = FastAPI()


# =========================
# MODEL
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
# REQUESTS
# =========================

class ImageRequest(BaseModel):
    image_id: str
    image_url: str


class SearchRequest(BaseModel):
    image_url: str
    match_count: int = 10


# =========================
# ROOT
# =========================

@app.get("/")
def root():
    return {
        "status": "ok"
    }


# =========================
# ENV TEST
# =========================

@app.get("/env-test")
def env_test():
    return {
        "url_exists": SUPABASE_URLs is not None,
        "key_exists": SUPABASE_KEY is not None
    }


# =========================
# SUPABASE TEST
# =========================

@app.get("/supabase-test")
def supabase_test():

    result = (
        supabase
        .table("ingresos")
        .select("*")
        .limit(1)
        .execute()
    )

    return result.data


# =========================
# HELPER
# =========================

def generate_vector(image_url: str):

    response = requests.get(image_url)

    image = Image.open(
        BytesIO(response.content)
    ).convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0)

    with torch.no_grad():
        embedding = model(image_tensor)

    vector = embedding[0].numpy().tolist()

    return vector


# =========================
# GENERATE EMBEDDING
# =========================

@app.post("/generate-embedding")
def generate_embedding(data: ImageRequest):

    vector = generate_vector(
        data.image_url
    )

    vector_string = (
        "[" +
        ",".join(map(str, vector))
        + "]"
    )

    (
        supabase
        .table("ingresos")
        .update({
            "embedding": vector_string,
            "processed": True
        })
        .eq(
            "id",
            data.image_id
        )
        .execute()
    )

    return {
        "saved": True,
        "dimensions": len(vector)
    }


# =========================
# PROCESS PENDING
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

    processed_count = 0
    failed_count = 0

    for row in rows.data:

        try:

            vector = generate_vector(
                row["image_url"]
            )

            vector_string = (
                "[" +
                ",".join(map(str, vector))
                + "]"
            )

            (
                supabase
                .table("ingresos")
                .update({
                    "embedding": vector_string,
                    "processed": True
                })
                .eq(
                    "id",
                    row["id"]
                )
                .execute()
            )

            processed_count += 1

        except Exception as e:

            print(
                f"Error {row['id']}: {e}"
            )

            failed_count += 1

    return {
        "processed": processed_count,
        "failed": failed_count
    }


# =========================
# SEARCH
# =========================

@app.post("/search")
def search(data: SearchRequest):

    vector = generate_vector(
        data.image_url
    )

    vector_string = (
        "[" +
        ",".join(map(str, vector))
        + "]"
    )

    result = (
        supabase
        .rpc(
            "search_forges",
            {
                "query_embedding": vector_string,
                "match_count": data.match_count
            }
        )
        .execute()
    )

    return result.data
