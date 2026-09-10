from fastapi import FastAPI
from pydantic import BaseModel

import requests
from PIL import Image
from io import BytesIO

import torch
import open_clip

app = FastAPI()

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

class ImageRequest(BaseModel):
    image_url: str


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/generate-embedding")
def generate_embedding(data: ImageRequest):

    response = requests.get(data.image_url)

    image = Image.open(
        BytesIO(response.content)
    ).convert("RGB")

    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model.encode_image(image_tensor)

    vector = embedding[0].cpu().numpy().tolist()

    return {
        "dimensions": len(vector),
        "embedding": vector[:10]
    }
