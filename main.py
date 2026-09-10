from fastapi import FastAPI
from pydantic import BaseModel
import requests
from PIL import Image
from io import BytesIO

import torch
from torchvision import models
from torchvision import transforms

app = FastAPI()

model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)
model.classifier = torch.nn.Identity()
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class ImageRequest(BaseModel):
    image_id: str
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

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model(image_tensor)

    vector = embedding[0].numpy().tolist()
    vector_string = "[" + ",".join(
    map(str, vector)
) + "]"

    supabase.table("images").update({
    "embedding": vector_string,
    "processed": True
}).eq(
    "id",
    data.image_id
).execute()


    return {
        "saved": True,
        "dimensions": len(vector)
    }
