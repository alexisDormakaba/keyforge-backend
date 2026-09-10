from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

class ImageRequest(BaseModel):
    image_url: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/analyze-image")
def analyze_image(data: ImageRequest):

    response = requests.get(data.image_url)

    return {
        "status_code": response.status_code,
        "size_bytes": len(response.content)
    }
