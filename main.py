from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ImageRequest(BaseModel):
    image_url: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/test-image")
def test_image(data: ImageRequest):
    return {
        "received_url": data.image_url
    }
