from __future__ import annotations

from pathlib import Path

import torch
import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms

from blueprintiq.models.detector import build_title_block_detector

app = FastAPI(title="BlueprintIQ API", version="0.1.0")


class PredictRequest(BaseModel):
    image_path: str
    score_threshold: float = 0.2


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_image_tensor(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return transforms.ToTensor()(image)


cfg = load_yaml(Path("blueprintiq/config/default.yaml"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_title_block_detector(num_classes=2)
ckpt_path = Path(cfg["training"]["checkpoint_dir"]) / cfg["training"]["checkpoint_name"]
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    image_path = Path(req.image_path)
    image_tensor = load_image_tensor(image_path)

    with torch.no_grad():
        pred = model([image_tensor.to(device)])[0]

    boxes = pred["boxes"].detach().cpu().tolist()
    scores = pred["scores"].detach().cpu().tolist()

    best_box = None
    best_score = 0.0

    for box, score in zip(boxes, scores):
        if score < req.score_threshold:
            continue
        if score > best_score:
            best_box = box
            best_score = score

    rounded_box = [round(x, 2) for x in best_box] if best_box is not None else None

    return {
        "image_path": str(image_path),
        "title_block_bbox": rounded_box,
        "score": round(best_score, 4),
    }