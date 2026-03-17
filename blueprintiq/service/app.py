from __future__ import annotations

from pathlib import Path

import torch
import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms

from blueprintiq.models.detector import build_title_block_detector
from blueprintiq.service.model import ModelService

app = FastAPI(title="BlueprintIQ API", version="0.1.0")
model_service = ModelService()


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
    try:
        return model_service.predict(
            image_path=req.image_path,
            score_threshold=req.score_threshold,
        )
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"unexpected error: {str(e)}"}