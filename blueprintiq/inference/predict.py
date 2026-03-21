from __future__ import annotations

import json
from pathlib import Path

import torch
import typer
import yaml
from PIL import Image
from torchvision import transforms

from blueprintiq.models.detector import build_title_block_detector

app = typer.Typer(help="BlueprintIQ inference CLI")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_image_tensor(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return transforms.ToTensor()(image)


@app.command()
def predict(
    input: str = typer.Option(..., "--input", help="Path to input image"),
    config: str = typer.Option("blueprintiq/config/default.yaml", "--config", help="Path to YAML config"),
    score_threshold: float | None = typer.Option(None, "--score-threshold", help="Minimum score threshold"),
) -> None:
    cfg = load_yaml(Path(config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    effective_score_threshold = (
        score_threshold
        if score_threshold is not None
        else cfg["inference"]["score_threshold"]
    )

    model = build_title_block_detector(num_classes=2)
    ckpt_path = Path(cfg["training"]["checkpoint_dir"]) / cfg["training"]["checkpoint_name"]
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    image_path = Path(input)
    image_tensor = load_image_tensor(image_path)

    with torch.no_grad():
        pred = model([image_tensor.to(device)])[0]

    boxes = pred["boxes"].detach().cpu().tolist()
    scores = pred["scores"].detach().cpu().tolist()

    best_box = None
    best_score = 0.0

    for box, score in zip(boxes, scores):
        if score < effective_score_threshold:
            continue
        if score > best_score:
            best_box = box
            best_score = score

    
    rounded_box = [round(x, 2) for x in best_box] if best_box is not None else None

    result = {
        "image_path": str(image_path),
        "title_block_bbox": rounded_box,
        "score": round(best_score, 4),
        "score_threshold": effective_score_threshold,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()