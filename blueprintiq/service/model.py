from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

from blueprintiq.models.detector import build_title_block_detector
from blueprintiq.monitoring.logger import log_prediction


class ModelService:
    def __init__(self, config_path: str = "blueprintiq/config/default.yaml"):
        self.cfg = self._load_yaml(Path(config_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = build_title_block_detector(num_classes=2)
        ckpt_path = Path(self.cfg["training"]["checkpoint_dir"]) / self.cfg["training"]["checkpoint_name"]

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _load_yaml(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_image_tensor(self, image_path: Path):
        image = Image.open(image_path).convert("RGB")
        return transforms.ToTensor()(image)

    def predict(self, image_path: str, score_threshold: float = 0.2):
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_tensor = self._load_image_tensor(path)

        with torch.no_grad():
            pred = self.model([image_tensor.to(self.device)])[0]

        boxes = pred["boxes"].detach().cpu().tolist()
        scores = pred["scores"].detach().cpu().tolist()

        best_box = None
        best_score = 0.0

        for box, score in zip(boxes, scores):
            if score < score_threshold:
                continue
            if score > best_score:
                best_box = box
                best_score = score

        rounded_box = [round(x, 2) for x in best_box] if best_box else None

        result = {
            "image_path": str(path),
            "title_block_bbox": rounded_box,
            "score": round(best_score, 4),
        }
        
        log_prediction(
            {
                "image_path": str(path),
                "score_threshold": score_threshold,
                "title_block_bbox": rounded_box,
                "score": round(best_score, 4),
            }
        )

        return result