from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class CocoAnn:
    id: int
    image_id: int
    category_id: int
    bbox: list[float]  # [x,y,w,h]
    area: float
    iscrowd: int


class CocoDetectionDataset:
    """
    Minimal COCO loader for single-class detection (title_block).
    Returns:
      image: PIL.Image
      target: dict with boxes (xyxy), labels, image_id
    """

    def __init__(self, root_dir: str, coco_json: str):
        self.root_dir = Path(root_dir)
        self.coco_path = Path(coco_json)

        coco = json.loads(self.coco_path.read_text(encoding="utf-8"))
        self.images = [CocoImage(**im) for im in coco["images"]]
        self.anns = [CocoAnn(**ann) for ann in coco["annotations"]]

        # index annotations by image_id
        self.ann_by_image: dict[int, list[CocoAnn]] = {}
        for ann in self.anns:
            self.ann_by_image.setdefault(ann.image_id, []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Image.Image, dict[str, Any]]:
        im = self.images[idx]
        img_path = self.root_dir / im.file_name
        image = Image.open(img_path).convert("RGB")

        anns = self.ann_by_image.get(im.id, [])
        boxes_xyxy = []
        labels = []
        for a in anns:
            x, y, w, h = a.bbox
            boxes_xyxy.append([x, y, x + w, y + h])
            labels.append(a.category_id)

        target = {
            "image_id": im.id,
            "boxes": boxes_xyxy,
            "labels": labels,
        }
        return image, target