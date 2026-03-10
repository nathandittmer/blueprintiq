from __future__ import annotations

from pathlib import Path

import torch
import yaml

from blueprintiq.datasets.coco_detection_dataset import CocoDetectionDataset
from blueprintiq.models.detector import build_title_block_detector
from blueprintiq.training.eval_utils import box_iou_xyxy


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_yaml(Path("blueprintiq/config/default.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = CocoDetectionDataset(
        root_dir=cfg["data_gen"]["output_dir"],
        coco_json=f'{cfg["data_gen"]["output_dir"]}/coco_title_block.json',
    )

    model = build_title_block_detector(num_classes=2)
    ckpt_path = Path(cfg["training"]["checkpoint_dir"]) / cfg["training"]["checkpoint_name"]
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    model.to(device)
    model.eval()

    n_eval = min(5, len(ds))
    matches = 0

    with torch.no_grad():
        for i in range(n_eval):
            image, target = ds[i]
            pred = model([image.to(device)])[0]

            gt_boxes = target["boxes"].cpu().tolist()
            pred_boxes = pred["boxes"].detach().cpu().tolist()
            pred_scores = pred["scores"].detach().cpu().tolist()

            if not gt_boxes:
                print(f"sample={i} no ground truth boxes")
                continue

            gt_boxes = gt_boxes[0]

            best_iou = 0.0
            best_score = 0.0

            for box, score in zip(pred_boxes, pred_scores):
                if score < 0.5:
                    continue
                iou = box_iou_xyxy(box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_score = score

            if best_iou >= 0.5:
                matches += 1

            print(
                f"sample={i} best_iou={best_iou:.3f}"
                f"best_score={best_score:.3f} matched={best_iou >= 0.5}"
            )

    print(f" matched {matches}/{n_eval} samples at IoU>=0.5")


if __name__ == "__main__":
    main()