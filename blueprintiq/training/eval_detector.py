from __future__ import annotations

from pathlib import Path

import torch
import yaml
import json

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

    n_eval = min(cfg["eval"]["n_eval_samples"], len(ds))
    score_threshold = cfg["eval"]["score_threshold"]
    match_iou_threshold = cfg["eval"]["match_iou_threshold"]

    matches = 0
    samples_with_pred = 0
    iou_sum = 0.0

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

            gt_box = gt_boxes[0]

            filtered = [
                (box, score)
                for box, score in zip(pred_boxes, pred_scores)
                if score >= score_threshold
            ]

            if filtered:
                samples_with_pred += 1

            best_iou = 0.0
            best_score = 0.0

            for box, score in filtered:
                iou = box_iou_xyxy(box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_score = score

            iou_sum += best_iou

            matched = best_iou >= match_iou_threshold
            if matched:
                matches += 1

            print(
                f"sample={i} "
                f"n_preds_above_thresh={len(filtered)} "
                f"best_iou={best_iou:.3f} "
                f"best_score={best_score:.3f} "
                f"matched={matched}"
            )

    avg_best_iou = iou_sum / max(n_eval, 1)
    match_rate = matches / max(n_eval, 1)
    pred_rate = samples_with_pred / max(n_eval, 1)

    summary = {
        "score_threshold": score_threshold,
        "match_iou_threshold": match_iou_threshold,
        "samples_evaluated": n_eval,
        "samples_with_prediction": samples_with_pred,
        "prediction_rate": pred_rate,
        "matched_samples": matches,
        "match_rate": match_rate,
        "average_best_iou": avg_best_iou,
        "model_version": ckpt.get("model_version", "unknown"),
        "model_description": ckpt.get("model_description", "unknown"),
        "trained_epochs": ckpt.get("num_epochs", None),
    }

    print("\n=== Evaluation Summary ===")
    print(f"score_threshold={score_threshold}")
    print(f"match_iou_threshold={match_iou_threshold}")
    print(f"samples_evaluated={n_eval}")
    print(f"samples_with_prediction={samples_with_pred}/{n_eval} ({pred_rate:.2%})")
    print(f"matched_samples={matches}/{n_eval} ({match_rate:.2%})")
    print(f"average_best_iou={avg_best_iou:.4f}")

    out_dir = Path("runs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "eval_report.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote evaluation report to {out_path}")


if __name__ == "__main__":
    main()