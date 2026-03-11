from __future__ import annotations

from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw

from blueprintiq.datasets.coco_detection_dataset import CocoDetectionDataset
from blueprintiq.models.detector import build_title_block_detector


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def draw_box(draw, box, color, width=4):
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


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

    out_dir = Path("runs/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():

        for i in range(min(5,len(ds))):

            image, target = ds[i]

            pred = model([image.to(device)])[0]

            img = image.permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype("uint8")

            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)

            # draw ground truth
            for gt_box in target["boxes"].cpu().tolist():
                draw_box(draw, gt_box, "green", 6)

            for box, score in zip(
                pred["boxes"].detach().cpu().tolist(),
                pred["scores"].detach().cpu().tolist(),
            ):
                if score < 0.5:
                    continue
                draw_box(draw, box, "red", 4)

            out_path = out_dir / f"sample_{i}.png"
            img_pil.save(out_path)

            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()