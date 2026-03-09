from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from blueprintiq.datasets.coco_detection_dataset import CocoDetectionDataset
from blueprintiq.models.detector import build_title_block_detector
from blueprintiq.training.utils import detection_collate_fn

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

    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=detection_collate_fn,
    )

    model = build_title_block_detector(num_classes=2)
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    running_loss = 0.0
    step_count = 0

    for step, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) if hasattr(v, "to") else v for k, v in t.items()}
            for t in targets
        ]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        running_loss +- loss_value
        step_count += 1

        print(f"step={step} loss={loss_value:.4f}")

        # keep Day 7 intentionally tiny
        if step + 1 >= cfg["training"]["max_steps_per_epoch"]:
            break

    avg_loss = running_loss / max(step_count, 1)
    print(f"average_loss={avg_loss:.4f}")

    out_dir = Path(cfg["training"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / cfg["training"]["checkpoint_name"]
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_loss": avg_loss,
            "steps": step_count,
        },
        ckpt_path,
    )

    print(f"Saved checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()