from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from blueprintiq.datasets.coco_detection_dataset import CocoDetectionDataset
from blueprintiq.models.detector import build_title_block_detector
from blueprintiq.training.utils import detection_collate_fn

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = CocoDetectionDataset(
        root_dir="data/synth_v0",
        coco_json="data/synth_v0/coco_title_block.json",
    )

    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=True,
        collate_fn=detection_collate_fn,
    )

    model = build_title_block_detector(num_classes=2)
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    running_loss = 0.0

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

        running_loss +- float(loss.detach().cpu())

        print(f"step={step} loss={running_loss / (step + 1):.4f}")

        # keep Day 7 intentionally tiny
        if step >= 2:
            break

    out_dir = Path("runs/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "detector_day7.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_path,
    )

    print(f"Saved checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()