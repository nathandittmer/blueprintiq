from __future__ import annotations

from blueprintiq.datasets.coco_detection_dataset import CocoDetectionDataset
from blueprintiq.models.detector import build_title_block_detector

def main():
    ds = CocoDetectionDataset(
        root_dir="data/synth_v0",
        coco_json="data/synth_v0/coco_title_block.json",
    )

    image, target = ds[0]

    model = build_title_block_detector(num_classes=2)
    model.train()

    images = [image]
    targets = [target]

    loss_dict = model(images, targets)

    print("Model forward pass successful")
    print("Loss keys:", list(loss_dict.keys()))
    print("Loss dict:", {k: float(v.detach().cpu()) for k, v in loss_dict.items()})


if __name__ == "__main__":
    main()