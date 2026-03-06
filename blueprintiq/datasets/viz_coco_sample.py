from __future__ import annotations

from pathlib import Path

from PIL import ImageDraw, ImageFont

from blueprintiq.datasets.coco_detection_dataset import CocoDetectionDataset


def _font(size: int = 28):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def main():
    ds = CocoDetectionDataset(
        root_dir="data/synth_v0",
        coco_json="data/synth_v0/coco_title_block.json",
    )

    image, target = ds[0]
    draw = ImageDraw.Draw(image)

    for box in target["boxes"]:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=6)
        draw.text((x1 + 10, y1 + 10), "title_block", fill=(255, 0, 0), font=_font())

    out_dir = Path("runs/sanity")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample_box.png"
    image.save(out_path)

    print(f"✅ Wrote: {out_path}")


if __name__ == "__main__":
    main()