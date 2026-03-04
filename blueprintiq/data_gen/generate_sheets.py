from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def _load_font(size: int = 24) -> ImageFont.ImageFont:
    # Safe default: PIL built-in font if truetype unavailable
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _rand_project_name(rng: random.Random) -> str:
    options = ["ACME HQ RENOVATION", "RIVER BRIDGE RETROFIT", "NORTH CAMPUS EXPANSION", "SOLAR ARRAY SITE"]
    return rng.choice(options)


def _rand_sheet_number(rng: random.Random) -> str:
    discipline = rng.choice(["A", "S", "M", "E", "C"])
    num = rng.randint(100, 399)
    return f"{discipline}{num}"


def generate_one(
    image_size: tuple[int, int],
    seed: int,
) -> tuple[Image.Image, dict[str, Any]]:
    rng = random.Random(seed)
    w, h = image_size

    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Border
    margin = 40
    draw.rectangle([margin, margin, w - margin, h - margin], outline=(0, 0, 0), width=3)

    # Title block (bottom-right)
    tb_w, tb_h = int(w * 0.33), int(h * 0.18)
    tb_x2, tb_y2 = w - margin, h - margin
    tb_x1, tb_y1 = tb_x2 - tb_w, tb_y2 - tb_h
    draw.rectangle([tb_x1, tb_y1, tb_x2, tb_y2], outline=(0, 0, 0), width=3)

    # Title block internal lines
    draw.line([tb_x1, tb_y1 + tb_h * 0.35, tb_x2, tb_y1 + tb_h * 0.35], fill=(0, 0, 0), width=2)
    draw.line([tb_x1, tb_y1 + tb_h * 0.7, tb_x2, tb_y1 + tb_h * 0.7], fill=(0, 0, 0), width=2)

    font = _load_font(28)

    project = _rand_project_name(rng)
    sheet = _rand_sheet_number(rng)
    rev = rng.choice(["0", "1", "2", "A", "B"])
    date = f"2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"

    draw.text((tb_x1 + 20, tb_y1 + 15), f"PROJECT: {project}", fill=(0, 0, 0), font=font)
    draw.text((tb_x1 + 20, int(tb_y1 + tb_h * 0.4)), f"SHEET: {sheet}", fill=(0, 0, 0), font=font)
    draw.text((tb_x1 + 20, int(tb_y1 + tb_h * 0.75)), f"REV: {rev}   DATE: {date}", fill=(0, 0, 0), font=font)

    # COCO bbox uses [x, y, width, height]
    bbox = [int(tb_x1), int(tb_y1), int(tb_w), int(tb_h)]
    record = {
        "project_name": project,
        "sheet_number": sheet,
        "revision": rev,
        "date": date,
        "title_block_bbox": bbox,
    }
    return img, record


def write_coco(
    images_meta: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    out_path: Path,
) -> None:
    coco = {
        "info": {"description": "BlueprintIQ synthetic title block dataset", "version": "0.1"},
        "licenses": [],
        "categories": [{"id": 1, "name": "title_block", "supercategory": "layout"}],
        "images": images_meta,
        "annotations": annotations,
    }
    out_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")


def generate_dataset(
    out_dir: Path,
    n: int,
    image_size: tuple[int, int],
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    images_meta: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    rng = random.Random(seed)

    ann_id = 1
    for i in tqdm(range(n), desc="Generating sheets"):
        ex_seed = rng.randint(0, 10_000_000)
        img, rec = generate_one(image_size=image_size, seed=ex_seed)

        file_name = f"sheet_{i:05d}.png"
        img_path = img_dir / file_name
        img.save(img_path)

        image_id = i + 1
        images_meta.append(
            {
                "id": image_id,
                "file_name": f"images/{file_name}",
                "width": image_size[0],
                "height": image_size[1],
            }
        )

        bbox = rec["title_block_bbox"]
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            }
        )
        ann_id += 1

    write_coco(images_meta, annotations, out_dir / "coco_title_block.json")