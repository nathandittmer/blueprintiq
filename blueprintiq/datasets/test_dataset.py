from blueprintiq.datasets.coco_detection_dataset import CocoDetectionDataset

ds = CocoDetectionDataset(
    root_dir="data/synth_v0",
    coco_json="data/synth_v0/coco_title_block.json",
)

img, target = ds[0]

print(img.shape)
print(target)