from __future__ import annotations

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_title_block_detector(num_classes: int = 2):
    """
    number_classes includes background.
    For one object class ('title_block'), use num_classes=2.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model