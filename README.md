# BlueprintIQ

## Project overview

BlueprintIQ is an end-to-end machine learning system for construction drawing understanding, focused on detecting title blocks in blueprint images using PyTorch object detection.

## Why this project matters

- Construction and AECO workflows often depend on extracting structured information from drawings.
- Title block detection is a practical first step toward drawing metadata extraction and document understanding.
- This project demonstrates the full ML lifecycle from synthetic data generation to model serving and lightweight monitoring.

## What this project includes

- Synthetic blueprint image generation
- COCO-style labeled detection dataset
- PyTorch Faster R-CNN title block detector
- IoU-based evaluation and saved evaluation reports
- Visualization of predicted vs. ground-truth boxes
- CLI inference for single-image prediction
- Batch inference over folders of images
- FastAPI service for model serving
- Prediction logging and monitoring summaries
- Dockerized API packaging

## System architecture

Synthetic Data Generator  
→ COCO Dataset  
→ PyTorch Dataset Loader  
→ Faster R-CNN Detector  
→ Evaluation + Visualization  
→ CLI / Batch Inference  
→ FastAPI Service  
→ Prediction Logging + Monitoring  
→ Dockerized Deployment

## Tech stack

Python, PyTorch, torchvision, FastAPI, Typer, Docker, MLflow, Pillow, OpenCV

## How to run

### Train the detector

make train-detector

### Evaluate the detector

make eval-detector

### Visualize predictions

make viz-predictions

### Run single-image inference

python -m blueprintiq.inference.predict --input data/synth_v0/images/sheet_00000.png

### Run batch inference

make batch-predict

### Start the API

make serve

### Build the Docker image

docker build -t blueprintiq-api .

## Example results

On a small synthetic evaluation sample, the detector achieved:

- 100% prediction rate
- 100% match rate at IoU ≥ 0.5
- Average best IoU of 0.7343

These results are based on synthetic data and are intended to demonstrate end-to-end ML engineering workflow rather than final production accuracy.