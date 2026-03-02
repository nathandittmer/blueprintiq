# BlueprintIQ

## Overview

BlueprintIQ is a production-oriented machine learning system for extracting structured metadata from construction drawings.

The system detects title blocks in plan sheets and extracts fields such as:

- Sheet Number
- Project Name
- Revision
- Date
- Discipline

This project simulates a real AECO (Architecture, Engineering, Construction, Operations) workflow aligned with industrial ML engineering standards.

## System Architecture

Synthetic Blueprint Generator  
↓  
Detection Model (PyTorch)  
↓  
Title Block Crop  
↓  
OCR + Postprocessing  
↓  
Structured JSON Output  
↓  
FastAPI Service  
↓  
Monitoring + Drift Metrics

## Design Principles

- Reproducible experiments
- Config-driven pipelines
- Modular ML architecture
- Production-style inference service
- Monitoring-first mindset

## Status

**Day 1:** Repository scaffolding complete.