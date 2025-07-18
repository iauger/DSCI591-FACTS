# Data Acquisition

This module defines a robust pipeline for collecting, cleaning, and storing a diverse set of question-answering (QA) datasets for downstream machine learning workflows. The end goal of the module is to have cleaned data files in local storage that are ready for the pre-processing stage and can be migrated to BigQuery. 

## Pipeline Overview

1. **Data Loading** (`loader.py`)  
   Downloads datasets from either public URLs or the Hugging Face Hub and saves them to `/data/raw/` in their original formats (e.g., JSONL, CSV).

2. **Data Cleaning** (`cleaner.py`)  
   Converts raw files into schema-consistent CSVs with five standardized fields: `id`, `title`, `context`, `question`, and `answers`. Malformed rows are logged separately for inspection.

## Extensibility

The pipeline is modular and designed for easy expansion. New datasets can be added by updating the loader’s configuration and writing a custom cleaner if needed.

