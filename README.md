# FACTS: Factuality and Consistency Scoring for LLM Outputs

This repository contains the codebase for the **FACTS metric**, a capstone project for Drexel’s MSDS program. The project explores signals of hallucination in large language model (LLM) outputs and develops the **Global Hallucination Index (GHI)**, a composite evaluation metric built from:  
- **Relative Agreement (RA)**  
- **Contradiction Consistency (CC)**  
- **Low-Hallucination Confidence (LHC)**  

We benchmark GHI against traditional metrics such as BLEU and ROUGE, with the goal of creating a more reliable tool for hallucination detection.

---

## Repository Structure (in progress)

- `data/` — Raw, cleaned, processed, and features datasets.  
- `data_acquisition` — Data acquisition module for accessing and cleaning raw data. 
- `data_pipeline` — Data pipeline module for managing cloud storage. 
- `features/` — Feature extraction modules (RA, CC, LHC, readability, entities).  
- `evaluation/` — Evaluation scripts and metrics (BLEU, ROUGE, METEOR, GHI).  
- `notebooks/` — Jupyter notebooks for exploratory analysis and experiments.  
- `scripts/` — End-to-end utilities for scoring datasets (`features_script.py`, `metrics_script.py`).  


## Data Usage
This repository includes the TruthfulQA dataset (Lin, Hilton, Evans, 2021) from the official repository (sylinrl/TruthfulQA). TruthfulQA is provided under the Apache License, Version 2.0. See the upstream LICENSE for terms. 

---

