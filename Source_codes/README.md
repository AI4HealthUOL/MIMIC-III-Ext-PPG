

# MIMIC-III-Ext-PPG Code Repository

This repository contains the full pipeline used in our paper:  
**“MIMIC-III-Ext-PPG, a PPG-based Benchmark Dataset for Cardiovascular and Respiratory Signal Analysis.”**

The code is organized into **three main parts** that take the reader from **raw data extraction** to **final stratified datasets** ready for analysis.

---

## 🔹 Part 1 – Preprocessing  
- Load waveform metadata and select PPG cases.  
- Extract demographic and clinical information (height, weight, heart rhythm) from CHARTEVENTS.csv.  
- Clean and filter events using conservative time-window rules.  
- Add RESP channel metadata where available.  
- Extract original WFDB segments (≤15 min before events).  
- Merge all per-record extractions into a single dataframe.  

---

## 🔹 Part 2 – Processing  
📖 See the dedicated [Processing README](./Processing/README.md) for full details.  
- Perform **signal quality assessment (SQI)** for PPG, ABP, ECG, and RESP.  
- Segment signals into **30s windows** and **10s sub-windows**.  
- Extract clinically relevant features: SBP, DBP, HR, RR intervals, etc.  
- Apply validation rules and discard poor-quality or corrupted segments.  
- Save all features into `.pkl` files for downstream use.  

---

## 🔹 Part 3 – Postprocessing  
- Merge multiple feature files into one combined dataframe.  
- Add derived features (averages, variability measures, etc.) and apply quality checks.  
- Exclude mislabeled or low-quality signals.  
- Stratify the dataset by age, diagnosis codes, and other metadata.  
- Build stratified folds for training, validation, and evaluation.  

---

## ⚙️ Requirements  
```bash
pip install -r requirements.txt
```
**Python Version:** 3.8+  
Main libraries: `neurokit2`, `numpy`, `pandas`, `scipy`, `matplotlib`, `wfdb`, `tqdm`

---

## 🚀 How to Run  
1. Run **Part 1** to prepare events and extract WFDB segments.  
2. Run **Part 2** to process signals, compute SQIs, and extract features.  
3. Run **Part 3** to clean, stratify, and generate folds for benchmarking.  

---


