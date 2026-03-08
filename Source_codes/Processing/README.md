
# 🧠 Physiological Signal Quality Assessment & Feature Extraction

The following repository is a modular and extensible Python pipeline for processing and analyzing physiological signals, including ABP, ECG, PPG, and RESP. This toolkit is part of our **MIMIC-III-EXt-PPG** project. It incorporates advanced preprocessing, signal quality assessment, and feature extraction — with NeuroKit2 integrated for peak detection and signal cleaning where applicable.

---

## Signal Processing Workflow

The pipeline processes each 30/10s-second segment of physiological signals, validates and scores their quality, and extracts clinically relevant metrics.

---

## 📁 Project Modules

| Module | Purpose |
|--------|---------|
| `main.py` | Entry point for batch processing 30-second signal windows. Handles data loading, segmentation, validation, SQI computation, and feature extraction. |
| `utils.py` | General-purpose utilities for NaN handling, signal validation, clipping, padding, and WFDB signal loading. |
| `abp_utils.py` | ABP-specific utilities for beat detection, SBP/DBP identification, physiological plausibility filtering, and SQI evaluation. |
| `resp_utils.py` | RESP signal preprocessing and SQI based on template similarity and breathing cycle consistency. |
| `sqi_utils.py` | Central dispatch for signal quality assessment (ABP, ECG, PPG, RESP), using template correlation or physiology-based rules. |
| `plotting_utils.py` | Tools to visualize ABP, ECG, and RESP signals with annotations for peaks, troughs, and fiducial points. |

---

## 🧪 Analysis Details

### 🔹 30-Second Window
Used for:
- 📈 RR interval analysis (median & IQR) from RESP
- ✅ SQI computation for RESP

### 🔹 10-Second Sub-Windows
Each 30s window is split into three 10s segments to extract:
- 🩺 SBP / DBP (median & IQR)
- ❤️ Heart Rate (HR) from ECG
- 🔦 PPG SQI, ABP SQI, and ECG SQI

All 10s-based metrics are stored as time-resolved vectors:
```python
vector_10s_sbp = [SBP_10s_1, SBP_10s_2, SBP_10s_3]
```

---

## 🧪 SQI Code Map

| SQI Code | Stage / Source         | Signal(s)       | Meaning                                                                                   | Trigger Condition / Notes                                                                 |
|----------|------------------------|------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| `1`      | SQI Calculation         | All              | ✅ Good-quality signal                                                                     | All checks passed (HR, RR, morphology, plausibility, etc.)                               |
| `0`      | SQI Calculation         | All              | ⚠️ Analyzable, but fails quality thresholds                                               | Detected peaks but failed R², HR, or RR consistency                                      |
| `-2`     | Validation              | All              | ❌ Flatline or repeated extreme values                                                    | Flatline (>1s) or repeated min/max values detected                                       |
| `-3`     | Validation              | All              | ❌ Signal contains NaNs or is empty                                                       | Corrupted, incomplete, or missing signal                                                 |
| `-4`     | Validation              | All              | ❌ Signal too short (<10s)                                                                | Not enough samples to analyze                                                            |
| `-10`    | SQI Calculation (RESP)  | RESP             | ❌ RESP signal too short (<30s)                                                           | Required segment length not met                                                          |
| `-11`    | SQI Calculation (RESP)  | RESP             | ❌ No valid peaks or troughs detected                                                     | No respiration cycles found                                                              |
| `-12`    | SQI Calculation (RESP)  | RESP             | ❌ Not enough valid breath cycles                                                         | Too few peaks for cycle duration analysis                                                |
| `-13`    | SQI Calculation (RESP)  | RESP             | ❌ Not enough valid segments for template matching                                        | Fewer than 2 breath segments for morphology check                                        |
| `-14`    | SQI Calculation (ECG/PPG)| ECG, PPG         | ❌ No RR intervals (not enough beats)                                                     | `rr_intervals` empty — invalid beat detection                                            |
| `-15`    | SQI Calculation (ECG/PPG)| ECG, PPG         | ❌ RR sample window too small                                                             | Cannot extract morphology templates                                                      |
| `-16`    | SQI Calculation (ECG/PPG)| ECG, PPG         | ❌ Not enough valid segments for template matching                                                            | Less than 2 normalized beats                                                             |
| `-17`    | SQI Calculation (ECG)    | ECG              | ⚠️ Invalid or insufficient R-peaks                                                       | <2 R-peaks                                                             |
| `-18`    | SQI Calculation (PPG)    | PPG              | ⚠️ Invalid or insufficient PPG peaks                                                     | <2 peaks                                                             |
| `-19`    | SQI Calculation (ABP)    | ABP              | ❌ Beat onset detection failed                                                           | ABP onset detection returned <2 beats                                                    |
| `-20`    | Dispatcher              | Any              | ❌ Unknown signal type                                                                   | Signal type not recognized in `listen_sqi2()`                                            |
| `-21`    | Exception handler       | ABP, RESP, All   | ❌ Uncaught exception during SQI computation                                             | Raised in `listen_resp_sqi()`, `listen_abp_sqi()`, or `listen_sqi2()`                    |

---

## ⚙️ Requirements

```bash
pip install -r requirements.txt
```

**Python Version:** 3.8+  
Main Libraries:
- `neurokit2`
- `numpy`, `pandas`, `scipy`
- `matplotlib`
- `wfdb`, `tqdm`

---

## 🚀 Quick Start

1. Place the address of WFDB files in `WFDB_folder/`, and the output folder of`.npy` PPG signals in `PPG_npy_folder/`.
2. Set `meta_data_path`, `start_index`, and `end_index` in `main.py`.
3. Run:
```bash
python main.py
```
4. Outputs are saved as `.pkl` files (e.g., `df_0_100.pkl`) with all extracted features.

---

## 📚 References

- Charlton, P. H., et al. (2021). *An impedance pneumography signal quality index...*
- Orphanidou, C., et al. (2014). *Signal-Quality Indices for ECG and PPG...*
- Wang, W., et al., (2023). *PulseDB: A large, cleaned dataset based on MIMIC-III and VitalDB*
- [📘 NeuroKit2 Documentation](https://neurokit2.readthedocs.io)



