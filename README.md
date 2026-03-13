
# 🫀 MIMIC-III-Ext-PPG (Official Project Page)


**MIMIC-III-Ext-PPG** is a large-scale dataset for **cardiorespiratory signal analysis** derived from the MIMIC-III Waveform Database. This repository serves as the **official hub for updates, documentation, and code resources** related to the dataset.

---

## 🔗 Dataset Access (PhysioNet)

📦 **MIMIC-III-Ext-PPG v1.0.0**

https://physionet.org/content/mimic-iii-ext-ppg/1.0.0/

---


# 📊 Overview

**MIMIC-III-Ext-PPG** is a large-scale physiological signal dataset designed for **machine learning and biomedical signal processing research**.

The dataset is derived from the **MIMIC-III Waveform Database Matched Subset** and provides curated physiological signals.

🧬 **Signals included**

- 🫀 **PPG (PLETH)**
- 💓 **ECG**
- 🩸 **Arterial Blood Pressure (ABP)**
- 🌬 **Respiratory signals (RESP)**

📑 Each segment is associated with:

- 🫀 heart rhythm annotations  
- 👤 demographic information  
- 🏥 clinical metadata  
- 📈 physiological measurements  

---

# 🔬 Supported Research Tasks

The dataset enables research in:

- 🧠 heart rhythm classification
- ❤️ atrial fibrillation detection
- 🩸 blood pressure estimation
- 🌬 respiratory rate estimation
- 💓 heart rate estimation
- 📉 signal quality assessment
- 🤖 machine learning for biomedical signals

---

# 🗂 Dataset Versions

## 🟢 v1.0.0 — Current Public Release

Available on **PhysioNet**.

📊 Approximate dataset statistics:

| Property | Value |
|------|------|
| 👥 Subjects | ~6,131 |
| 📦 Segments (30 s) | ~4.9 million |
| 📡 Signals | PPG, ECG, ABP, RESP |
| 📑 Metadata | demographics, rhythm labels, physiological measurements |

---

## 🟡 v1.1.0 — Upcoming Release

Version **v1.1.0** has been uploaded to PhysioNet and is currently **under review**.


MIMIC-III-Ext-PPG  
│  
├── 👤 Subject (subject_id)  
│  
├── 📁 Record (record_id)  
│   │  
│   ├── 🫀 Event (event_id)  
│   │   │  
│   │   ├── 📦 Segment (30 seconds)  


---

## 💻 MIMIC-III-Ext-PPG Code Repository

For the **source code used in the dataset processing, benchmarking, and analysis**, please visit the dedicated code repository:

➡️ **MIMIC-III-Ext-PPG Code Repository**

Source code and pipelines are available in the **`Source_codes`** directory:

---


# 📚 Citation

If you use this dataset, please cite:

Moulaeifard M, Charlton P H, Strodthoff N. MIMIC-III-Ext-PPG: A PPG Benchmark Dataset for Cardiorespiratory Analysis (version 1.0.0). PhysioNet. 2026. RRID:SCR_007345. Available from: https://doi.org/10.13026/nmwb-6h34

or BibTeX:

@article{PhysioNet-mimic-iii-ext-ppg-1.0.0,
  author = {Moulaeifard, Mohammad and Charlton, Peter H and Strodthoff, Nils},
  title = {{MIMIC-III-Ext-PPG: A  PPG Benchmark Dataset for Cardiorespiratory Analysis}},
  journal = {{PhysioNet}},
  year = {2026},
  month = feb,
  note = {Version 1.0.0},
  doi = {10.13026/nmwb-6h34},
  url = {https://doi.org/10.13026/nmwb-6h34}
}

---

# 💬 Contact

For questions, suggestions, or collaboration inquiries:

👉 Please send an email to mohammad.moulaeifard@uol.de


## 👥 Contributors

- 💻 [**Mohammad Moulaeifard**](https://github.com/mMoulaeifard)  
- 💻 [**Peter Charlton**](https://github.com/peterhcharlton)  
- 💻 [**Nils Strodthoff**](https://github.com/nstrodt)


## 📜 License

### Code
The source code in this repository is released under the **MIT License**.

