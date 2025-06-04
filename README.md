## Multimodal Lie Detection System

This project is an assignment for the Artificial Intelligence major at Taylor's University.

### Project Description

This project aims to detect whether a person is lying by analyzing multiple modalities including audio, video, and text. We use deep learning and feature fusion strategies to build a comprehensive judgment model.

---

### Key Features

| English |
|---------|
| Multimodal Data Loader |
| Audio Feature Extraction |
| Text Feature Extraction |
| Visual Feature Extraction |
| Model Training with Fusion Strategies |
| Feature Importance Analysis |
| Visualization of Results |

---

### Tech Stack

- **Python 3.x**
- **PyTorch** – Deep Learning Framework
- **NumPy/Pandas** – Data Processing
- **Matplotlib/Seaborn** – Data Visualization
- **Git + GitHub** – Version Control and Collaborative Development

---

### 📁 Key Files

| Filename | Description |
|---------|-------------|
| [DataConfig.py](file:///Users/guotao/PycharmProjects/TAIAssignment2/DataConfig.py) | Data path configuration class |
| [Main.py](file:///Users/guotao/PycharmProjects/TAIAssignment2/Main.py) | Main program entry point |
| [ModelTrainer.py](file:///Users/guotao/PycharmProjects/TAIAssignment2/ModelTrainer.py) | Model training and fusion logic |
| [MultimodalDataLoader.py](file:///Users/guotao/PycharmProjects/TAIAssignment2/MultimodalDataLoader.py) | Multimodal data loading and processing |
| [TextFeatureExtractor.py](file:///Users/guotao/PycharmProjects/TAIAssignment2/TextFeatureExtractor.py) | Text feature extraction module |
| [VisualFeatureExtractor.py](file:///Users/guotao/PycharmProjects/TAIAssignment2/VisualFeatureExtractor.py) | Visual feature extraction module |
| [AudioFeatureExtractor.py](file:///Users/guotao/PycharmProjects/TAIAssignment2/AudioFeatureExtractor.py) | Audio feature extraction module |

---

### 📦 Dataset Structure

```
dataset/
├── Clips/          # Video clips
│   ├── false/
│   └── true/
├── Transcription/  # Text transcripts
│   ├── false/
│   └── true/
├── audio/          # Audio files
│   ├── false/
│   └── true/
└── Annotation/     # Annotation files
    └── annotation.csv
```


---

### How to Use

1. **Install dependencies:**
   ```bash
   pip install torch numpy pandas matplotlib seaborn
   ```


2. **Place your dataset in the `dataset/` directory.**

3. **Run the main program:**
   ```bash
   python Main.py
   ```


4. The program will automatically load multimodal data, train fusion models, and generate feature analysis charts.

---

### Team Members

| Name |
|------|
| Xiao Changhe |
| Guo Tao |
| Kan YiMing |
| Zhang ZhiAng |
| Zheng YaXin |

---

### License

MIT License - Commercial use allowed, please retain original author attribution.

---

### Contact

For questions or collaboration opportunities, please contact:  
guotao2beijing@gmail.com