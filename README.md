## 多模态谎言检测系统（Multimodal Lie Detection System）

本项目是 Taylor’s University 计算机人工智能专业课程作业。  
This project is an assignment for the Artificial Intelligence major at Taylor's University.

### 项目描述

该项目旨在通过分析音频、视频和文本等多种模态数据，识别一个人是否在说谎。我们结合了深度学习与特征融合策略来构建一个综合判断模型。

This project aims to detect whether a person is lying by analyzing multiple modalities including audio, video, and text. We use deep learning and feature fusion strategies to build a comprehensive judgment model.

---

###  功能特性

| 中文 | English |
|------|---------|
| 多模态数据加载器 | Multimodal Data Loader |
| 音频特征提取 | Audio Feature Extraction |
| 文本特征提取 | Text Feature Extraction |
| 视觉特征提取 | Visual Feature Extraction |
| 模型训练与融合策略 | Model Training with Fusion Strategies |
| 特征重要性分析 | Feature Importance Analysis |
| 可视化图表输出 | Visualization of Results |

---

###  技术栈 / Tech Stack

- **Python 3.x**
- **PyTorch** – 深度学习框架
- **NumPy/Pandas** – 数据处理
- **Matplotlib/Seaborn** – 数据可视化
- **Git + GitHub** – 版本控制与协作开发

---

### 📁 主要文件说明 / Key Files

| 文件名 | 描述 |
|-------|------|
| [DataConfig.py](file:///Users/guotao/PycharmProjects/TAIAssignment2/DataConfig.py) | 数据路径配置类 |
| [Main.py](file:///Users/guotao/PycharmProjects/TAIAssignment2/Main.py) | 主程序入口 |
| [ModelTrainer.py](file:///Users/guotao/PycharmProjects/TAIAssignment2/ModelTrainer.py) | 模型训练与融合逻辑 |
| [MultimodalDataLoader.py](file:///Users/guotao/PycharmProjects/TAIAssignment2/MultimodalDataLoader.py) | 多模态数据加载与处理 |
| [TextFeatureExtractor.py](file:///Users/guotao/PycharmProjects/TAIAssignment2/TextFeatureExtractor.py) | 文本特征提取模块 |
| [VisualFeatureExtractor.py](file:///Users/guotao/PycharmProjects/TAIAssignment2/VisualFeatureExtractor.py) | 视觉特征提取模块 |
| [AudioFeatureExtractor.py](file:///Users/guotao/PycharmProjects/TAIAssignment2/AudioFeatureExtractor.py) | 音频特征提取模块 |

---

### 📦 数据集结构 / Dataset Structure

```
dataset/
├── Clips/          # 视频片段 Video clips
│   ├── false/
│   └── true/
├── Transcription/  # 文本转录 Text transcripts
│   ├── false/
│   └── true/
├── audio/          # 音频文件 Audio files
│   ├── false/
│   └── true/
└── Annotation/     # 标注文件 Annotations
    └── annotation.csv
```


---

###  使用方法 / How to Use

1. 安装依赖项：
   ```bash
   pip install torch numpy pandas matplotlib seaborn
   ```


2. 将数据集放入 `dataset/` 目录下。

3. 运行主程序：
   ```bash
   python Main.py
   ```


4. 程序会自动加载多模态数据并训练融合模型，最后生成特征分析图。

---

###  小组成员 / Team Members

| 姓名 | 
|------|
| Xiao Changhe | 
| Guo Tao | 
| Kan YiMing |
| Zhang ZhiAng |
| Zheng YaXin | 

---

### 开源协议 / License

MIT License - 允许商业用途，请保留原始作者声明。

MIT License - Commercial use allowed, please retain original author attribution.

---

###  联系方式 / Contact
如有问题或合作意向，请联系：guotail@outlook.com

For questions or collaboration opportunities, please contact: guotao2beijing@gmail.com
