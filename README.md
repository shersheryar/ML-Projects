# 🤖 Machine Learning Projects Portfolio

A comprehensive collection of deep learning and machine learning projects demonstrating expertise in NLP, time-series forecasting, and classification tasks.

---

## 📂 Projects Overview

| Project | Domain | Techniques | Status |
|---------|--------|------------|--------|
| [Air Quality Prediction](#1-attention-based-deep-learning-for-hyper-local-air-quality-prediction) | Time-Series Forecasting | Transformer, LSTM, Attention | ✅ Complete |
| [Fake News Detection](#2-fake-news-detection) | NLP / Classification | TF-IDF, Random Forest, Streamlit | ✅ Complete |
| [Next Word Prediction](#3-next-word-prediction-lstm) | NLP / Language Modeling | LSTM, RNN | ✅ Complete |

---

## 1. Attention-Based Deep Learning for Hyper-Local Air Quality Prediction

### 🎯 Objective
Develop a **Hybrid Transformer-LSTM** deep learning model that forecasts hourly PM₂.₅ levels up to **12 hours ahead** using multivariate environmental time-series data.

### 📊 Dataset
- **Source**: Beijing Multi-Site Air Quality Dataset (2013-2017)
- **Stations**: 12 monitoring stations across Beijing
- **Features**: PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, WSPM

### 🏗️ Architecture
```
Input (48h) → Transformer Encoder → Global Pooling → LSTM Decoder → Output (12h forecast)
```

**Key Components:**
- Multi-Head Self-Attention (4 heads)
- Positional Encoding
- Stacked LSTM Decoder (2 layers, 128 units)
- Baseline Stacked LSTM for comparison

### 📈 Features
- Sliding window time-series sequences
- Attention heatmap visualization for interpretability
- Comprehensive evaluation metrics (RMSE, MAE, R²)
- Pollution peak analysis

### 📁 Structure
```
Attention-Based Deep Learning for Hyper-Local Air Quality Prediction/
├── dataset/                    # Beijing Air Quality CSV files (12 stations)
├── model/                      # Saved trained models
└── PM25_Hybrid_Transformer_LSTM_Forecasting.ipynb
```

---

## 2. Fake News Detection

### 🎯 Objective
Build a machine learning classifier to detect fake news articles using NLP techniques and deploy it as an interactive web application.

### 📊 Dataset
- Training and test datasets with labeled news articles
- Text preprocessing and feature extraction

### 🏗️ Approach
- **Text Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Model**: Random Forest Classifier
- **Deployment**: Streamlit web application

### 📈 Features
- Text preprocessing pipeline
- Model serialization (pickle)
- Interactive web interface for real-time predictions
- Comprehensive project documentation

### 📁 Structure
```
fake-news-detection/
├── datasets/
│   ├── train.csv
│   ├── test (1).csv
│   └── evaluation.csv
├── document/                   # Project reports and guidelines
├── Sheryar_Sher_fake-news-detection.ipynb
├── streamlit.py               # Web application
├── random_forest_model.pkl    # Trained model
└── vectorizer.pkl             # TF-IDF vectorizer
```

### 🚀 Run the App
```bash
cd fake-news-detection
streamlit run streamlit.py
```

---

## 3. Next Word Prediction LSTM

### 🎯 Objective
Build an LSTM-based language model that predicts the next word in a sequence, trained on Shakespeare's works.

### 📊 Dataset
- **Source**: Tiny Shakespeare dataset
- **Content**: Complete works of Shakespeare in plain text

### 🏗️ Architecture
- Embedding Layer
- Stacked LSTM Layers
- Dense Output with Softmax

### 📈 Features
- Character/Word-level language modeling
- Text generation capabilities
- Sequence-to-sequence learning

### 📁 Structure
```
next-word-prediction-lstm/
├── lstm_project.ipynb
└── tinyshakespeare.txt
```

---

## 🛠️ Technologies Used

### Deep Learning Frameworks
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)

### Data Science Libraries
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

### Visualization
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)

### Deployment
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

---

## 📋 Requirements

```txt
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
streamlit>=1.0.0
```

### Installation
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn streamlit
```

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ML-Projects
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Navigate to a project**
   ```bash
   cd "Attention-Based Deep Learning for Hyper-Local Air Quality Prediction"
   jupyter notebook
   ```

4. **Run notebooks**
   - Open the `.ipynb` file in Jupyter Notebook/Lab
   - Execute cells sequentially

---

## 📊 Project Highlights

### Skills Demonstrated
- ✅ **Deep Learning**: Transformer, LSTM, RNN architectures
- ✅ **NLP**: Text preprocessing, TF-IDF, Language modeling
- ✅ **Time-Series Analysis**: Sliding windows, Multi-step forecasting
- ✅ **Attention Mechanisms**: Self-attention, Multi-head attention
- ✅ **Model Deployment**: Streamlit web applications
- ✅ **Data Visualization**: Matplotlib, Seaborn, Attention heatmaps

### Best Practices
- 📝 Comprehensive code documentation
- 🔄 Reproducible experiments (random seeds)
- 📈 Proper train/validation/test splits
- 💾 Model serialization and saving
- 📊 Detailed evaluation metrics

---

## 👤 Author

**Sheryar Sher**

---

## 📄 License

This project is for educational purposes.

---

## 🙏 Acknowledgments

- Beijing Multi-Site Air Quality Dataset
- Tiny Shakespeare Dataset
- TensorFlow and Keras teams
- Open-source community

---

*Last Updated: November 2025*

