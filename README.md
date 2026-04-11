# 🛍️ Vietnamese E-commerce Sentiment Analysis

> End-to-end Machine Learning pipeline for Vietnamese product review classification  
> Built with NLP, TF-IDF, and deployed via Gradio on Hugging Face Spaces

---

## 📌 Overview

This project implements a complete **sentiment analysis system** for Vietnamese e-commerce reviews.

The goal is to automatically classify product reviews into **positive** or **negative** sentiment, helping businesses better understand customer feedback and improve decision-making.

The system follows a full **end-to-end Machine Learning workflow**, from raw data processing to deployment as an interactive web application.

---

## 🎯 Objectives

- Build a robust pipeline for Vietnamese text processing
- Apply Machine Learning techniques for sentiment classification
- Deploy a real-time inference system using Gradio
- Simulate a production-ready ML workflow

---

## 🌟 Key Features

- 🇻🇳 Vietnamese text preprocessing (cleaning, tokenization, stopword removal)
- 🧠 TF-IDF feature extraction with n-grams
- 🤖 Machine Learning models (SVM)
- ⚡ Real-time prediction via Gradio interface
- 🚀 Deployment on Hugging Face Spaces
- 🧩 Modular pipeline design (easy to extend & maintain)

---

## 🔄 Pipeline

The system processes input text through the following pipeline:

Raw Review
- Text Cleaning
- Tokenization (PyVi)
- Stopword Removal
- TF-IDF Vectorization
- Model Prediction
- Sentiment Output (Positive / Negative)


---

## 🧠 Methodology

### 1. Data Preprocessing
- Convert text to lowercase
- Remove URLs, mentions (@), hashtags (#)
- Remove numbers and special characters
- Normalize whitespace

### 2. Tokenization
- Use `pyvi` for Vietnamese word segmentation

### 3. Stopword Removal
- Filter out common Vietnamese stopwords to reduce noise

### 4. Feature Engineering
- Apply **TF-IDF (Term Frequency - Inverse Document Frequency)**
- Support n-grams to capture contextual information

### 5. Modeling
- Support Vector Machine (SVM)
- Lightweight and efficient for real-time inference

---

## 🚀 Quick Start

### 1. Clone repository

```bash
git clone https://github.com/<your-username>/Shopee_Sentiment_Analysis.git
cd Shopee_Sentiment_Analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run application

```bash
python app.py
```

🌐 Demo

👉 Try the live demo on Hugging Face Spaces:
<https://huggingface.co/spaces/cngchis/Shopee_Sentiment_Analysis>

---

📊 Results
- Efficient sentiment classification for Vietnamese reviews
- Fast inference using lightweight ML models
- Scalable pipeline design for real-world applications

---

🚧 Future Improvements
- 🔥 Upgrade to Deep Learning models (LSTM, BERT, PhoBERT)
- 📊 Add multi-class sentiment (positive / neutral / negative)
- ⚡ Optimize preprocessing & feature engineering
- 🌐 Deploy as REST API using FastAPI
- 📈 Add monitoring & logging for production

---

🧠 Lessons Learned
Importance of text preprocessing in NLP tasks
Trade-offs between traditional ML and deep learning
Designing modular pipelines for scalability
Deploying ML models as real-time applications

---

📄 License
MIT

---

🙌 Acknowledgements
- scikit-learn for Machine Learning models
- pyvi for Vietnamese NLP processing
- Gradio for building interactive UI
- Hugging Face Spaces for deployment