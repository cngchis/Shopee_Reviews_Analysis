# Shopee Reviews Analysis

## Introduction

This project performs analysis and classification of Shopee reviews using Machine Learning and Natural Language Processing (NLP) techniques for the Vietnamese language.

## Project Structure

- `Shopee_Reviews_Analysis.ipynb`: Jupyter Notebook containing the entire source code for analysis, data preprocessing, and model training.
- `Data/`: Directory containing training and validation data.
  - `train.csv`: Training dataset.
  - `val.csv`: Validation dataset.
- `Stopword/`: Directory containing the list of stopwords.
  - `vietnamese-stopwords.txt`: List of Vietnamese stopwords.

## System Requirements

To run this project, you need to install the following Python libraries:

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- pyvi (Vietnamese text processing library)
- scikit-learn

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn plotly pyvi scikit-learn
```

## Methodology

### 1. Data Preprocessing

- **Text Cleaning:**
  - Convert to lowercase.
  - Remove URLs, mentions (@), hashtags (#).
  - Remove numbers and special characters.
  - Standardize whitespace.
- **Tokenization:** Use the `pyvi` library for Vietnamese word segmentation.
- **Stopword Removal:** Remove words with little meaning based on the list in `vietnamese-stopwords.txt`.

### 2. Feature Extraction

- Use **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical feature vectors.

### 3. Modeling

- Use the **Support Vector Machine (SVM)** algorithm for review classification.

## Usage

1.  Ensure you have installed all necessary libraries.
2.  Open the `Shopee_Reviews_Analysis.ipynb` file using Jupyter Notebook or Google Colab.
3.  Run the cells in the notebook sequentially to perform the analysis and model training process.

## Results

The project analyzes the distribution of labels in the dataset and builds a model to predict the sentiment/classification of new reviews.
