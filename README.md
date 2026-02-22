# Sentiment Analysis on Emotions Dataset

## 📌 Project Overview

This project performs **Sentiment Analysis** on an emotions dataset obtained from Kaggle. 
The objective is to build a machine learning model capable of accurately classifying text into predefined emotion categories (e.g., joy, anger, sadness, fear, love, surprise).

Dataset
https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp/data

The project covers the complete end-to-end data science workflow:

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Model development
* Model evaluation and interpretation
* Visualization of results


---

## 📊 Dataset Description

The dataset consists of text samples labeled with emotion categories. Each record contains:

* **Text** – A sentence or short paragraph
* **Emotion Label** – The emotion expressed in the text

### Target Classes

Examples of emotions:

* Joy
* Anger
* Sadness
* Fear
* Love
* Surprise

---

# 🔎 Data Cleaning and Preprocessing

The dataset was cleaned and prepared for modeling using the following steps:

### ✅ Missing Values

* Checked for null values.
* No significant missing data detected.
* If present, missing values were removed or imputed appropriately.

### ✅ Duplicate Records

* Duplicate entries were identified and removed to avoid bias.

### ✅ Text Cleaning

* Converted text to lowercase
* Removed punctuation
* Removed special characters
* Removed stopwords
* Tokenization applied
* Lemmatization applied (if included in notebook)

### ✅ Feature Engineering

* Transformed raw text into numerical features using:

  * **TF-IDF Vectorization**
* Created clean feature matrices for model training.

---

# 📈 Exploratory Data Analysis (EDA)

EDA was performed to understand:

* Distribution of emotion classes
* Text length distribution
* Class imbalance
* Word frequency patterns

### Visualizations Included:

* Class distribution bar plots
* Word count distribution histograms
* Most frequent words per emotion
* Correlation heatmaps (if applicable)

### Key Insights:

* Some emotions are more frequent than others.
* Text lengths vary significantly.
* Class imbalance was evaluated before modeling.

All plots include:

* Descriptive titles
* Labeled axes
* Appropriate scaling
* Clear legends
* Proper subplot usage when necessary

---

# 🤖 Modeling

## Model Selection

* Naive Bayes baseline model was implemented

---

## 📏 Evaluation Metrics

The following evaluation metrics were used:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**

### Rationale for Metric Selection

* Accuracy provides overall correctness.
* F1-score balances precision and recall, especially important if class imbalance exists.
* Confusion matrix helps understand per-class performance.

---

## 📊 Model Performance

The Multinomial Naive Bayes model, using TF-IDF vectorization, achieved an overall accuracy of 62%. 
While it showed high precision for specific labels like anger (0.96) and love (1.00), it struggled with recall for minority classes and completely failed to predict the surprise category.

### Interpretation

* The model performs well across majority classes.
* Some confusion exists between closely related emotions.
* Performance could improve using:

  * Hyperparameter tuning
  * Deep learning models (LSTM/BERT)
  * Class balancing techniques

---

# 📌 Key Findings

* Emotion classification can be effectively performed using TF-IDF features.
* Logistic Regression provides a strong and interpretable baseline.
* Class imbalance may impact minority emotion detection.
* Text preprocessing significantly improves performance.

---

# 🚀 Future Improvements

* Implement deep learning models (LSTM, GRU)
* Use transformer-based models (BERT)
* Apply hyperparameter tuning (GridSearchCV)
* Handle class imbalance using SMOTE

---
