# Sentiment Analysis on Emotions Dataset  
**Capstone Project – End-to-End NLP Pipeline**

---

## Project Overview

This project builds a **multi-class emotion classification system** using Natural Language Processing (NLP) techniques on a Kaggle emotions dataset.

The objective is to automatically classify text into human emotions such as:

- Joy  
- Anger  
- Sadness  
- Fear  
- Love  
- Surprise  

This project demonstrates a **complete data science workflow**, including:

- Data cleaning and preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Handling class imbalance  
- Model development (ML + Deep Learning)  
- Model evaluation and interpretation  
- Actionable insights and recommendations  

---

## Business Understanding

Understanding human emotions from text has significant real-world applications:

- Customer feedback analysis  
- Social media sentiment tracking  
- Chatbots and virtual assistants  
- Mental health monitoring  

Organizations can leverage emotion classification to **improve customer experience, detect dissatisfaction early, and make data-driven decisions at scale**.

---

## Dataset Description

- Source: Kaggle Emotions Dataset  
- Type: Multi-class text classification
- Ref: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp/data

### Features:
- **Text**: Input sentence  
- **Emotion Label**: Target variable  

---

# Data Cleaning and Preprocessing

The dataset was cleaned to ensure high-quality input for modeling.

### Steps Performed:

- Converted text to lowercase  
- Removed punctuation and special characters  
- Removed duplicate records  
- Checked and handled missing values  
- Standardized text formatting  

---

# Feature Engineering

To improve model performance, additional features were created:

### Engineered Features:
- Text length  
- Word count  
- Average word length  

### Text Vectorization:
- Applied **TF-IDF Vectorization** to convert text into numerical features  

### Final Feature Set:
- Combined **TF-IDF features + engineered numerical features**  

---

# Exploratory Data Analysis (EDA)

EDA was conducted to understand patterns and distributions in the dataset.

### Key Analyses:
- Distribution of emotion classes (categorical)  
- Text length and word count (continuous)  
- Class imbalance detection  

### Visualizations:

- Bar plots for categorical variables (emotion distribution)  
- Histograms for continuous variables (text length, word count)  
- Subplots used for comparative analysis  

---

# Handling Class Imbalance

The dataset exhibited imbalance across emotion classes.

### Technique Used:
- **RandomOverSampler**

### Impact:
- Improved model performance on minority classes  
- Reduced bias toward majority classes  

---

# Modeling

Multiple models were implemented and compared.

---

## Baseline Model: Multinomial Naive Bayes

- Trained using TF-IDF features  
- Fast and effective for text classification  

---

## Advanced Model: Random Forest

- Used combined feature set (TF-IDF + engineered features)  

### Hyperparameter Tuning:
- Performed using **GridSearchCV**

### Parameters Tuned:
- Number of estimators  
- Max depth  
- Minimum samples split  
- Minimum samples leaf  

---

## Deep Learning Model: Bidirectional LSTM

### Architecture:
- Embedding layer  
- Bidirectional LSTM  
- Dense layers with dropout  
- Softmax output layer  

### Training:
- Loss: Sparse categorical crossentropy  
- Optimizer: Adam  
- Epochs: 5  
- Validation split: 10%  

Captures contextual relationships in text  

---

# Cross-Validation

- Applied during model tuning using **GridSearchCV**  
- Ensures model generalization and robustness  
- Prevents overfitting  

---

# Evaluation Metrics

The following metrics were used:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

### Rationale:

- **Accuracy**: Overall correctness  
- **Precision/Recall**: Performance on individual classes  
- **F1-score**: Balance between precision and recall  
- **Confusion Matrix**: Detailed class-level insights  

---

## Model Performance Summary

| Model | Accuracy | Weighted F1 | Key Insight |
|------|---------|------------|------------|
| Naive Bayes (Baseline) | 62% | 0.53 | Poor minority class performance |
| Naive Bayes + Oversampling | 81% | 0.82 | Major improvement in balance |
| Random Forest (Tuned) | 85% | 0.85 | Best overall ML model |
| LSTM (Deep Learning) | 89% | 0.89 | Best contextual understanding |

---

## Detailed Model Results

### Naive Bayes (Baseline)
- Accuracy: 62%
- Weak on minority classes (surprise, love)

### Naive Bayes + Oversampling
- Accuracy: 81%
- Balanced performance across classes

### Random Forest (Best ML Model)
- Accuracy: 85%
- Strong across all classes

### LSTM Model
- Accuracy: 89%
- Best at capturing context in text

---

## Key Insights

- Class imbalance significantly impacts performance  
- Oversampling improves fairness  
- Random Forest provides strong baseline  
- LSTM achieves highest accuracy with context awareness  

---

# Conclusion

This project successfully demonstrates a **complete end-to-end NLP pipeline** for emotion classification.

It integrates:

- Data preprocessing  
- Feature engineering  
- EDA  
- Class imbalance handling  
- Multiple machine learning models  
- Deep learning (LSTM)  
- Model evaluation and interpretation  

---