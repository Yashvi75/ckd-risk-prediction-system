# Chronic Kidney Disease (CKD) Prediction

##  Overview

This project builds a machine learning model to predict the presence of Chronic Kidney Disease (CKD) using clinical and laboratory features. The pipeline follows a structured, research-oriented approach from data preprocessing to model evaluation and interpretation.

---

## Objective

To develop a reliable and interpretable classification model that can identify CKD cases with high recall, minimizing the risk of missed diagnoses.

---

## Machine Learning Pipeline

### 1. Data Preprocessing

* Cleaned inconsistent values (`?`, `\t?`)
* Removed rows with missing target labels
* Converted incorrect data types (e.g., pcv, wc, rc)
* Encoded categorical variables using Label Encoding
* Dropped non-informative features (e.g., ID)

### 2. Leakage Prevention

* Performed train-test split before scaling
* Applied scaling **only on training data**
* Ensured strict separation between training and test sets

### 3. Model Training

* Used Stratified Train-Test Split
* Applied Stratified K-Fold Cross Validation
* Compared multiple models:

  * Logistic Regression
  * Random Forest
  * XGBoost

### 4. Model Selection

* Selected **Logistic Regression** based on:

  * Comparable performance
  * Lower variance
  * Better interpretability
  * Simpler deployment

### 5. Hyperparameter Tuning

* Performed GridSearchCV on Logistic Regression
* Optimized regularization parameter (C)

---

## Model Performance

### Classification Report

* Precision: 1.00
* Recall: 1.00
* F1-score: 1.00

### Confusion Matrix

```
[[30  0]
 [ 0 47]]
```

* No false positives
* No false negatives

### ROC-AUC Score

* ROC-AUC: **1.00**

---

## Model Interpretation (SHAP)

Top contributing features:

1. Specific Gravity
2. Hemoglobin
3. Hypertension
4. Packed Cell Volume
5. Appetite
6. Diabetes Mellitus

These features align with clinical understanding of CKD, improving trust in model predictions.

---

## Critical Discussion

Although the model achieved perfect performance on the test dataset, this result should be interpreted cautiously.

Possible reasons:

* High feature separability
* Strong correlations between predictors and target
* Limited dataset size

Further validation on larger and real-world datasets is necessary to ensure generalizability.

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* SHAP

---

## Project Structure

```
CKD_Project/
├── data/
├── notebooks/
├── models/
├── src/
├── README.md
```

---

## Future Work

* Integrate model with Django frontend
* Validate on external datasets
* Deploy as a clinical decision support tool

---
## Advanced Improvements

### Feature Selection

* Reduced feature space using SHAP-based importance
* Retained top clinical predictors (e.g., hemoglobin, specific gravity, hypertension)
* Maintained performance while improving interpretability

### Model Calibration

* Applied sigmoid calibration using cross-validation
* Improved reliability of predicted probabilities for clinical decision-making

### Threshold Optimization

* Evaluated different probability thresholds
* Demonstrated trade-off between recall and precision
* Identified thresholds that maximize recall (critical for CKD detection)

### Cross-Validated Evaluation

* Performed 5-fold cross-validation on selected features
* Achieved stable recall with low variance
* Confirms robustness of the model

### Key Outcome

* Model remains highly accurate even with reduced features
* Improved trustworthiness and interpretability
* Better suited for real-world deployment scenarios

---

## Live Demo

The application is deployed using Streamlit and can be accessed here:

https://ckd-predictor.streamlit.app

### Features:

* Real-time CKD prediction
* User-friendly interface
* Calibrated ML model
* Risk probability output

## Author

Yashvi Vyas

