# Chronic Kidney Disease Prediction

## Objective
Build a machine learning model to predict CKD using clinical parameters.

## Project Structure
- data/ → dataset
- notebooks/ → EDA, preprocessing, training, evaluation
- models/ → saved models
- src/ → utility scripts

## Setup
```bash
pip install -r requirements.txt

## EDA Summary
- Dataset contains significant missing values
- Presence of inconsistent entries like '?' and '\t?'
- Some numeric columns stored as object
- Hemoglobin and serum creatinine show strong predictive patterns
