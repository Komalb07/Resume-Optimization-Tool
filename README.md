# Resume Optimization Tool

This repository contains the **Resume Optimization Tool**, a machine-learning-based system designed to quantify and predict the compatibility between resumes and job descriptions. This tool aims to address inefficiencies in traditional recruitment processes by providing an automated and objective solution.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Data Description](#data-description)
- [Implementation](#implementation)
- [Results](#results)
- [Future Work](#future-work)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Recruitment often relies on manual resume screening, which can be time-consuming, inconsistent, and subjective. The Resume Optimization Tool introduces a data-driven, scalable approach to streamline this process. By leveraging machine learning models and advanced feature engineering, the tool evaluates resume-job compatibility based on textual similarity and skill overlap.

## Features

- **Text Similarity Measurement:** Uses TF-IDF vectorization to calculate cosine similarity between resumes and job descriptions.
- **Skill Overlap Analysis:** Quantifies alignment of skills between candidate qualifications and job requirements.
- **Machine Learning Models:** Implements Linear Regression, Random Forest, and XGBoost to predict compatibility scores.
- **Compatibility Index:** Combines text similarity and skill overlap into a dynamic, weighted index for robust predictions.

## Data Description

The tool uses two primary datasets:
1. **Resumes Dataset:** Contains resumes categorized into 12 fields, capturing diverse skill sets.
2. **Job Postings Dataset:** Includes 27 attributes for data science roles, such as job title, location, and detailed descriptions.

A fabricated dataset of over 62,000 records was created, enriched with:
- Cleaned resumes and job descriptions.
- Cosine similarity and skill overlap scores.
- A dynamically generated compatibility index as the target variable.

## Implementation

### Key Steps:
1. **Data Cleaning:** Removed noise, duplicate entries, and irrelevant content from resumes and job descriptions.
2. **Feature Engineering:** 
   - Calculated cosine similarity using TF-IDF.
   - Computed skill overlap scores based on predefined technical and domain-specific skills.
3. **Modeling:** 
   - Trained Linear Regression, Random Forest, and XGBoost models.
   - Performed hyperparameter tuning and cross-validation to optimize performance.
4. **Evaluation:** Assessed models using MAE, MSE, and R² metrics.

### Technologies Used:
- Python (Pandas, NumPy, Scikit-learn, XGBoost)
- Natural Language Processing (TF-IDF, regex)
- Data Visualization (Matplotlib, Seaborn)

## Results

- **Linear Regression:** Achieved the best performance with R² = 0.9752, demonstrating a strong linear relationship between features and compatibility scores.
- **XGBoost:** Delivered competitive results with R² = 0.9744, excelling in handling outliers and complex interactions.
- **Random Forest:** Performed robustly with R² = 0.9650, effectively modeling non-linear relationships.

## Future Work

- Integrate advanced NLP embeddings like BERT or Word2Vec for enhanced semantic analysis.
- Expand feature sets with additional factors such as job seniority levels and salary ranges.
- Develop a web-based interface for real-time compatibility predictions.
- Explore deep learning architectures for improved accuracy and scalability.
