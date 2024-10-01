---
layout: default
---

## CS 4641 Project Proposal

### Group 115

## Introduction and Background

Machine Learning has the potential to transform the education industry in several different ways. Particularly, we are interested in the use of ML methods to predict student dropout rates.

Previous work describes “the pipeline built and followed for developing an explainable student dropout prediction system” [1]. Additional research demonstrated that the Logistic Regression model is more accurate and efficient with a **75.63%** accuracy rate in comparison to the K-NN model (**68.907%**) for predicting student dropout rates [2].

In our project, we aim to identify the key academic success factors and predict student dropouts using a Kaggle dataset. The dataset is tabular and contains several features collected at the time of student enrollment, including (but not limited to) marital status, previous grades, and major.

## Problem and Motivation

Thousands of students enroll in universities each year but do not graduate for various reasons. Some large and established universities in the United States report that barely **50%** of incoming freshmen complete their degree. Given the importance of education in society, it is crucial to identify the root causes of this issue and help universities address potential problems before they worsen. Our goal is to accurately predict whether a student is likely to graduate or not, providing valuable insights for university faculty to support students with a higher risk of dropping out.

## Data Processing

- **Exploratory Data Analysis (EDA) & Feature Selection/Engineering**: We will analyze which features provide the most valuable information and identify irrelevant features.
- **Principal Component Analysis (PCA)**: This feature reduction technique will help us boost the most important features and flatten those that do not add value to the prediction model.
- **One-Hot Encoding**: We may need to use this technique to transform categorical features into a format more suitable for machine learning models [3].

## Machine Learning Algorithms/Models

Our dataset contains the target label, making this a supervised learning task. As a baseline for our dropout classification problem, we will use the following models:

- **Random Forest (RandomForestClassifier)**: A robust model for classification tasks.
- **K-Nearest Neighbors (KNN)**: A simple but effective classification algorithm.
- **Logistic Regression (LogisticRegression)**: A popular and effective binary classification model.
- **Gradient Boosting**: We will explore using this ensemble method to improve model performance by combining multiple models [4].

## Results and Discussion

### Metrics

- **Test Accuracy**: We will split the dataset into training and testing sets, with the test set being approximately 20% of the total data. We will evaluate model accuracy on this unseen test data.
- **Precision/Recall**: Given that the dataset is unbalanced, we will use the AUC ROC metric to get a more unbiased view of model performance.
- **Cross-Validation Accuracy**: We will use cross-validation to fine-tune hyperparameters and measure model robustness.

## References

1. Corrêa Krüger J.G. (2023). An explainable machine learning approach for student dropout prediction. _Expert Systems with Applications_. Available at: [https://www.sciencedirect.com/science/article/pii/S0957417423014355](https://www.sciencedirect.com/science/article/pii/S0957417423014355) (Accessed: 28 September 2024).
2. Sharma M. and Yadav M. (2022). Predicting students’ drop-out rate using machine learning models: A comparative study. _IEEE Xplore_. Available at: [https://ieeexplore.ieee.org/document/9917841](https://ieeexplore.ieee.org/document/9917841) (Accessed: 28 September 2024).
3. Documentation of scikit-learn 0.21.3¶ (no date). _Learn_. Available at: [https://scikit-learn.org/0.21/documentation.html](https://scikit-learn.org/0.21/documentation.html) (Accessed: 28 September 2024).
4. XGBoost documentation (no date). _XGBoost Documentation - xgboost 2.1.1 documentation_. Available at: [https://xgboost.readthedocs.io/en/stable/](https://xgboost.readthedocs.io/en/stable/) (Accessed: 28 September 2024).

## Gantt Chart

[View on Google Drive](https://docs.google.com/spreadsheets/d/19o6ZakfyxPPRYyXEH4_JTpknSqOLHHVz/edit?usp=sharing&ouid=112025817987775005881&rtpof=true&sd=true)

## Contributions

| Name               | Contributions                                            |
| ------------------ | -------------------------------------------------------- |
| Aman Patel         | Introduction, Dataset Description, Metrics, Video        |
| Marko Gjurevski    | Gantt Table, Video, Slides, Problem and Motivation Video |
| Rustam Jumazhanov  | Literature Review, GitHub Pages Setup (Jekyll & Docker)  |
| Oleksandr Horielko | Introduction, Literature Review, Reference Section       |
| Aldinash Seitenov  | Data Processing, Machine Learning Algorithms/Model       |

## Video Presentation

<iframe width="560" height="315" src="https://www.youtube.com/embed/C_gpO43Xtxg?si=Y1Oo1im8FzmFBEA3" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
