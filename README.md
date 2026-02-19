# machine-learning-specialization-capstone# Machine Learning Specialization – Titanic Survival Prediction (Capstone)

## Project Overview
This project is an end-to-end machine learning capstone developed as part of the **Machine Learning Specialization by DeepLearning.AI (Andrew Ng)**.  
The goal is to predict passenger survival on the Titanic using supervised learning techniques, with a strong focus on **model evaluation, comparison, and interpretability**.

---

## Problem Statement
Given passenger demographic and travel information, the task is to build a binary classification model that predicts whether a passenger survived the Titanic disaster.

---

## Dataset
- Source: Kaggle – *Titanic: Machine Learning from Disaster*
- File used: `train.csv`
- Rows: 891 passengers
- Target variable: `Survived` (0 = No, 1 = Yes)

### Key Features
| Feature | Description |
|------|------------|
| Pclass | Passenger ticket class (1st, 2nd, 3rd) |
| Sex | Passenger gender |
| Age | Age in years |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Fare | Ticket fare |
| Embarked | Port of embarkation |

---

## Feature Engineering
Additional features were created to capture family structure:
- **FamilySize** = SibSp + Parch
- **IsAlone** = 1 if passenger traveled alone, else 0

Categorical variables were encoded and missing values handled appropriately.

---

## Models Used
Two supervised classification models were implemented and compared:

### Logistic Regression
- Scaled features using `StandardScaler`
- Implemented via `Pipeline`
- Focused on interpretability and probabilistic output

### Random Forest Classifier
- Ensemble, non-linear model
- Used `class_weight="balanced"` to handle class imbalance
- No feature scaling required

---

## Evaluation Metrics
Models were evaluated using:
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- **ROC-AUC score**
- ROC Curve visualization

---

## Results

| Model | Accuracy | ROC-AUC |
|------|----------|---------|
| Logistic Regression | 0.8045 | **0.8513** |
| Random Forest | **0.8212** | 0.8288 |

---

## Model Selection Rationale
Although Random Forest achieved slightly higher accuracy, **Logistic Regression was selected as the final model** due to:
- Higher ROC-AUC score (better class separability)
- More stable and well-calibrated probability estimates
- Greater interpretability
- Lower risk of overfitting

This decision aligns with best practices emphasized in the Machine Learning Specialization:  
**evaluation should go beyond accuracy alone**.

---

## ROC Curve
The ROC curve demonstrates that Logistic Regression significantly outperforms a random classifier, achieving high true positive rates even at low false positive rates.

---

## Technologies Used
- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook
- Git & GitHub

---

## How to Run
```bash
git clone https://github.com/Sara-Hosseini/machine-learning-specialization-capstone
cd machine-learning-specialization-capstone
pip install -r requirements.txt
jupyter notebook
