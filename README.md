# <h1 align=center> **SALARY FORECAST MODEL** </h1>

## Objective

Develop a predictive model to forecast an individual's salary based on a the given dataset.
Design and implement a predictive model to forecast salaries, including all necessary preprocessing steps, feature engineering, and model evaluation logic.


# Table of Contents
1. [Data Source Description](#1)
2. [Data Analysis and Preprocessing - ETL and EDA](#2)
3. [Machine Learning Model Comparison](#3)
    - [3.1. Hyperparameters Tuning Results](#4)
    - [3.2. Performance Comparison (Train vs Test)](#5)
    - [3.3. Cross-Validation Results](#6)
    - [3.4. Feature importance](#7)
4. [Conclusions](#8)



# <h2 id="1">**1. Data Source Description**</h2>

Dataset was provided. There are 3 files.
1. [Peolple.csv](./datasets/1.%20Original%20Dataset/people.csv): contains information about Age, Gender, Education Level, Job Title and Years of Experience for each employee.
2. [Salary.csv](./datasets/1.%20Original%20Dataset/salary.csv): contains salaries values for each employee in the people.csv.
3. [Descriptions.csv](./datasets/1.%20Original%20Dataset/descriptions.csv): contains the job description written by the employee.


# <h2 id="2">**2. Data Analysis, Preprocessing & Feature Engineering**</h2>

- Data contains 375 employees' jobs information.
- Collected and cleaned salary dataset.
    - There are very few missing values. They are filled with information from Job Description when it's possible. Rows with null values were eliminated.
- Main EDA conclusions:
    - Gender: de dataset is balanced for this feature.
    - There are not outliers.
    - Dataset contains most employees with Bachelor's degree with Senior level of Seniority.
    - Highest Salaries are for PhD Eductation Level and Director/Executive Seniority.


<p align="center">
<img src="./img/Salary Boxplot.png"  style="max-width: 100%; height: auto;">
</p>

- Correlation:
    - Years of Experience and Age present the highest correlation with Salary
    - Eductation Level and Seniority have lower correlation with Salary
    - Gender has no correlation with any variable


<p align="center">
<img src="./img/Heatmap Correlation Matrix.png"  style="max-width: 100%; height: auto;">
</p>

- Feature Engineering:
    - Job Title was analyzed in [NLP_Job_Titles](./notebooks/1.%20NLP_Job_Tiltes.ipynb) and split into Seniority and Industry.
    - Job Description was analyzed in [NLP_Job_Descriptions](./notebooks/2.%20NLP_Job_Descriptions.ipynb) to find words like Skills that helps the model, but none were found.
    - 'Job Title' and 'Description' were eliminated from the final dataset for final use into ML Model.
    - Normalization and PCA was applied. There's not a significant dimensionality reduction beacause of the few features.

# <h2 id="3">**3. Machine Learning Model Comparison**</h2>

This project aims to predict employee salaries using Machine Learning models. Various regression models were evaluated, including dimensionality reduction with PCA to improve efficiency.
The main goal was to compare different models, tune hyperparameters, and evaluate their performance using multiple metrics.

- **Evaluated Models**:  
  - Linear Regression  
  - Random Forest Regressor  
  - XGBoost Regressor  
  - Neural Network (MLPRegressor)  
- **Model Evaluation**:  
  - `GridSearchCV` was used for hyperparameter tuning.  
  - Metrics calculated: `Bias`, `MAE`, `RMSE`, `%RMSE`, and `R²`.  
  - Compared **training vs. test performance** to check for overfitting.
  - **cross-validation** was performed using `cross_val_score` to assess model stability.
  - Bootstrap was applied to calculate Confidence Intervals.
  - Feature importance and SHAP analysis was applied.
  


# <h3 id="4"> **3.1. Hyperparameters Tuning Results**</h3>

- **Linear Regression:** Default parameters
- **Random Forest:** `max_depth=8`, `min_samples_split=5`, `n_estimators=200`
- **XGBoost:** `learning_rate=0.2`, `max_depth=3`, `n_estimators=50`
- **Neural Network:** `alpha=0.0001`, `hidden_layer_sizes=(128, 64)`, `learning_rate_init=0.01`

---

**Key Observations:**
1. XGBoost and the Neural Network demonstrate the best performance on the test data, achieving the lowest RMSE and the highest R² but Neural Network shows the highest bias, indicating a stronger systematic underprediction.
3. Random Forest performs slightly worse than XGBoost in terms of RMSE but shows competitive results overall.
4. The 95% confidence intervals demonstrate the stability of the XGBoost model, with relatively narrow ranges for MAE and RMSE.
5. The R² interval indicates the model consistently explains a high proportion of variance in the test data.

<div align="center">
| Model              | Bias      | MAE      | RMSE     | %RMSE  | R²    |  
|-------------------|----------|----------|----------|--------|-------|  
| **Linear Regression**  | -70.940,84 | 11.362,59 | 16.472,49 | 16,86% | 0,8922 |  
| **Random Forest**      | 91.563,08 | 9.815,12  | 16.129,81 | 16,51% | 0,8967 |  
| **XGBoost**           | -36.244,40 | 10.069,25 | 15.401,40 | 15,76% | 0,9058 |  
| **Neural Network**    | -297.478,93  | 10.704,20 | 15.397,06 | 15,76% | 0,9058 | 
</div>


- Confidence Intervals (95% CI) for XGBoost results:
    - **MAE:** [7.834,30, 12.808,93]
    - **RMSE:** [11.157,84, 19.961,32]
    - **R²:** [0,8569, 0,9476]

---

# <h3 id="5"> **3.2. Performance Comparison (Train vs Test)**</h3>

**Key Observations:**
- Linear Regression performs the worst across all metrics, which is expected given the complexity of the data.
- The gap between Train R² and Test R² for XGBoost is not excessively large, but it does suggest some degree of overfitting. However, the Test R² (0.9058) is still quite high, indicating that the model generalizes well.


| Model             | Train R² | Test R² | Train RMSE | Test RMSE | Train MAE | Test MAE | Train %RMSE | Test %RMSE |
|------------------|---------|--------|-----------|-----------|---------|---------|-------------|------------|
| **Linear Regression** | 0.9112  | 0.8922  | 14,220.76  | 16,472.49  | 10,555.59  | 11,362.59  | 13.99% | 16.86% |
| **Random Forest**    | 0.9771  | 0.8969  | 7,222.74  | 16,114.95  | 4,621.83  | 9,815.12  | 7.10% | 16.49% |
| **XGBoost**          | 0.9828  | 0.9058  | 6,260.04  | 15,401.40  | 4,389.41  | 10,069.25  | 6.16% | 15.76% |
| **Neural Network**    | 0.9475  | 0.9058  | 10,934.10  | 15,397.06  | 7,753.75  | 10,704.20  | 10.75% | 15.76% |

---

# <h3 id="6"> **3.3. Cross-Validation Results**</h3>

**Key Observations:**
- XGBoost and Neural Network have the **highest cross-validation scores** (0.89) with relatively low standard deviation, demonstrating their consistency.

| Model               | CV Mean R² | CV Std R² |  
|---------------------|------------|------------|  
| **Linear Regression** | 0.88       | 0.07       |  
| **Random Forest**     | 0.87       | 0.08       |  
| **XGBoost**          | 0.89       | 0.07       |  
| **Neural Network**   | 0.89       | 0.06       |  

---

# <h3 id="7"> **3.4. Feature importance**</h3>
The most influential features in the **XGBoost model** are PC2 meaning that Industry is not important and ther're similar contribution from every feature but Gender.
<p align="center">
<img src="./img/XGBoost Feature Importance.png"  style="max-width: 100%; height: auto;">
</p>


# <h3 id="8"> **4. Conclusion**</h3>

- **XGBoost and Neural Network performed the best**, achieving the lowest RMSE (~15.76% error rate).  
- **XGBoost** provides a balanced trade-off between predictive performance and computational efficiency. May require more careful hyperparameter Possible actions to reduce overfitting:
    1. Increase regularization (reg_lambda, reg_alpha in XGBoost).
    2. Reduce model complexity (max_depth, min_child_weight).
    3. Increase training data to improve generalization.
    4. Apply early stopping to prevent overfitting.
- **Cross-validation confirmed that Neural Networks had the most stable performance**, with the lowest variance in `R²`.  
- **Feauture SHAP Analysis** shows PC2 (has similar contribution from every feature but Gender) is the main feature explaining the model.


- **Future Work:**
    - Further fine-tune XGBoost parameters.
    - Perform deeper feature importance analysis to identify the most influential predictors.
    - Consider feature engineering for improving model accuracy.
    - Investigate alternative architectures for the Neural Network.
    - Explore ensemble methods to combine predictions from the top-performing models.
    

# <h3>**Requirements**</h3>
- Python 3.7 o superior
- padas
- numpy
- ydata_profiling.
- matplotlib.
- seaborn.
- scikit-learn
- xgboost
- shap