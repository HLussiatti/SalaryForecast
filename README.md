# <h1 align=center> ** SALARY FORECAST MODEL ** </h1>

## Introduction


## Objective

Develop a predictive model to forecast an individual's salary based on a the given dataset.
Design and implement a predictive model to forecast salaries, including all necessary preprocessing steps, feature engineering, and model evaluation logic.


# TABLA DE CONTENIDO
1. [Data Source Description](#1)
2. [Data Analysis and Preprocessing - ETL and EDA](#2)
3. [Machine Learning Model Comparison](#3)



# <h2 id="1">**1. Data Source Description**</h2>

Dataset was provided. There are 3 files.
1. [Peolple.csv](./datasets/1.%20Original%20Dataset/people.csv): contains information about Age, Gender, Education Level, Job Title and Years of Experience for each employee.
2. [Salary.csv](./datasets/1.%20Original%20Dataset/salary.csv): contains salaries values for each employee in the people.csv.
3. [Descriptions.csv](./datasets/1.%20Original%20Dataset/descriptions.csv): contains the job description written by the employee.


# <h2 id="2">**2. Data Analysis, Preprocessing & Feature Engineering**</h2>

- Data contains 375 employees' jobs information.
- Collected and cleaned salary dataset.
    - There are very few missing values. They are filled with information from Job Description when it's possible. Rows with null values were eliminated.
    - There are not outliers.

- Feature Engineering:
    - Job Title was analyzed in [NLP_Job_Titles](./notebooks/1.%20NLP_Job_Tiltes.ipynb) and split into Seniority and Industry.
    - Job Description was analyzed in [NLP_Job_Descriptions](./notebooks/2.%20NLP_Job_Descriptions.ipynb) to find words like Skills that helps the model, but none were found.
    - 'Job Title' and 'Description' were eliminated from the final dataset for final use into ML Model.
    - Normalization and PCA was applied.


# <h2 id="3">**3. Machine Learning Model Comparison**</h2>

We developed and optimized machine learning models to predict salaries based on various features. 
The main goal was to compare different models, tune hyperparameters, and evaluate their performance using multiple metrics.
Lienar Regression was used as a baseline model.

Results were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Percentage Root Mean Squared Error (%RMSE), and R-squared (R2) score.
These metrics help assess the accuracy and reliability of the predictions.
MAE and RMSE helps to quantify errors in prediction, RMSE penalize large errors more than MAE. %RMSE helps to quantify relative error to the predictive varbiable (Salary). Finally R2 quantify how much of the prediction variance is explained by the model.


# <h3 id="4"> **3.1. Initial Results (Before Hyperparameter Tuning)**</h3>

- Implemented **Linear Regression, Random Forest, XGBoost, and Neural Network** models.
- Evaluated models using **MAE, RMSE, %RMSE, and R²**.

| Model               | MAE    | RMSE    | %RMSE  | R²   |
|----------------------|--------|--------|--------|------|
| **Linear Regression** | 9579.99 | 13087.34 | 13.08% | 0.93 |
| **Random Forest**     | 7737.37 | 11778.31 | 11.77% | 0.94 |
| **XGBoost**          | 8030.99 | 12429.49 | 12.42% | 0.93 |
| **Neural Network**    | 9782.02 | 12843.88 | 12.84% | 0.90 |

---

# <h3 id="5"> **3.2. Best Hyperparameters Found and Results After Hyperparameter Tuning**</h3>

- Applied **Grid Search** and **Random Search** to optimize hyperparameters.
- Selected the best hyperparameters for each model.
🔹 **Best Model:** *Neural Network*, with the lowest RMSE and highest R².

| Model              | Best Hyperparameters |
|---------------------|-----------------------------------------------|
| **Linear Regression** | `{}` (no changes) |
| **Random Forest**    | `max_depth=8, min_samples_split=10, n_estimators=300` |
| **XGBoost**         | `learning_rate=0.1, max_depth=3, n_estimators=100` |
| **Neural Network**   | `alpha=0.01, hidden_layer_sizes=(64,32), learning_rate_init=0.01` |

---

| Model               | MAE    | RMSE    | %RMSE  | R²   |
|----------------------|--------|--------|--------|------|
| **Linear Regression** | 9580.00 | 13087.34 | 13.08% | 0.93 |
| **Random Forest**     | 8281.45 | 12637.87 | 12.63% | 0.93 |
| **XGBoost**          | 8645.05 | 12102.86 | 12.10% | 0.94 |
| **Neural Network**    | 6906.78 | 9880.52  | 9.87%  | 0.96 |

---

# <h3 id="6"> **3.3. Performance Comparison (Train vs Test R² and RMSE)**</h3>

- Compared **training vs. test performance** to check for overfitting.
- Conducted **cross-validation** to ensure model stability.
- Identified **Neural Network** as the best-performing model.
🔹 *Random Forest and XGBoost show slight overfitting, but Neural Network maintains the best balance between train and test performance.*

| Model               | Train R² | Test R² | Train RMSE | Test RMSE |
|----------------------|----------|---------|------------|-----------|
| **Linear Regression** | 0.92     | 0.93    | 13623.02   | 13087.34  |
| **Random Forest**     | 0.97     | 0.93    | 8432.13    | 12637.87  |
| **XGBoost**          | 0.98     | 0.94    | 7306.06    | 12102.86  |
| **Neural Network**    | 0.95     | 0.96    | 10943.24   | 9880.52   |

---

# <h3 id="7"> **3.4. Cross-Validation Results (Average R² ± Std)**</h3>

🔹 *Neural Network maintains the highest stability and performance in cross-validation.*

| Model               | Cross-Validation R² (Mean ± Std) |
|----------------------|----------------------------------|
| **Linear Regression** | 0.90 ± 0.05 |
| **Random Forest**     | 0.89 ± 0.07 |
| **XGBoost**          | 0.90 ± 0.05 |
| **Neural Network**    | 0.92 ± 0.03 |


---

# <h2 id="8">**Final Conclusion**</h2>
- 📌 *Neural Network* achieved the **best results** across all key metrics (lowest RMSE, highest R², and best cross-validation).  
- 📌 *XGBoost* also performed well, offering a good balance between training and test performance.  
- 📌 *Random Forest* had solid results but showed **slight overfitting**.  
- 📌 *Linear Regression* had acceptable performance but was outperformed by the other models.  

✅ **Recommendation:** Use **Neural Network** as the final optimized model. 🚀
