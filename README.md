# Forecasting Nigeria's Inflation: A Machine Learning Approach to Economic Insights

## Table of Contents
- [Projecct overview](#project-overview)
- [Dataset](#dataset)
- [Tools Used](#tools-used)
- [Libraries](#libraries)
- [Methodology](#methodology)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Key Findings of the Project](#key-findings-of-the-Project)

## Project Overview

This project aims to forecast inflation rates in Nigeria using machine learning techniques and historical economic data. We analyze the relationships between various economic indicators such as crude oil prices, CPI (Consumer Price Index) categories, and inflation rates. The goal is to develop a predictive model that provides actionable insights for economic policymakers and businesses operating in Nigeria.

---

## Dataset

The dataset consists of monthly economic data from Nigeria, with features like:

- **Inflation Rate** (target variable)
- **Crude Oil Price**
- **CPI categories**: Food, Energy, Health, Transport, Communication, and Education
- **Oil Production and Exports**

The dataset contains 198 entries and 12 columns, This specific dataset can be found on [Kaggle](https://www.kaggle.com/datasets/iamhardy/nigeria-inflation-rates)


## Tools Used
This project leverages the following tools for data analysis and visualization:

Programming Language: Python 3

### Libraries:
1. ### Data Handling and Manipulation
- pandas
- NumPy
  2.### Data Visualization
- Matplotlib
- Seaborn
  3.### Machine Learning
- sci-kit-learn
- 4. ### Model Interpretability:
   SHAP (Shapley Additive exPlanations)
   
6. ### Other
IPython and Jupyter

# Methodology

## Data Preprocessing

### 1. Loading the Data
```python
import pandas as pd

# Load the dataset
df = pd.read_csv('path_to_dataset.csv')

# Preview the data
df.head()
```

### 2. Handling Missing Values and Outliers
- Missing values are imputed with the mean or median for continuous variables.
- Outliers are removed using the Z-score method.

```python
# Filling missing values with median
df.fillna(df.median(), inplace=True)

# Removing outliers using Z-score
from scipy import stats
df_no_outliers = df[(np.abs(stats.zscore(df.select_dtypes(include=['float64']))) < 3).all(axis=1)]
```

### 3. Feature Engineering
- A **Date** column is created by combining the `Year` and `Month` columns for time-series analysis.

```python
df_no_outliers['Date'] = pd.to_datetime(df_no_outliers[['Year', 'Month']].assign(DAY=1))
```
---

## Exploratory Data Analysis (EDA)

### 1. Summary Statistics
```python
# Get descriptive statistics
df_no_outliers.describe()
```

### 2. Data Visualization

#### a) Distribution of Inflation Rates
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Plot the distribution of inflation rates
sns.histplot(df_no_outliers['Inflation_Rate'], bins=30, kde=True)
plt.title('Distribution of Inflation Rates in Nigeria')
plt.show()
```

#### b) Inflation Over Time
```python
# Plot Inflation Rate over time
plt.plot(df_no_outliers['Date'], df_no_outliers['Inflation_Rate'], marker='o')
plt.title('Inflation Rate Over Time in Nigeria')
plt.xlabel('Date')
plt.ylabel('Inflation Rate')
plt.show()
```
![image](https://github.com/user-attachments/assets/8935d7a2-e4b2-47ec-841b-3c13d2abd515)

![image](https://github.com/user-attachments/assets/de4fa1cf-37cd-485d-87e2-e8692b3b2d44)

![image](https://github.com/user-attachments/assets/f7d06128-2fdd-45e0-98df-77e0d41bf30b)

![image](https://github.com/user-attachments/assets/d8d24f16-b6cf-49c3-9306-84abc099a164)

#### c) Correlation Heatmap
```python
import seaborn as sns

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_no_outliers.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Economic Features')
plt.show()
```
![image](https://github.com/user-attachments/assets/62dc0c93-5df9-473f-83ae-82ff45099e5e)

![image](https://github.com/user-attachments/assets/f50c3f1a-f21b-4f7b-9114-b3710e5826f2)

# Modeling

### 1. Train-Test Split
```python
from sklearn.model_selection import train_test_split

# Define features and target
X = df_no_outliers.drop(['Inflation_Rate', 'Date'], axis=1)
y = df_no_outliers['Inflation_Rate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2. Linear Regression Model
```python
from sklearn.linear_model import LinearRegression

# Initialize the model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train, y_train)

# Predictions
y_pred = linear_model.predict(X_test)
```

### 3. Random Forest Regression (for comparison)
```python
from sklearn.ensemble import RandomForestRegressor

# Initialize the Random Forest model
rf_model = RandomForestRegressor()

# Train the model
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)
```

## Evaluation

### 1. Linear Regression Metrics
```python
from sklearn.metrics import mean_squared_error, r2_score

# Linear Regression metrics
rmse_linear = mean_squared_error(y_test, y_pred, squared=False)
r2_linear = r2_score(y_test, y_pred)

print(f"Linear Regression RMSE: {rmse_linear}")
print(f"Linear Regression R² Score: {r2_linear}")
```

### 2. Random Forest Metrics
```python
# Random Forest metrics
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest RMSE: {rmse_rf}")
print(f"Random Forest R² Score: {r2_rf}")
```
![image](https://github.com/user-attachments/assets/2e855e89-d84e-4ac4-96ac-3420d9cca9ff)

# Key Findings of the Project: Forecasting Nigeria's Inflation

1. **Inflation Rate Prediction with Machine Learning**:
   - The **Linear Regression** model performed well in predicting inflation rates with an **R² score of 0.72**. This means the model explained 72% of the variance in the inflation rate, making it a reasonably good predictor.
   - The model's **Root Mean Square Error (RMSE)** was **1.93**, indicating that on average, the predicted inflation rate was within 1.93 percentage points of the actual rate.

2. **Feature Importance**:
   - **Crude Oil Price** and **Crude Oil Exports** showed a significant relationship with inflation, highlighting the importance of oil-related economic factors in Nigeria's inflation dynamics.
   - **Consumer Price Index (CPI)** categories, such as **CPI Food** and **CPI Energy**, also strongly correlate with inflation rates, as food and energy prices are major components of household expenses in Nigeria.

3. **Economic Insights**:
   - The analysis revealed that inflation trends in Nigeria are highly influenced by changes in oil prices and production levels, which is consistent with Nigeria’s oil-dependent economy.
   - The close correlation between CPI categories (especially food and energy) and inflation rates underlines the impact of consumer goods' price volatility on inflation.

4. **Outlier Impact**:
   - Outlier removal improved the model's performance by reducing the influence of extreme data points. This highlights the importance of cleaning and preprocessing economic data for better model accuracy.

5. **Potential for Enhanced Models**:
   - The project demonstrated that while **Linear Regression** provides solid baseline results, more sophisticated models (e.g., **Random Forest**, **ARIMA**, or **LSTM** for time series) could further improve prediction accuracy. Future work could involve hyperparameter tuning and implementing advanced time-series forecasting techniques.

6. **Correlation Between Features**:
   - The **correlation matrix** and **heatmap** showed a strong positive correlation between various CPI categories, suggesting interdependence between food, energy, and other sectors, which can amplify inflation effects when prices in one category rise sharply.

### Business and Policy Implications:
- Policymakers can leverage these insights to develop more informed strategies for inflation control by monitoring key factors such as oil prices and CPI metrics.
- Businesses can use this model to anticipate inflationary periods and adjust their pricing strategies, particularly in the food, energy, and transport sectors.
