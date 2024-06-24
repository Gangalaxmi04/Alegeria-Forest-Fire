# Weather-Fire Dataset

This repository contains a dataset of daily weather observations and their relation to fire occurrences. The data includes various meteorological variables and a target variable indicating whether a fire occurred on a given day. Additionally, we provide the results of our machine learning models which achieved a remarkable 99% accuracy in predicting fire occurrences.

## Dataset Description

The dataset consists of the following attributes:
- `day`: Day of the month
- `month`: Month of the year
- `year`: Year of observation
- `Temperature`: Daily temperature (°C)
- `RH`: Relative humidity (%)
- `Ws`: Wind speed (km/h)
- `Rain`: Rainfall (mm)
- `FFMC`: Fine Fuel Moisture Code
- `DMC`: Duff Moisture Code
- `DC`: Drought Code
- `ISI`: Initial Spread Index
- `BUI`: Buildup Index
- `FWI`: Fire Weather Index
- `Classes`: Fire occurrence (`fire` or `not fire`)

## Usage

The dataset can be used for various purposes including:
- Analyzing the relationship between weather conditions and fire occurrences.
- Developing predictive models for fire occurrences.

### Model Performance

We trained and evaluated multiple regression models on the dataset, achieving an impressive accuracy rate of 99%. Below are the details of each model:

#### 1. Lasso Regression

Lasso regression adds a penalty equal to the absolute value of the magnitude of coefficients, helping in both variable selection and regularization.

- **Accuracy:** Lasso Regression MSE: 0.9593621779333094
                Lasso Regression R²: 0.9783246631008717

#### 2. Ridge Regression

Ridge regression adds a penalty equal to the square of the magnitude of coefficients, which helps in handling multicollinearity among the features.

- **Accuracy:** Ridge Regression MSE: 0.6949198918152113
                Ridge Regression R²: 0.9842993364555512
#### 3. Linear Regression

Linear regression, the baseline model without any regularization, also performed exceptionally well on this dataset.

- **Accuracy:**Linear Regression MSE: 0.6742766873791625
                Linear Regression R²: 0.984765738426695
#example code for each model
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from sklearn.metrics import mean_squared_error, r2_score
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print("Linear Regression MSE:", mse_lr)
print("Linear Regression R²:", r2_lr)
# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(x_train, y_train)
y_pred_lasso = lasso.predict(x_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print("Lasso Regression MSE:", mse_lasso)
print("Lasso Regression R²:", r2_lasso)
# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(x_train, y_train)
y_pred_ridge = ridge.predict(x_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print("Ridge Regression MSE:", mse_ridge)
print("Ridge Regression R²:", r2_ridge)
