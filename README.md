# Time_Series_ML_Project

# Context

The task at hand involves time series forecasting, where we will be working on predicting store sales using data from Corporation Favorita, a major grocery retailer based in Ecuador. The goal in this project is to develop a model that can provide more precise predictions for the unit sales of various items sold across different Favorita stores. 

# Procedure

The document will comprehensively detail the steps and procedures undertaken to successfully complete this project at every stage. The following steps were meticulously followed to attain the project's objectives.

# Steps

1. **Data Collection**: The Time series sales data utilized in this project is sourced from various provided databases and files, including a SQL Server database consisting of table1,2 and 3, as well as csv files from designated zip files and one drive. The dataset encompasses valuable details such as store_nbr,family,sales,onpromotion,test.csv,transaction.csv,sample_submission.csv,stores.csv,oil.csv,holidays_events.csv.
2. **Data Loading**: The collected data is loaded into the code and transformed into a suitable format for analysis. The pyodbc package is used to connect to the SQL Server database and fetch data from the a given table. The data from the CSV files is read using the pandas library and concatenated with the SQL data to create a comprehensive dataset.
3. **Data Evaluation (EDA)**: Exploratory data analysis is performed to gain insights into the dataset. This includes summarizing the data, checking for duplicates, handling missing values, and performing visual analyses using the sarima and adf test to check to spot pattern and trends within the given data. The pandas, numpy, matplotlib, and seaborn libraries are utilized for data manipulation and visualization.
4. **Data Processing and Engineering**: The dataset undergoes data processing steps to cleanse and preprocess it. These steps involve addressing missing values, transforming categorical variables, and potentially generating new features. Techniques from the pandas library are applied to prepare the dataset for subsequent analysis.
5. **Hypothesis Testing**: Time series-related hypotheses are formulated and subjected to statistical testing using methods from the scipy library. Hypothesis tests, such as the Chi-Square Test, Independence Test, and t-test, are employed to assess the significance of various factors.
6. **Answering Questions with Visualizations**: Essential inquiries concerning time series are addressed through informative visualizations. Utilizing the matplotlib and seaborn libraries, we create meaningful plots and charts that effectively illustrate the relationships between variables and time series data.
7. **Power BI Deployment**: The analysis and visualizations were deployed in Power BI, enabling interactive exploration and sharing with stakeholders. The insights obtained from the analysis were presented effectively using Power BI's dashboarding and reporting features.
8. **Train and Evaluate Four Models**: In this project, four machine learning models, namely ARIMA, SARIMA, XGBoost Regressor, and CatBoost Regressor, are trained and evaluated using both the imbalanced and balanced datasets. The evaluation metrics used for assessing model performance include Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Root Mean Log Squared Error (RMLSE).
9. **Evaluate Chosen Model**:  Advanced Model Improvement: For selected models, GridSearchCV is employed to conduct hyperparameter tuning. The best-tuned models and their optimized parameters are obtained through this process, and predictions are subsequently made using these refined models.
10. **Future Predictions**: The trained and validated time series model can be utilized to make predictions on new, unseen data. This enables businesses to forecast various time-dependent outcomes and take proactive measures accordingly. The model can be deployed in production to continuously monitor and predict future events or trends.

# Installation

* [ ] pyodbc
* [ ] sqlalchemy
* [ ] lightgbm
* [ ] catboost
* [ ] python-dotenv
* [ ] pandas
* [ ] numpy
* [ ] matplotlib
* [ ] seaborn
* [ ] scipy
* [ ] pmdarima

# Packages

* [ ] import pyodbc
* [ ] import sqlalchemy as sa
* [ ] import pandas as pd
* [ ] import numpy as np
* [ ] import zipfile
* [ ] import matplotlib.pyplot as plt
* [ ] import seaborn as sns
* [ ] import calendar
* [ ] from sklearn.preprocessing import StandardScaler, OneHotEncoder
* [ ] from sklearn.compose import ColumnTransformer
* [ ] from sklearn.pipeline import Pipeline
* [ ] from statsmodels.tsa.seasonal import seasonal_decompose
* [ ] from statsmodels.tsa.stattools import adfuller
* [ ] from sklearn.model_selection import train_test_split
* [ ] from dotenv import dotenv_values
* [ ] from scipy import stats
* [ ] from lightgbm import LGBMRegressor
* [ ] from statsmodels.tsa.arima.model import ARIMA
* [ ] from statsmodels.tsa.statespace.sarimax import SARIMAX
* [ ] from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
* [ ] from xgboost import XGBRegressor
* [ ] from pmdarima.arima import auto_arima
* [ ] from sklearn.model_selection import GridSearchCV
* [ ] from sklearn.linear_model import LinearRegression
* [ ] from sklearn.svm import SVR
* [ ] from catboost import CatBoostRegressor
* [ ] from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
* [ ] import warnings
* [ ] warnings.filterwarnings("ignore")

# Authors and Aknowledgement

Below is a table of the initial contributors of the project with their respective Github ID and Articles written to document their individual perspective of the project.

| Project LP3 | Contribitors        | Article Link | Github Profile |
| ----------- | ------------------- | ------------ | -------------- |
| 1.          | Israel Anaba Ayamga |              | Israel-Anaba   |
| 2.          | Isaac Sarpong       |              | IsaacSarpong   |
| 3.          | Peter Mutwiri       |              | PETERMUTWIRI   |
| 4.          | Emmanuel Morkeh     |              | Ekmorkeh       |

# Conclusion

In conclusion, this project involves tackling a time series forecasting problem. The utilization of time-dependent data and advanced modeling techniques has enabled us to make accurate predictions and gain valuable insights from the temporal patterns in the dataset. By leveraging the power of time series analysis, we can make informed decisions and effectively plan for the future.

# License

MIT-LICENSE.txt is an open-source software license widely used for distributing and sharing software, code, and other creative works.
