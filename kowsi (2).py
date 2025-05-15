#Import necessary libraries for data handling, analysis, and machine learning models
import pandas as pd  # Data handling and manipulating
import numpy as np  # Numerical operations
import seaborn as sns  # Data visualization
import matplotlib.pyplot as plt  # Data visualization
from sklearn.model_selection import train_test_split  # Train-Test split
from sklearn.ensemble import RandomForestRegressor  # Random Forest model
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.svm import SVR  # Support Vector Regression (SVR)
from sklearn.tree import DecisionTreeRegressor  # Decision Tree model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Evaluation metrics
from sklearn.preprocessing import LabelEncoder  # Encode non-numeric features
# If the AirQuality.csv file is in the current working directory:
air_data = pd.read_csv('AirQuality.csv', delimiter=';') #replace with the path of the dataset
# OR, If the AirQuality.csv file is in the 'data' subdirectory:
# air_data = pd.read_csv('data/AirQuality.csv', delimiter=';')
# OR, If the file is located elsewhere, replace with absolute path of the dataset:
# air_data = pd.read_csv('/path/to/your/file/AirQuality.csv', delimiter=';')  
air_data.head()
# Drop unnamed columns
air_data = air_data.drop(columns=['Unnamed: 15', 'Unnamed: 16'])  

# Replace incorrect characters (',' to '.')
air_data = air_data.replace({',': '.'}, regex=True)

# Fill missing numeric columns with their mean
numeric_cols = air_data.select_dtypes(include=np.number).columns
air_data[numeric_cols] = air_data[numeric_cols].fillna(air_data[numeric_cols].mean())

# Drop rows with all NaN values (if any)
air_data.dropna(axis=0, how="all", inplace=True)

# Create a datetime feature by combining Date and Time
air_data['datetime'] = pd.to_datetime(air_data['Date'] + ' ' + air_data['Time'], format='%d/%m/%Y %H.%M.%S')

# Derive additional features such as month, day, day of the week, and hour
air_data['month'] = air_data['datetime'].dt.month
air_data['day'] = air_data['datetime'].dt.day
air_data['dayofweek'] = air_data['datetime'].dt.dayofweek
air_data['hour'] = air_data['datetime'].dt.hour

# Drop the original Date and Time columns
air_data.drop(columns=['Date', 'Time'], inplace=True)

# Convert datetime column to numeric: number of days since minimum date
air_data['datetime'] = (air_data['datetime'] - air_data['datetime'].min()).dt.days
# Visualize correlations using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(air_data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Map of Features")
plt.show()
# Data Preparation: Separate features (X) and target (y)
X = air_data.drop(['CO(GT)'], axis=1)  # Features
y = pd.to_numeric(air_data['CO(GT)'], errors='coerce')  # Target

# Encode non-numeric columns in X using LabelEncoder
non_numeric_columns = X.select_dtypes(exclude=['number']).columns
encoder = LabelEncoder()

for column in non_numeric_columns:
    X[column] = encoder.fit_transform(X[column])

# Handle missing values
X = X.fillna(X.mean())  # Fill missing values in X with mean

# Fill missing values in target variable y
y = y.fillna(y.mean())  # Fill missing target values with mean
# Split the data into training and testing sets (80%-20% split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define models for training
models = {
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "LinearRegressor": LinearRegression(),
    "SVR": SVR(),
    "DecisionTreeRegressor": DecisionTreeRegressor()
}

# Dictionary to store evaluation metrics
results = {}
# Train and evaluate each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics: MAE and R2 Score
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store results in the dictionary
    results[model_name] = {
        "Mean Absolute Error": round(mae, 4),
        "R2 Score": round(r2, 4)
    }
    # Visualization of Model Evaluation Results

# Prepare Data for plotting
model_names = list(results.keys())
mae_scores = [results[model]['Mean Absolute Error'] for model in model_names]
r2_scores = [results[model]['R2 Score'] for model in model_names]

# Plot Mean Absolute Error and R2 Score with enhancements
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot Mean Absolute Error (MAE) with Horizontal Bars
axes[0].barh(model_names, mae_scores, color='lightblue', edgecolor='black')
axes[0].set_title('Model Comparison: Mean Absolute Error (MAE)', fontsize=14)
axes[0].set_xlabel('Mean Absolute Error (MAE)', fontsize=12)
axes[0].set_ylabel('Model', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.7)

# Annotate MAE values on bars
for i, (score, model) in enumerate(zip(mae_scores, model_names)):
    axes[0].text(score + 1, i, f'{score:.4f}', va='center', fontsize=11)

# Plot R2 Score with Horizontal Bars
axes[1].barh(model_names, r2_scores, color='lightgreen', edgecolor='black')
axes[1].set_title('Model Comparison: R2 Score', fontsize=14)
axes[1].set_xlabel('R2 Score', fontsize=12)
axes[1].set_ylabel('Model', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.7)

# Annotate R2 scores on bars
for i, (score, model) in enumerate(zip(r2_scores, model_names)):
    axes[1].text(score + 0.03, i, f'{score:.4f}', va='center', fontsize=11)

# Add Average Line for MAE
avg_mae = np.mean(mae_scores)
axes[0].axvline(avg_mae, color='red', linestyle='--', label=f'Avg MAE: {avg_mae:.4f}')
axes[0].legend(loc='best')

# Add Average Line for R2
avg_r2 = np.mean(r2_scores)
axes[1].axvline(avg_r2, color='red', linestyle='--', label=f'Avg R2: {avg_r2:.4f}')
axes[1].legend(loc='best')

plt.tight_layout()
plt.show()