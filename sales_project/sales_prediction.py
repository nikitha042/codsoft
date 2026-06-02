# Sales Prediction Using Python

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Dataset
df = pd.read_csv('Advertising.csv')

# Display First 5 Rows
print("Dataset Preview:")
print(df.head())

# Dataset Information
print("\nDataset Info:")
print(df.info())

# Define Features and Target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict Sales
predictions = model.predict(X_test)

# Evaluate Model
score = r2_score(y_test, predictions)

print("\nR2 Score:", score)

# Sample Predictions
results = pd.DataFrame({
    'Actual Sales': y_test,
    'Predicted Sales': predictions
})

print("\nSample Predictions:")
print(results.head(10))

# Visualization
plt.scatter(y_test, predictions)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

