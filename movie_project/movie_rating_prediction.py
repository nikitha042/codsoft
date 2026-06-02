import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("movies.csv", encoding="latin1")

print("First 5 Rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

# Keep only required columns
df = df[['Rating', 'Votes']]

# Remove missing values
df = df.dropna()

# Clean Votes column
df['Votes'] = df['Votes'].astype(str)
df['Votes'] = df['Votes'].str.replace(',', '', regex=False)
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

# Convert Rating to numeric
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# Remove invalid rows
df = df.dropna()

print("\nRows Remaining:", len(df))

# Features and Target
X = df[['Votes']]
y = df['Rating']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Sample Predictions
results = pd.DataFrame({
    'Actual Rating': y_test.values,
    'Predicted Rating': predictions
})

print("\nSample Predictions:")
print(results.head(10))

# Visualization
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Movie Rating Prediction")
plt.show()