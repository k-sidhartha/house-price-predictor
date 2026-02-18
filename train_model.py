import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

print("hello from train_model")

# Load dataset
df = pd.read_csv("house_data.csv")

# Select features & target
X = df[["bedrooms", "bathrooms", "sqft_living", "sqft_lot",  "floors", "waterfront", "view", "condition",
        "sqft_above", "sqft_basement", "yr_built", "yr_renovated"]]
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model using joblib (SAFE)
joblib.dump(model, "house_price_model.joblib")

print("✅ Model trained and saved successfully")
