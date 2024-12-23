import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib

def load_data():
    # Load datasets
    train_df = pd.read_csv("data/train.csv", low_memory=False)
    store_df = pd.read_csv("data/store.csv", low_memory=False)

    # Merge datasets
    data = train_df.merge(store_df, on="Store", how="left")
    return data

def preprocess_data(data):
    # Fill missing values for numerical columns
    data["CompetitionDistance"] = data["CompetitionDistance"].fillna(data["CompetitionDistance"].median())
    data["PromoInterval"] = data["PromoInterval"].fillna("None")

    # Encode PromoInterval
    promo_intervals = {"None": 0, "Jan,Apr,Jul,Oct": 1, "Feb,May,Aug,Nov": 2, "Mar,Jun,Sep,Dec": 3}
    data["PromoInterval"] = data["PromoInterval"].map(promo_intervals)

    # Drop unnecessary columns
    data.drop(columns=["Date", "Customers"], inplace=True, errors="ignore")

    # Convert categorical columns to category dtype
    for col in ["StoreType", "Assortment", "StateHoliday"]:
        data[col] = data[col].astype("category")

    # One-hot encode categorical columns
    data = pd.get_dummies(data, drop_first=True)

    # Separate features and target
    X = data.drop(columns=["Sales"])
    y = data["Sales"]
    return X, y

def train_model():
    # Load and preprocess data
    data = load_data()
    X, y = preprocess_data(data)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    model = XGBRegressor(enable_categorical=True)
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, predictions))
    print(f"Model RMSE: {rmse}")

    # Save model
    joblib.dump(model, "model.pkl")
    print("Model saved as model.pkl")

if __name__ == "__main__":
    train_model()
