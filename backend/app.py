from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn

# Load the trained model
model = joblib.load("model.pkl")

# Initialize FastAPI app
app = FastAPI()


# Input schema for single prediction
class SalesInput(BaseModel):
    store_id: int
    day_of_week: int
    promo: bool
    state_holiday: str
    school_holiday: bool


@app.get("/")
def home():
    return {"message": "Retail Sales Prediction API is running!"}


@app.post("/predict/")
def predict_sales(data: SalesInput):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([{
        "Store": data.store_id,
        "DayOfWeek": data.day_of_week,
        "Promo": int(data.promo),
        "StateHoliday": data.state_holiday,
        "SchoolHoliday": int(data.school_holiday),
        "Open": 1,  # Default value if 'Open' is missing
        "CompetitionDistance": 0,  # Default or estimated value
        "CompetitionOpenSinceMonth": 0,
        "CompetitionOpenSinceYear": 0,
        "Promo2": 0,
        "Promo2SinceWeek": 0,
        "Promo2SinceYear": 0,
        "PromoInterval": 0
    }])

    # Match feature names from training
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Ensure all columns match model expectations
    missing_cols = set(model.feature_names_in_) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0

    # Reorder columns to match the training data
    input_df = input_df[model.feature_names_in_]

    # Make prediction
    prediction = model.predict(input_df)[0]
    return {"predicted_sales": float(prediction)}  # Convert to native Python float



@app.post("/batch_predict/")
async def batch_predict(file: UploadFile = File(...)):
    # Load test dataset
    test_df = pd.read_csv(file.file)

    # Merge with store data if required (assuming `store.csv` is available)
    store_df = pd.read_csv("data/store.csv")
    test_df = test_df.merge(store_df, on="Store", how="left")

    # Preprocess test data
    test_df.fillna(0, inplace=True)  # Handle missing values
    test_df = pd.get_dummies(test_df, drop_first=True)  # Encode categorical columns

    # Predict
    predictions = model.predict(test_df)
    test_df["predicted_sales"] = predictions

    return {"predictions": test_df["predicted_sales"].tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
