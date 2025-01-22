from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from mangum import Mangum 
import pickle
import numpy as np
import pandas as pd
from typing import Tuple

# Initialize FastAPI app
app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Load the model and scaler
model = pickle.load(open('air_quality.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Route for the frontpage
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("frontpage.html", {"request": request})

# Route for the frontend (predict form)
@app.get("/frontend", response_class=HTMLResponse)
async def frontend(request: Request):
    return templates.TemplateResponse("frontend.html", {"request": request})

# Route to handle prediction
@app.post("/predict", response_class=JSONResponse)
async def predict(
    temperature: float = Form(...),
    humidity: float = Form(...),
    pm25: float = Form(...),
    pm10: float = Form(...),
    so2: float = Form(...),
    no2: float = Form(...),
    co: float = Form(...),
    proximity: float = Form(...),
    population: float = Form(...)):

    # Calculate PM
    pm = pm25 - pm10
    features = pd.DataFrame([[temperature, humidity, pm, no2, so2, co, proximity, population]], 
                            columns=['Temperature', 'Humidity', 'PM', 'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density'])

    # Scale features
    features_scaled = scaler.transform(features)
    scaled_features_df = pd.DataFrame(features_scaled, columns=features.columns)

    # Predict using the model
    prediction = model.predict(scaled_features_df)
    prediction = int(prediction[0])
    
    # Return result as a JSON response
    if prediction == 0:
        result_message = "Prediction: Uh oh 😞, the Air Quality is POOR in your area. Please stay safe 🙏"
    else:
        result_message = "Prediction: The Air Quality in your area looks GOOD 😊."

    return {"Prediction": result_message}

# Run the FastAPI app on uvicorn server
import uvicorn

if __name__ == "__main__":
    uvicorn.run("fast:app", host="127.0.0.1", port=8000, reload=True)

handler = Mangum(app)