import joblib
import numpy as np
import pandas as pd
import logging
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

# ----------------- LOGGING -----------------
logger = logging.getLogger("heart-app")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("app.log", mode="a", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ----------------- MODEL LOADING -----------------
try:
    model = joblib.load("model.pkl")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.exception(f"Failed to load model: {e}")
    raise

# ----------------- FASTAPI -----------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ----------------- ENDPOINTS -----------------
@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.get("/health")
async def health_check():
    return {"status": "OK"}


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    systolic_blood_pressure: float = Form(...),
    bmi: float = Form(...),
    heart_rate: float = Form(...),
    previous_heart_problems: int = Form(...),
    cholesterol: float = Form(...),
    diet: int = Form(...),
    triglycerides: float = Form(...),
    stress_level: int = Form(...),
):
    try:
        input_data = {
            "cholesterol": cholesterol,
            "heart_rate": heart_rate,
            "diet": diet,
            "previous_heart_problems": previous_heart_problems,
            "stress_level": stress_level,
            "bmi": bmi,
            "triglycerides": triglycerides,
            "systolic_blood_pressure": systolic_blood_pressure
        }

        df = pd.DataFrame([input_data])
        logger.info(f"Incoming data: {input_data}")

        prediction = model.predict(df)[0]
        result = "High risk (1)" if prediction == 1 else "Low risk (0)"
        logger.info(f"Prediction: {result}")

    except Exception as e:
        result = f"Prediction error: {e}"
        logger.exception(f"Prediction failed with error: {e}")

    return templates.TemplateResponse("form.html", {
        "request": request,
        "result": result
    })


# ----------------- LAUNCHING -----------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)