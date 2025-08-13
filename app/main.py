from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.models import load_model
from app.predict import predict_stunting

model, scaler = load_model("app/model.pkl")

app = FastAPI(
    title="Stunting Prediction API",
    description="API untuk memprediksi status stunting berdasarkan data pengguna",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StuntingInput(BaseModel):
    gender: str  # Jenis Kelamin
    age_months: int  # Umur dalam bulan
    height_cm: float  # Tinggi Badan dalam cm
    weight_kg: float  # Berat Badan dalam kg

class StuntingPredictionResponse(BaseModel):
    prediction: str  # Kelas prediksi: Tall, Stunted, Normal, Severely Stunted

@app.post("/predict/stunting", response_model=StuntingPredictionResponse)
async def predict(data: StuntingInput):
    """Endpoint untuk prediksi stunting"""
    try:
        pred_class = predict_stunting(model, scaler, data.gender, data.age_months, data.height_cm, data.weight_kg)

        class_names = ["Tall", "Stunted", "Normal", "Severely Stunted"]
        prediction = class_names[pred_class]
        
        return StuntingPredictionResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

@app.get("/", response_model=dict)
def read_root():
    return {"message": "Welcome to the Stunting Prediction API!"}

@app.get("/health", response_model=dict)
def health_check():
    """Endpoint untuk memeriksa status kesehatan API dan model"""
    try:
        model_status = "Loaded" if model else "Not Loaded"
        scaler_status = "Loaded" if scaler else "Not Loaded"
        
        return {
            "status": "healthy",
            "model_status": model_status,
            "scaler_status": scaler_status,
            "message": "API and models are running"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")
