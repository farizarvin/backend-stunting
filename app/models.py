import joblib

def load_model(model_path: str):
    """Memuat model dan scaler dari file .pkl"""
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["scaler"]
