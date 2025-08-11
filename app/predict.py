def predict_stunting(model, scaler, gender, age_months, height_cm, weight_kg):
    # Encoding jenis kelamin (Laki-laki = 1, Perempuan = 0)
    gender_encoded = 1 if gender == "Laki-laki" else 0
    
    # Menyiapkan fitur input
    features = [[gender_encoded, age_months, height_cm, weight_kg]]
    
    # Normalisasi data dengan scaler
    scaled_features = scaler.transform(features)
    
    # Melakukan prediksi (model multi-class)
    prediction = model.predict(scaled_features)
    
    # Mengembalikan hasil prediksi (kelas target)
    # Kelas: 0 = Tall, 1 = Stunted, 2 = Normal, 3 = Severely Stunted
    return prediction[0]  # Prediksi kelas target
