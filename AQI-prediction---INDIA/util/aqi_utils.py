import pandas as pd

class IndiaAQICalculator:

    def __init__(self):
        self.breakpoints = {
            "PM2.5":[(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,1000,401,500)],
            "PM10":[(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400),(431,1000,401,500)],
            "NO2":[(0,40,0,50),(41,80,51,100),(81,180,101,200),(181,280,201,300),(281,400,301,400),(401,800,401,500)],
            "SO2":[(0,40,0,50),(41,80,51,100),(81,380,101,200),(381,800,201,300),(801,1600,301,400),(1601,3000,401,500)],
            "CO":[(0.0,1.0,0,50),(1.0,2.0,51,100),(2.0,10.0,101,200),(10.0,17.0,201,300),(17.0,34.0,301,400),(34.0,50.0,401,500)],
            "O3":[(0,50,0,50),(51,100,51,100),(101,168,101,200),(169,208,201,300),(209,748,301,400),(749,1000,401,500)]
        }

    def calculate_sub_index(self, concentration, breakpoints):

        if concentration is None or pd.isna(concentration):
            return None

        for Clow, Chigh, Ilow, Ihigh in breakpoints:
            if Clow <= concentration <= Chigh:
                return ((Ihigh-Ilow)/(Chigh-Clow))*(concentration-Clow)+Ilow

        return breakpoints[-1][3]

    def calculate_aqi(self,row):

        sub_indices={
            "PM2.5":self.calculate_sub_index(row.PM2_5_ugm3,self.breakpoints["PM2.5"]),
            "PM10":self.calculate_sub_index(row.PM10_ugm3,self.breakpoints["PM10"]),
            "NO2":self.calculate_sub_index(row.NO2_ugm3,self.breakpoints["NO2"]),
            "SO2":self.calculate_sub_index(row.SO2_ugm3,self.breakpoints["SO2"]),
            "O3":self.calculate_sub_index(row.O3_ugm3,self.breakpoints["O3"]),
            "CO":self.calculate_sub_index(row.CO_ugm3/1000,self.breakpoints["CO"])
        }

        sub_indices={k:v for k,v in sub_indices.items() if v is not None}

        if len(sub_indices)==0:
            return None

        return round(max(sub_indices.values()))

def aqi_category(aqi):

    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"
    
def predict_aqi(data, model, label_encoders, feature_names):

    df = pd.DataFrame([data])

    for col, encoder in label_encoders.items():
        if col in df.columns and df[col].dtype == "object":
            df[col] = encoder.transform(df[col])

    safe_features = [f for f in feature_names if f != "AQI_target"]  # Guard against stale feature_names.pkl
    df = df[safe_features]

    prediction = model.predict(df)

    return prediction[0]