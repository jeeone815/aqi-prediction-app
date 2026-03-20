import pandas as pd
import numpy as np
from util.aqi_utils import IndiaAQICalculator
from sklearn.preprocessing import LabelEncoder

def clean_dataset(df):

    null_pct = df.isnull().sum()/len(df)*100
    cols_all_null = null_pct[null_pct >= 99].index.tolist()

    df = df.drop(columns=cols_all_null)

    return df

def feature(df):

    df["Winter"] = df["Month"].isin([11,12,1,2]).astype(int)
    df["Rush_Hour"] = df["Hour"].isin([7,8,9,17,18,19,20]).astype(int)

    return df

def shifing(df):

    pollutant = ["PM2_5_ugm3", "PM10_ugm3", "O3_ugm3"]

    for pol in pollutant:
        df[f"{pol}_lag1"] = df.groupby("City")[pol].shift(1)
        df[f"{pol}_lag3"] = df.groupby("City")[pol].shift(3)
        df[f"{pol}_roll6"] = df.groupby("City")[pol].transform(lambda x: x.rolling(6).mean())
        df[f"{pol}_roll12"] = df.groupby("City")[pol].transform(lambda x: x.rolling(12).mean())

    df["AQI_target"] = df.groupby("City")["AQI"].shift(-1)

    df["AQI_lag1"] = df.groupby("City")["AQI"].shift(1)
    df["AQI_lag2"] = df.groupby("City")["AQI"].shift(2)

    df = df.dropna()

    return df

def fill_missing(df):

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    df[numeric_cols] = df.groupby("City")[numeric_cols].transform(
        lambda x: x.fillna(x.median())
    )

    return df


def encode_categorical(df):

    categorical_cols = df.select_dtypes(include=["object"]).columns

    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders

def clean_pollutant_data(df):

    POLLUTANT_LIMITS = {
        "PM2_5_ugm3": (0, 600),
        "PM10_ugm3": (0, 1000),
        "O3_ugm3": (0, 1000),
        "CO_ugm3": (0, 50000)
    }

    df = df.copy()

    for col, (low, high) in POLLUTANT_LIMITS.items():
        if col in df.columns:
            df[col] = df[col].clip(low, high)

    return df

def datetime_features(df):

    df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour"]])
    df = df.sort_values("datetime")

    return df


def drop_not_used(df):

    # To be updated based on feature importance and EDA insights

    drop_cols = [
        "Day_Name","AQI_Category","PM25_Category_India","EU_AQI",
        "EU_AQI_PM25","EU_AQI_PM10","US_AQI","US_AQI_PM25",
        "US_AQI_PM10","US_AQI_NO2","US_AQI_O3","US_AQI_CO",
        "Datetime","Day_of_Week","Week_of_Year",
        "Is_Weekend",
        "Quarter",
        "Season",
        "Time_of_Day","Humidity_Category",
        "Wind_Category","Wind_Stagnation", "Heavy_Rain",
        "Is_Daytime","PM_Ratio","Temp_Inversion", 
        'Solar_Radiation_Wm2', 'Direct_Radiation_Wm2', 
        'Diffuse_Radiation_Wm2', 
        'Cloud_Cover_Percent', 
        'Cloud_Low_Percent', 
        'Cloud_Mid_Percent', 'Cloud_High_Percent', 'Sunshine_Seconds',
        'Precipitation_mm', 
        'Rain_mm', 
        'Surface_Pressure_hPa', 
        'Dew_Point_C', 'Wind_Dir_10m', 'Wind_Gusts_kmh', 
        'AOD'
    ]

    drop_cols = [c for c in drop_cols if c in df.columns]

    df = df.drop(columns=drop_cols)

    return df

def calc_AQI(df):
    aqi_calc = IndiaAQICalculator()

    df["AQI"] = [
        aqi_calc.calculate_aqi(row)
        for row in df.itertuples(index=False)
    ]

    df = df[df["AQI"].notna()]

    return df

def process_data(df):

    df = clean_dataset(df)
    df = drop_not_used(df)
    df = clean_pollutant_data(df)
    df = fill_missing(df)
    df = datetime_features(df)
    df = feature(df)

    df = calc_AQI(df)

    df = shifing(df)
    
    df, label_encoders = encode_categorical(df)

    return df, label_encoders