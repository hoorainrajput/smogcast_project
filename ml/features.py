# Shared feature engineering for training AND the live app, so that the
# two can never drift apart.
#
# Leakage rules
#   * Past features use time t and earlier only.
#   * "fx_*" features are the weather at the TARGET time t+h. In
#     production they come from a weather-forecast API; in training the
#     archived (observed) weather is used as a stand-in (the standard
#     "perfect prognosis" setup). train_v3.py measures how much forecast
#     error in those columns hurts (noise-sensitivity experiment).
#   * No future PM2.5 or PM10 value is ever used as a feature.
import numpy as np
import pandas as pd

CITIES = ["Lahore", "Karachi", "Islamabad", "Peshawar"]
WEATHER = ["temperature_2m", "relative_humidity_2m", "precipitation",
           "wind_speed_10m"]
UNHEALTHY = 55.4          # PM2.5 (ug/m3) threshold used for "smog episode"
MIN_HISTORY = 169         # hours of history needed before a row is usable

# ---- feature groups (used for the ablation study) ----
G_LAG = (["pm2_5", "lag1", "lag2", "lag3", "lag6", "lag12", "lag24", "lag48",
          "lag168", "hs", "hc", "ms", "mc", "dow"]
         + [f"c_{c}" for c in CITIES])
G_ROLL = ["rm6", "rm24", "rm72", "rs24", "rmax24", "rmin24", "d1", "d6", "d24"]
G_PM10 = ["pm10", "pm10l6", "ratio"]
G_WX = WEATHER + ["temperature_2m_d6", "relative_humidity_2m_d6",
                  "wind_speed_10m_d6", "rain24", "wind_rm24"]
PAST_COLS = G_LAG + G_ROLL + G_PM10 + G_WX
FX_COLS = [f"fx_{c}" for c in WEATHER] + ["fx_wind_rm6"]


def build_past_features(df):
    """df: columns time, city, pm2_5, pm10 + WEATHER. Hourly, per city.
    Rows whose PM2.5 is NaN (e.g. future rows holding forecast weather)
    are kept; they simply get NaN features."""
    df = df.sort_values(["city", "time"]).reset_index(drop=True).copy()
    gaps = df.groupby("city")["time"].diff().dropna()
    if (gaps != pd.Timedelta(hours=1)).any():
        raise ValueError("Data must be strictly hourly per city, no gaps")
    g = df.groupby("city")
    hour, month = df["time"].dt.hour, df["time"].dt.month
    df["hour"], df["month"] = hour, month
    df["hs"] = np.sin(2 * np.pi * hour / 24)
    df["hc"] = np.cos(2 * np.pi * hour / 24)
    df["ms"] = np.sin(2 * np.pi * month / 12)
    df["mc"] = np.cos(2 * np.pi * month / 12)
    df["dow"] = df["time"].dt.dayofweek
    for lag in (1, 2, 3, 6, 12, 24, 48, 168):
        df[f"lag{lag}"] = g["pm2_5"].shift(lag)
    for w in (6, 24, 72):
        df[f"rm{w}"] = g["pm2_5"].transform(lambda s: s.rolling(w).mean())
    df["rs24"] = g["pm2_5"].transform(lambda s: s.rolling(24).std())
    df["rmax24"] = g["pm2_5"].transform(lambda s: s.rolling(24).max())
    df["rmin24"] = g["pm2_5"].transform(lambda s: s.rolling(24).min())
    df["d1"] = df["pm2_5"] - df["lag1"]
    df["d6"] = df["pm2_5"] - df["lag6"]
    df["d24"] = df["pm2_5"] - df["lag24"]
    df["ratio"] = df["pm2_5"] / (df["pm10"] + 1)
    df["pm10l6"] = g["pm10"].shift(6)
    for c in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m"):
        df[f"{c}_d6"] = df[c] - g[c].shift(6)
    df["rain24"] = g["precipitation"].transform(lambda s: s.rolling(24).sum())
    df["wind_rm24"] = g["wind_speed_10m"].transform(
        lambda s: s.rolling(24).mean())
    for c in CITIES:
        df[f"c_{c}"] = (df["city"] == c).astype(float)
    return df


def future_weather(df, h):
    """Weather at target time t+h for every row (df from
    build_past_features, which may also contain future weather rows)."""
    g = df.groupby("city")
    out = pd.DataFrame(index=df.index)
    for c in WEATHER:
        out[f"fx_{c}"] = g[c].shift(-h)
    out["fx_wind_rm6"] = g["wind_speed_10m"].transform(
        lambda s: s.shift(-h).rolling(6).mean())
    return out[FX_COLS]


def aqi_band(v):
    if v <= 12:
        return "Good"
    if v <= 35.4:
        return "Moderate"
    if v <= 55.4:
        return "Unhealthy for Sensitive Groups"
    if v <= 150.4:
        return "Unhealthy"
    if v <= 250.4:
        return "Very Unhealthy"
    return "Hazardous"
