# SmogCast PK: Hourly Smog Nowcasting for Pakistan

Machine-learning system that forecasts **hourly PM2.5** air pollution **1 to 24 hours ahead** for **Lahore, Karachi, Islamabad and Peshawar**. It also gives an 80% uncertainty interval, the chance of a smog episode, and profile-based health advice through a 3-page web dashboard.

**Live demo:**

---

## 1. Problem and objectives
Smog is a serious health problem in Pakistan, and people usually only see the current air-quality reading. This project predicts what the air will be like in the next hours so that people can plan.

- Forecast PM2.5 for horizons of 1, 3, 6, 9, 12, 18 and 24 hours.
- Give an 80% prediction interval, not just a single number.
- Predict whether PM2.5 will exceed 55.4 µg/m³ (Unhealthy level).
- Show everything in an easy dashboard for the general public.

## 2. Dataset
- **Source:** [Open-Meteo](https://open-meteo.com) Air Quality API (CAMS model data: PM2.5, PM10) and Historical Weather API (temperature, humidity, precipitation, wind speed).
- **Size:** 134,496 hourly rows, 4 cities, January 2023 to 1 November 2026.
- **File:** `ml/data/pakistan_air_quality_hourly.csv`
- **Note:** PM2.5 comes from a model (CAMS), not from ground sensors. Open-Meteo's free API is for non-commercial use.

## 3. Methodology
| Step | What was done |
|---|---|
| Inspection and cleaning | Checked each city is strictly hourly with no gaps, no missing values; dropped the first 168 hours per city that lack lag history |
| Feature engineering | 39 past-only features: lags up to 1 week, rolling mean/std/max/min, trends, PM10, weather, hour/month (sin/cos), day of week, city |
| Forecast-weather features | 5 features with the weather at the target hour (in production this comes from a weather-forecast API) |
| Model | One scikit-learn `HistGradientBoostingRegressor` per horizon, trained on log(1+PM2.5) |
| Uncertainty | Quantile models plus conformal calibration (CQR) for the 80% interval |
| Episode alert | `HistGradientBoostingClassifier` for PM2.5 > 55.4 |
| Robustness | Forecast-weather inputs are noise-augmented in training (0.4 std) because real forecasts are imperfect |

## 4. Evaluation (no data leakage)
- **Rolling-origin time-series cross-validation** on 5 held-out months: Nov 2025, Dec 2025, Jan 2026 (winter smog), Jul 2026 (monsoon) and Oct 2026.
- A 24-hour purge gap separates training and test data.
- **4 baselines:** persistence, seasonal-naive, 24h mean, climatology.
- **Also:** ablation study, day-block bootstrap confidence intervals, weather-noise sensitivity test, permutation importance.

**Main results (MAE in µg/m³, pooled over the 5 folds, 24h ahead):**

| Method | MAE |
|---|---|
| Persistence baseline | 14.70 |
| Model A (past data only) | 13.68 |
| Model B (+ forecast weather, perfect weather) | **11.48** |
| Model B (deployed, with 0.4 std weather noise) | 12.41 |

- Model B is about 22% better than persistence at 24h (95% bootstrap CI of the gain: 2.66 to 3.82 µg/m³).
- Episode detection F1: 0.856 (6h), 0.829 (12h), 0.819 (24h), compared with 0.618, 0.549 and 0.795 for the persistence rule.

## 5. Limitations
- PM2.5 is CAMS model data, not ground-sensor data.
- Using only past data, the 24h model beats persistence by only about 7%; most of the gain needs weather forecasts.
- Training used observed weather in place of forecasts, and the size of real forecast error was assumed, not measured.
- The nominal "80%" intervals covered 70% to 87% of outcomes across folds; they under-cover in the winter months.
- The dashboard is currently a **demo replay**: the forecast is issued at 31 Oct 2026 23:00 and compared with what actually happened.

## 6. Project structure
```
.
├── site/                    # Web dashboard (static HTML/CSS/JS), deployed on Vercel
│   ├── index.html           # Page 1: Forecast
│   ├── evaluation.html      # Page 2: Evaluation
│   ├── method.html          # Page 3: Method
│   ├── style.css, app.js
│   ├── data/                # forecast.json, metrics.json
│   └── assets/              # evaluation charts
└── ml/
    ├── features.py          # shared feature engineering
    ├── train_v3.py          # training and evaluation pipeline
    ├── build_site_data.py   # builds the dashboard's JSON files from the models
    ├── models_v3/           # trained models
    ├── data/                # dataset
    ├── metrics_v3.json      # all evaluation results
    └── backtest_predictions.csv
```

## 7. Tools
Python, pandas, NumPy, scikit-learn, Matplotlib, joblib; HTML, CSS, JavaScript; GitHub; Vercel.

## 8. Setup and how to run

**Run the dashboard locally** (needs only Python):
```bash
cd site
python -m http.server 8000
```
Open http://localhost:8000 in your browser. Do not double-click `index.html`, because the browser blocks loading the JSON data from a file.

**Retrain the models (optional, about 15 to 20 minutes):**
```bash
cd ml
pip install pandas numpy scikit-learn matplotlib joblib
python train_v3.py
SITE=../site python build_site_data.py  
```

## 9. Deployment
The `site/` folder is a static website deployed on Vercel: import the GitHub repository, set **Root Directory** to `site`, Framework Preset to **Other**, and deploy.
