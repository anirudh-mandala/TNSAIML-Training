# Air Quality Index (AQI) Prediction Using Environmental Data

A beginner-friendly machine-learning dashboard built with Python, scikit-learn, pandas, Matplotlib, and Streamlit. It follows the uploaded **Streamlit Basics for ML Dashboards** guide: data exploration, summary metrics, charts, user-entered features, a trained model, prediction output, validation, and a clear limitations note.

## Features
- Dashboard overview with record/feature/missing-value counts
- Dataset preview, descriptive statistics, and charts
- CSV upload with schema and numeric validation
- AQI prediction from PM2.5, PM10, NO2, SO2, CO, and O3
- Random Forest regression model and MAE/RMSE/R² evaluation
- Model training script and organized project folders

## Project structure
```text
aqi_prediction_project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── aqi_environmental_data.csv
├── models/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── data_utils.py
    └── model_utils.py
```

## Important data note
The included CSV is **synthetic demonstration data**, generated for testing the complete software workflow. It is not real sensor data, not official AQI data, and must not be used to make health or policy decisions. For a credible academic project, replace it with a real, cited environmental dataset and document its location, measurement units, sampling interval, date range, missing-value handling, and AQI calculation/label source.

Expected CSV columns (exact spelling):
`PM2.5, PM10, NO2, SO2, CO, O3, AQI`

The `AQI` column is the numeric target used for regression. If your dataset uses different names or a different target definition, update `src/data_utils.py` and the model feature selection accordingly. CO units must match between training data and prediction form (the form currently labels CO in mg/m³).

## Setup on Windows
Open Command Prompt or the VS Code terminal inside this folder:

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your Python installation is unusually new and a dependency does not yet support it, use Python 3.11 or 3.12 in a fresh virtual environment.

## Run the dashboard
```bash
streamlit run app.py
```

## Retrain the model
```bash
python train_model.py
```

The dashboard currently trains a model in memory from the active dataset when it starts, which keeps the example self-contained and allows it to work with the included demo data. `train_model.py` is provided as a separate reproducible training step and saves a pipeline artifact to `models/aqi_model.joblib`.

## Suggested academic workflow
1. Find a reputable real-world air-quality dataset (for example, from a government environmental monitoring portal or a documented open-data repository).
2. Verify pollutant units and target AQI methodology.
3. Explore distributions, missing values, outliers, and feature correlations.
4. Train/test split; train a baseline and compare with Random Forest or another model.
5. Report MAE, RMSE, and R² on held-out data.
6. Discuss limitations, data leakage risks, and generalization across cities/seasons.
7. Deploy only after validating on realistic data.

## Limitations
- The synthetic dataset uses a constructed formula, so strong evaluation metrics do not establish real-world predictive accuracy.
- The category thresholds are simplified illustrative thresholds, not a claim of official regulatory classification.
- This is an educational demonstration, not an official monitoring service or health advisory.
