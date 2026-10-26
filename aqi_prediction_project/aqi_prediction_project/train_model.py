from pathlib import Path
import joblib
from src.data_utils import load_dataset, make_demo_dataset, REQUIRED_COLUMNS
from src.model_utils import train_model

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "aqi_environmental_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

if DATA.exists():
    df = load_dataset(DATA)
    source = "provided CSV"
else:
    df = make_demo_dataset()
    source = "synthetic demo data"

model, metrics = train_model(df)
joblib.dump(model, MODEL_DIR / "aqi_model.joblib")
print(f"Trained using {source}; rows={len(df)}")
print("Metrics:", metrics)
print(f"Saved model to {MODEL_DIR / 'aqi_model.joblib'}")
