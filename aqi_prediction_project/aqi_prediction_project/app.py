
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from src.data_utils import (
    load_dataset,
    make_demo_dataset,
    validate_columns,
    REQUIRED_FEATURES,
)
from src.model_utils import train_model, predict_aqi_category, CATEGORY_ORDER

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "aqi_environmental_data.csv"
MODEL_PATH = ROOT / "models" / "aqi_model.joblib"

st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 Air Quality Index (AQI) Prediction")
st.caption(
    "Environmental data exploration and educational machine-learning predictions"
)


@st.cache_data
def get_data():
    if DATA_PATH.exists():
        try:
            df = load_dataset(DATA_PATH)
            validate_columns(df)
            return df, "Project dataset"
        except Exception as exc:
            st.warning(
                f"Project dataset could not be loaded ({exc}). "
                "Using generated demo data."
            )

    return make_demo_dataset(), "Generated demo dataset"


@st.cache_resource
def get_model_and_metrics(data_fingerprint):
    df, _ = get_data()
    model, metrics = train_model(df)
    return model, metrics


df, dataset_label = get_data()

with st.sidebar:
    st.header("Dashboard controls")
    st.write(f"Data source: **{dataset_label}**")

    page = st.radio(
        "Go to",
        [
            "Overview",
            "Explore data",
            "Predict AQI",
            "Model evaluation",
            "About",
        ],
    )

    st.divider()
    st.caption("Predictions are estimates, not official monitoring readings.")

model, metrics = get_model_and_metrics((len(df), tuple(df.columns)))


if page == "Overview":
    st.subheader("Project overview")

    st.write(
        "This dashboard uses environmental measurements to estimate AQI "
        "and classify the result into an illustrative air-quality category."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Records", f"{len(df):,}")
    c2.metric("Features", len(REQUIRED_FEATURES))
    c3.metric("Missing values", int(df.isna().sum().sum()))
    c4.metric("Model R²", f"{metrics['r2']:.3f}")

    st.subheader("Environmental data preview")
    st.dataframe(df.head(10), use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.markdown("**AQI distribution**")

        fig, ax = plt.subplots()
        ax.hist(df["AQI"], bins=20)
        ax.set_xlabel("AQI")
        ax.set_ylabel("Record count")
        ax.set_title("Distribution of target AQI")

        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("**Correlation with AQI**")

        corr = (
            df[REQUIRED_FEATURES + ["AQI"]]
            .corr(numeric_only=True)["AQI"]
            .drop("AQI")
            .sort_values()
        )

        st.bar_chart(corr)


elif page == "Explore data":
    st.subheader("Explore environmental data")

    uploaded = st.file_uploader(
        "Upload a CSV (must contain the required columns)",
        type=["csv"],
    )

    if uploaded is not None:
        try:
            user_df = pd.read_csv(uploaded)
            validate_columns(user_df)

            st.success("CSV schema validated.")
            st.dataframe(user_df.head(30), use_container_width=True)

            st.write("Missing values by column")
            st.dataframe(user_df.isna().sum().rename("missing_count"))

            st.write("Numeric summary")
            st.dataframe(user_df.describe(), use_container_width=True)

        except Exception as exc:
            st.error(f"Could not use the uploaded file: {exc}")

    else:
        st.dataframe(df, use_container_width=True)

        st.write("Numeric summary")
        st.dataframe(df.describe(), use_container_width=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    selected = st.selectbox(
        "Choose a numeric column to chart",
        numeric_cols,
        index=numeric_cols.index("AQI"),
    )

    st.line_chart(df[selected])


elif page == "Predict AQI":
    st.subheader("Enter environmental measurements")

    st.info(
        "Enter measurements in the same units as your training data. "
        "Default values are illustrative demo values."
    )

    with st.form("aqi_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            pm25 = st.number_input(
                "PM2.5 (µg/m³)",
                min_value=0.0,
                max_value=1000.0,
                value=35.0,
                step=1.0,
            )

            pm10 = st.number_input(
                "PM10 (µg/m³)",
                min_value=0.0,
                max_value=1500.0,
                value=70.0,
                step=1.0,
            )

        with col2:
            no2 = st.number_input(
                "NO₂ (µg/m³)",
                min_value=0.0,
                max_value=1000.0,
                value=25.0,
                step=1.0,
            )

            so2 = st.number_input(
                "SO₂ (µg/m³)",
                min_value=0.0,
                max_value=1000.0,
                value=8.0,
                step=1.0,
            )

        with col3:
            co = st.number_input(
                "CO (mg/m³)",
                min_value=0.0,
                max_value=100.0,
                value=0.8,
                step=0.1,
            )

            o3 = st.number_input(
                "O₃ (µg/m³)",
                min_value=0.0,
                max_value=1000.0,
                value=30.0,
                step=1.0,
            )

        submitted = st.form_submit_button("Predict AQI")

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "PM2.5": pm25,
                    "PM10": pm10,
                    "NO2": no2,
                    "SO2": so2,
                    "CO": co,
                    "O3": o3,
                }
            ]
        )

        pred = float(model.predict(input_df)[0])
        category = predict_aqi_category(pred)

        st.success("Prediction completed")

        a, b = st.columns(2)
        a.metric("Estimated AQI", f"{pred:.1f}")
        b.metric("Illustrative category", category)

        st.caption(
            "This is a model estimate from the supplied values, not an "
            "official government AQI reading or health advice."
        )


elif page == "Model evaluation":
    st.subheader("Model evaluation")

    a, b, c = st.columns(3)

    a.metric("MAE", f"{metrics['mae']:.2f}")
    b.metric("RMSE", f"{metrics['rmse']:.2f}")
    c.metric("R²", f"{metrics['r2']:.3f}")

    st.write(
        "Metrics are calculated on a held-out test split of the dataset "
        "currently used by the app."
    )

    st.write("Feature importance (Random Forest)")

    # FIX: feature_importances_ belongs to the Random Forest
    # estimator inside the scikit-learn Pipeline.
    importance_values = model.named_steps["regressor"].feature_importances_

    importance = pd.Series(
        importance_values,
        index=REQUIRED_FEATURES,
    ).sort_values()

    st.bar_chart(importance)

    st.write(
        "Target AQI is a numeric regression target. Category labels are "
        "simplified for demonstration and may not match the official AQI "
        "standard used in your region."
    )


else:
    st.subheader("About this project")

    st.markdown(
        """
        **Objective:** Predict AQI from environmental pollutant measurements
        using a machine-learning regression model.

        **Input features:** PM2.5, PM10, NO₂, SO₂, CO, and O₃.

        **Algorithm:** Random Forest Regression.

        **Workflow:** Data loading → validation → preprocessing →
        model training → evaluation → Streamlit dashboard.

        **Important limitation:** The included generated demo data is
        synthetic. The reported results are only a software demonstration.
        For meaningful academic evaluation, replace it with a reliable
        real-world dataset and document its source, units, date range,
        location, and AQI calculation method.
        """
    )

    st.write("Project structure and setup instructions are provided in README.md.")
