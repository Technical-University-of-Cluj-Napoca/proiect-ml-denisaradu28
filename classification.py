import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/classification_model.pkl")


def show_classification_page():
    st.title("F1 Race Strategy Classification")
    st.subheader("Predicting Pit Stop Next Lap")

    st.markdown("""
    Această pagină prezice dacă un pilot va intra la boxe în turul următor.
    Modelul folosește date despre tur, poziție, pneuri și progresul cursei.
    """)

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        lap_number = st.number_input("LapNumber", min_value=1, max_value=100, value=67)
        position = st.number_input("Position", min_value=1, max_value=20, value=6)
        stint = st.number_input("Stint", min_value=1, max_value=10, value=2)
        tyre_life = st.number_input("TyreLife", min_value=0.0, max_value=80.0, value=41.0)

    with col2:
        normalized_tyre_life = st.slider("Normalized_TyreLife", 0.0, 1.0, 0.80)

        compound_name = st.selectbox(
            "Compound",
            ["HARD", "INTERMEDIATE", "MEDIUM", "SOFT", "WET"]
        )

        compound_map = {
            "HARD": 0,
            "INTERMEDIATE": 1,
            "MEDIUM": 2,
            "SOFT": 3,
            "WET": 4
        }

        compound = compound_map[compound_name]

        lap_time = st.number_input("LapTime (s)", min_value=40.0, max_value=200.0, value=81.197)
        cumulative_degradation = st.number_input("Cumulative_Degradation", value=-32.86)

    with col3:
        lap_time_delta = st.number_input("LapTime_Delta", value=5.946)
        position_change = st.number_input("Position_Change", value=-4.0)
        race_progress = st.slider("RaceProgress", 0.0, 1.0, 0.86)

    X = pd.DataFrame([{
        "LapNumber": lap_number,
        "Position": position,
        "Stint": stint,
        "TyreLife": tyre_life,
        "Normalized_TyreLife": normalized_tyre_life,
        "Compound": compound,
        "LapTime (s)": lap_time,
        "Cumulative_Degradation": cumulative_degradation,
        "LapTime_Delta": lap_time_delta,
        "Position_Change": position_change,
        "RaceProgress": race_progress
    }])

    st.write("Input data:")
    st.dataframe(X)

    if st.button("Predict Pit Stop"):
        pred = model.predict(X)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[0][1]
            st.metric("Pit Stop Probability", f"{prob * 100:.2f}%")

        if pred == 1:
            st.success("Prediction: Pit Stop Next Lap")
        else:
            st.info("Prediction: No Pit Stop Next Lap")

    st.divider()