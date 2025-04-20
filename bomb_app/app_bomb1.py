import streamlit as st
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk

# Load trained model and encoders
model = joblib.load("bomb_best_xgb_model.joblib")
target_encoder = joblib.load("bomb_target_encoder.joblib")
cat_encoders = joblib.load("bomb_label_encoders.joblib")

# Define feature lists
numeric_features = [
    "Latitude",
    "Longitude",
    "Elevation(m)",
    "Terrain_Ruggedness_Index",
    "Urbanization_Level(%)",
    "Vegetation_Density(%)",
    "Proximity_Civilian_Infrastructure(km)",
    "Area_Size_Target(km2)",
    "Enemy_Concentration(units_per_km2)",
    "Visibility_Level"
]

categorical_features = [
    "Soil_Stability",
    "Target_Type",
    "Fortification_Level",
    "Urgency_of_Mission"
]

st.set_page_config(page_title="Bombing Type Prediction App", layout="wide")
st.title("💥 Bombing Type Prediction Dashboard")
st.markdown("Predict the optimal bombing munition type for a given location and conditions.")

# User name input
default_name = st.session_state.get("username", "")
with st.expander("👤 Set or change your name", expanded=True):
    username = st.text_input("Enter your name (optional)", value=default_name, key="username")

# CSV upload for sample data
df_sample = None
row_data = None
uploaded_file = st.file_uploader("📄 Upload CSV with sample rows", type=["csv"])
if uploaded_file:
    df_uploaded = pd.read_csv(uploaded_file)
    df_sample = df_uploaded[numeric_features + categorical_features]
    st.markdown("### 📊 Uploaded Data Preview")
    st.dataframe(df_sample)
    idx = st.selectbox("🔍 Select row to auto-fill inputs:", df_sample.index.tolist(), key="row_selector")
    row_data = df_sample.loc[idx]

# Input widgets
st.markdown("### 🧠 Input Features")
raw_inputs = {}
col1, col2 = st.columns(2)

# Numeric inputs
for i, feat in enumerate(numeric_features):
    default_val = float(row_data[feat]) if row_data is not None else 0.0
    with (col1 if i % 2 == 0 else col2):
        raw_inputs[feat] = st.number_input(feat.replace("_", " "), value=default_val, key=feat)

# Categorical inputs
for j, feat in enumerate(categorical_features):
    options = cat_encoders[feat].classes_.tolist()
    default_cat = row_data[feat] if row_data is not None else options[0]
    with (col1 if (len(numeric_features) + j) % 2 == 0 else col2):
        raw_inputs[feat] = st.selectbox(feat.replace("_", " "), options, index=options.index(default_cat), key=feat)

# Prediction button
if st.button("📍 Locate on Map and Predict"):
    st.session_state['run_prediction'] = True
else:
    st.session_state['run_prediction'] = False

# Initialize
prediction_proba = None
pred_label = ""
marker_color = [100, 100, 255, 160]
tooltip = ""

# Run prediction
if st.session_state.get('run_prediction', False):
    st.session_state['run_prediction'] = False
    # Encode inputs
    encoded_list = []
    for feat in numeric_features:
        encoded_list.append(raw_inputs[feat])
    for feat in categorical_features:
        encoded_list.append(cat_encoders[feat].transform([raw_inputs[feat]])[0])
    input_array = np.array([encoded_list])
    pred_code = model.predict(input_array)[0]
    pred_label = target_encoder.inverse_transform([pred_code])[0]
    prediction_proba = model.predict_proba(input_array)[0]
    # Color map
    color_map = {
        "Airburst Munitions": [255, 69, 0, 200],
        "Bunker-Busters": [139, 0, 139, 200],
        "Cluster Bombs": [255, 215, 0, 200],
        "Drone-Based Tactical Strikes": [30, 144, 255, 200],
        "Precision-Guided Munitions": [34, 139, 34, 200],
    }
    marker_color = color_map.get(pred_label, marker_color)
    tooltip = f"Prediction: {pred_label}"

# Map display
if pred_label:
    loc_df = pd.DataFrame([{"lat": raw_inputs['Latitude'], "lon": raw_inputs['Longitude'], "tooltip": tooltip}])
    view = pdk.ViewState(latitude=raw_inputs['Latitude'], longitude=raw_inputs['Longitude'], zoom=12, pitch=45)
    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=loc_df,
        get_position='[lon, lat]',
        get_color=marker_color,
        get_radius=500,
        pickable=True
    )
    text = pdk.Layer(
        "TextLayer",
        data=loc_df,
        get_position='[lon, lat]',
        get_text='tooltip',
        get_size=20,
        get_color='[255, 255, 255]',
        get_alignment_baseline='bottom'
    )
    st.markdown("---")
    st.markdown("### 🗺️ Location Map")
    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/satellite-v9",
            initial_view_state=view,
            layers=[scatter, text],
            tooltip={"text": "{tooltip}"}
        )
    )
    # Legend under map
    st.markdown("### 📖 Legend")
    legend_items = {
        "Airburst Munitions": [255, 69, 0],
        "Bunker-Busters": [139, 0, 139],
        "Cluster Bombs": [255, 215, 0],
        "Drone-Based Tactical Strikes": [30, 144, 255],
        "Precision-Guided Munitions": [34, 139, 34],
    }
    legend_html = ""
    for label, color in legend_items.items():
        hex_color = '#%02x%02x%02x' % tuple(color)
        legend_html += f"<span style='display:inline-block;width:15px;height:15px;background-color:{hex_color};margin-right:5px;'></span> {label} &nbsp;&nbsp;"
    st.markdown(legend_html, unsafe_allow_html=True)

# Results display
if prediction_proba is not None:
    st.markdown("---")
    with st.expander("📌 Prediction Results", expanded=True):
        st.success(f"✅ Predicted Bombing Type: {pred_label}")
        st.markdown("#### 🔢 Class Probabilities")
        proba_dict = {cls: f"{prob:.4f}" for cls, prob in zip(target_encoder.classes_, prediction_proba)}
        st.json(proba_dict)

# Log predictions
if "prediction_log" not in st.session_state:
    st.session_state.prediction_log = []
if prediction_proba is not None:
    from datetime import datetime
    entry = {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "User": username or "Anonymous"}
    entry.update(raw_inputs)
    entry["Predicted Type"] = pred_label
    for cls, prob in zip(target_encoder.classes_, prediction_proba):
        entry[cls] = prob
    st.session_state.prediction_log.append(entry)

# Display log
if st.session_state.prediction_log:
    log_df = pd.DataFrame(st.session_state.prediction_log)
    st.markdown("### 🗃️ All Predictions Log")
    st.dataframe(log_df, use_container_width=True)
    csv = log_df.to_csv(index=False).encode('utf-8')
    st.download_button("📦 Download All Predictions as CSV", data=csv, file_name="predictions_log.csv", mime="text/csv")

st.markdown("---")
