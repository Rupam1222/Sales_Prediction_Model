import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="📊",
    layout="centered"
)

st.title("📈 Sales Prediction App")
st.write("Predict sales based on product and outlet details")


# Load Model & Preprocessor

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

model, preprocessor = load_model()


# User Input Form

st.subheader("🔢 Enter Input Features")

item_weight = st.number_input("Item Weight", min_value=0.0, step=0.1)
item_fat_content = st.selectbox("Item Fat Content", ["Low_Fat", "Regular"])
item_visibility = st.number_input("Item Visibility", min_value=0.0, step=0.001)
item_type = st.selectbox(
    "Item Type",
    [
        "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables",
        "Household", "Baking Goods", "Snack Foods", "Frozen Foods",
        "Breakfast", "Health and Hygiene", "Hard Drinks",
        "Canned", "Breads", "Starchy Foods", "Others", "Seafood"
    ]
)

item_mrp = st.number_input("Item MRP", min_value=0.0, step=1.0)
fat_content_map = {
    "Low Fat": "Low_Fat",
    "Regular": "Regular"
}

outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
outlet_location_type = st.selectbox(
    "Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"]
)
outlet_type = st.selectbox(
    "Outlet Type",
    ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"]
)
outlet_year = st.number_input(
    "Outlet Establishment Year",
    min_value=1985,
    max_value=2025,
    value=2005
)

outlet_age = 2025 - outlet_year


# Prediction

if st.button(" Predict Sales"):

    input_data = pd.DataFrame({
        "Weight": [item_weight],
        "FatContent": [item_fat_content],
        "ProductType": [item_type],
        "MRP": [item_mrp],
        "OutletSize": [outlet_size],
        "LocationType": [outlet_location_type],
        "OutletType": [outlet_type],
        "OutletAge": [outlet_age]
    })

    try:
        input_transformed = preprocessor.transform(input_data)

        prediction_log = model.predict(input_transformed)
        prediction = np.expm1(prediction_log)

        st.success(f"Predicted Sales: ₹ {prediction[0]:,.2f}")

    except Exception as e:
        st.error("❌ Error during prediction")
        st.exception(e)



# Footer

st.markdown("---")
st.caption("Made by Rupam Vishwakarma 🚀")
