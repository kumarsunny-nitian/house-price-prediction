import streamlit as st
import pickle
import numpy as np

#  Load Saved Files

with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Create Page Title

st.title("🏠 House Price Prediction App")

st.write(
    "Enter house details and predict the price."
)


#  Create Input Fields

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=20,
    value=3
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1.0,
    max_value=10.0,
    value=2.0
)

sqft_living = st.number_input(
    "Living Area (sqft)",
    value=1800
)

sqft_lot = st.number_input(
    "Lot Size (sqft)",
    value=5000
)

floors = st.number_input(
    "Floors",
    value=1
)

# Create Remaining Inputs

waterfront = st.number_input("Waterfront", value=0)
view = st.number_input("View", value=0)
condition = st.number_input("Condition", value=3)
grade = st.number_input("Grade", value=7)

sqft_above = st.number_input(
    "Sqft Above",
    value=1800
)

sqft_basement = st.number_input(
    "Sqft Basement",
    value=0
)

yr_built = st.number_input(
    "Year Built",
    value=2005
)

yr_renovated = st.number_input(
    "Year Renovated",
    value=0
)

zipcode = st.number_input(
    "Zipcode",
    value=98001
)

lat = st.number_input(
    "Latitude",
    value=47.5
)

long = st.number_input(
    "Longitude",
    value=-122.2
)

sqft_living15 = st.number_input(
    "Sqft Living15",
    value=1700
)

sqft_lot15 = st.number_input(
    "Sqft Lot15",
    value=4800
)





# Create Predict Button

if st.button("Predict Price"):

    features = [[
        bedrooms,
        bathrooms,
        sqft_living,
        sqft_lot,
        floors,
        waterfront,
        view,
        condition,
        grade,
        sqft_above,
        sqft_basement,
        yr_built,
        yr_renovated,
        zipcode,
        lat,
        long,
        sqft_living15,
        sqft_lot15
    ]]

    # Scale Input
    scaled_features = scaler.transform(
        features
    )

    # Predict
    prediction = model.predict(
        scaled_features
    )

    # Display result
    st.success(
        f"Predicted House Price: ${prediction[0]:,.2f}"
    )





