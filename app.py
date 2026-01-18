import streamlit as st

st.set_page_config(page_title="Ground water Level Prediction")

st.title("Ground water Level Prediction")

st.write("Enter the values below to predict groundwater level.")

# User inputs
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)
temperature = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
year = st.number_input("Year", min_value=1900, step=1)

# Button
if st.button("Predict"):
    st.success("Prediction button clicked successfully!")
