import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load your trained model
model = joblib.load('titanic_model.pkl')  # Replace with your model's path

# Title of the application
st.title("Titanic Survival Prediction")

# Instructions
st.write("Fill in the details below to predict if a passenger would survive or not:")

# Input fields for user data
Pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2)
Age = st.number_input("Age (years)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
Sex = st.radio("Gender", ['male', 'female'])
SibSp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0, step=1)
Parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0, step=1)
Fare = st.number_input("Fare Paid", min_value=0.0, max_value=500.0, value=50.0, step=1.0)

# Convert Gender to numerical value
Sex = 1 if Sex == 'male' else 0

# Button to trigger prediction
if st.button("Predict Survival"):
    # Create a DataFrame with input features
    feature_names = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex']
    input_features = np.array([[Pclass, Age, SibSp, Parch, Fare, Sex]])

    # Make prediction
    prediction = model.predict(input_features)[0]  # Binary prediction (0 or 1)

    # Display the result
    if prediction == 1:
        st.success("The passenger would survive!")
    else:
        st.error("The passenger would not survive.")
