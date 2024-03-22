import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# app heading
st.write("""
# Wine Quality Prediction 
This app predicts the **Wine Quality** type!
""")

# creating sidebar for user input features
st.sidebar.header('Input Parameters')

def user_input_features():
    fixed_acidity = st.sidebar.slider('Fixed Acidity', 4.0, 16.0, 8.0, step=0.1)
    volatile_acidity = st.sidebar.slider('Volatile Acidity', 0.0, 2.0, 0.5, step=0.01)
    citric_acid = st.sidebar.slider('Citric Acid', 0.0, 1.0, 0.5, step=0.01)
    residual_sugar = st.sidebar.slider('Residual Sugar', 0.0, 20.0, 2.5, step=0.1)
    chlorides = st.sidebar.slider('Chlorides', 0.0, 1.0, 0.08, step=0.01)
    free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide', 0.0, 100.0, 15.0, step=1.0)
    total_sulfur_dioxide = st.sidebar.slider('Total Sulfur Dioxide', 0.0, 300.0, 46.0, step=1.0)
    density = st.sidebar.slider('Density', 0.98, 1.06, 0.995, step=0.001)
    pH = st.sidebar.slider('pH', 2.0, 4.5, 3.3, step=0.1)
    sulphates = st.sidebar.slider('Sulphates', 0.0, 2.5, 0.65, step=0.01)
    alcohol = st.sidebar.slider('Alcohol', 8.0, 15.0, 10.4, step=0.1)

    data = {'fixed acidity': fixed_acidity,
            'volatile acidity': volatile_acidity,
            'citric acid': citric_acid,
            'residual sugar': residual_sugar,
            'chlorides': chlorides,
            'free sulfur dioxide': free_sulfur_dioxide,
            'total sulfur dioxide': total_sulfur_dioxide,
            'density': density,
            'pH': pH,
            'sulphates': sulphates,
            'alcohol': alcohol}

    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()
st.subheader('Input parameters')
st.write(df)

# reading csv file
with open("winequality-red.csv", "r") as f:
    data = [line.strip().split(';') for line in f.readlines()]

column_names = data[0]
data = data[1:]
data = pd.DataFrame(data, columns=column_names)
data.columns = [col.replace('"', '') for col in data.columns]  # Remove double quotes from column names
data = data.astype(float)

# create X and Y arrays
X = np.array(data.iloc[:, :-1].values)  # select all rows and all columns except the last one
Y = np.array(data.iloc[:, -1].values)  # select all rows and the last column (target variable)

# random forest model
wine_quality_rfr = RandomForestRegressor()
wine_quality_rfr.fit(X, Y)

prediction = wine_quality_rfr.predict(df)

st.text("")  # Add an empty space to align the header to the right
st.subheader('Wine Quality')
st.subheader(str(np.round(prediction[0], 2)))
