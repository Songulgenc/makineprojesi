import streamlit as st
import pickle
import numpy as np

# Modelleri ve scaler'ı yükleme
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

# Streamlit uygulaması
st.title('Wine Quality Prediction')
fixed_acidity = st.slider('Fixed Acidity', 4.0, 16.0)
volatile_acidity = st.slider('Volatile Acidity', 0.1, 1.5)
citric_acid = st.slider('Citric Acid', 0.0, 1.0)
residual_sugar = st.slider('Residual Sugar', 0.9, 15.5)
chlorides = st.slider('Chlorides', 0.012, 0.611)
free_sulfur_dioxide = st.slider('Free Sulfur Dioxide', 1.0, 72.0)
total_sulfur_dioxide = st.slider('Total Sulfur Dioxide', 6.0, 289.0)
density = st.slider('Density', 0.990, 1.004)
pH = st.slider('pH', 2.74, 4.01)
sulphates = st.slider('Sulphates', 0.33, 2.0)
alcohol = st.slider('Alcohol', 8.4, 14.9)

input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
                        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
input_data_scaled = scaler.transform(input_data)
input_data_pca = pca.transform(input_data_scaled)
prediction = model.predict(input_data_pca)

st.write('Predicted Wine Quality:', prediction[0])

