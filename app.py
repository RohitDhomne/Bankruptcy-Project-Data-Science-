# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:24:07 2023

@author: Atharva Gujar
"""

import pickle
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Load the model
load = open('rnd.pkl', 'rb')
model = pickle.load(load)

# Sample data for visualization
# Replace this with actual data or relevant data processing for visualization
time_points = np.arange(0, 10, 0.1)
data_points = np.sin(time_points)

# Visualization function
def visualize_line_chart(time_points, data_points):
    fig, ax = plt.subplots()
    ax.plot(time_points, data_points, label='Sample Data')
    ax.set_xlabel('Time')
    ax.set_ylabel('Data')
    ax.legend()
    return fig

# Prediction function
def predict(Industrial_Risk, Management_Risk, Financial_Flexibility, Credibility, Competitiveness, Operating_Risk, New_Class):
    prediction = model.predict([[Industrial_Risk, Management_Risk, Financial_Flexibility, Credibility, Competitiveness, Operating_Risk,]])
    return prediction[0]  # Use prediction[0] to get the actual prediction value

def main():
    st.title('Bankruptcy Prediction Project 🧬')
    st.markdown('This is a Random Forest machine learning model to predict Bankruptcy or Not... 🦾')

    Industrial_Risk = st.selectbox('Industrial_Risk:', [0, 0.5, 1], key='unique_key_6')
    Management_Risk = st.selectbox('Management_Risk:', [0, 0.5, 1], key='unique_key_1')
    Financial_Flexibility = st.selectbox('Financial_Flexibility:', [0, 0.5, 1], key='unique_key_2')
    Credibility = st.selectbox('Credibility:', [0, 0.5, 1], key='unique_key_3')
    Competitiveness = st.selectbox('Competitiveness:', [0, 0.5, 1], key='unique_key_4')
    Operating_Risk = st.selectbox('Operating_Risk:', [0, 0.5, 1], key='unique_key_5')

# Visualization
    st.subheader('Example Visualization: Line Chart')
    st.write('This is an example line chart. Replace it with your actual visualization.')

    # Display the line chart
    line_chart = visualize_line_chart(time_points, data_points)
    st.pyplot(line_chart)

    if st.button('Predict'):
        Result = predict(Industrial_Risk, Management_Risk, Financial_Flexibility, Credibility, Competitiveness, Operating_Risk, 0)
        if Result == 0:
            st.success("Bankruptcy")
        else:
            st.success("Non Bankruptcy")

if __name__ == '__main__':
    main()

