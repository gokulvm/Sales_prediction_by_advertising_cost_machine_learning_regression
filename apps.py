
"""
@author: Gokulraj
"""
import numpy as np
import pickle
import pandas as pd
import webbrowser
import streamlit as st

from PIL import Image

model=pickle.load(open('advertising_model.pkl', 'rb'))
transformer=pickle.load(open('rb.pkl', 'rb'))



def predict_message(tv,radio,news):

    values = transformer.transform([[tv,radio,news]])[:,:-1]
    prediction = model.predict(values)
    ans = prediction[0]
    ans = str(round(prediction[0],2)) +" " + "Dollars"
    return ans



def main():
    st.text("@author: Gokulraj.T")
    st.text("Machine Learning App Built with Streamlit")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Sales Prediction by Advertising Cost</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    tv = st.number_input('TV Advertisement Cost ', value=1.)
    radio = st.number_input('Radio Advertisement Cost ', value=1.)
    news = st.number_input('News Paper Advertisement Cost ', value=1.)
    result=""
    if st.button("Predict"):
        result=predict_message(tv,radio,news)
        
    
    st.success(result)
    if st.button("About"):
        st.text("This model can predict sales by advertising cost")
        st.text("Used Algorithm : Random Forest Regressor ")
        st.text("Accuracy : 97%")
        link = '[Code](https://github.com/gokulvm/Sales_prediction_by_advertising_cost_machine_learning_regression)'
        st.markdown(link, unsafe_allow_html=True)
       
if __name__=='__main__':
    main()
    
       
