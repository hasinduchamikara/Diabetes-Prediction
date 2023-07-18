import streamlit as st 
import pickle
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

def load_model():
    
    with open('lr_classifier.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

lr = data["model"]
race = data["race"]
gender = data["gender"]
metformin = data["metformin"]
insulin = data["insulin"]

def show_lr_page():
    # st.image("logo.png", width=435)
    st.title("Logistic Regression")
    st.subheader("First we need some medical details about you.")

    race_encoder = data["race"]
    gender_encoder = data["gender"]
    metformin_encoder = data["metformin"]
    insulin_encoder = data["insulin"]

    race = ('Caucasian','AfricanAmerican','Asian','Hispanic','Other')
    gender = ('Female','Male')
    metformin = ('No','Steady','Up','Down')
    insulin = ('No','Up','Steady','Down')


    race = st.selectbox("Select your race ",race, key="race_Option")
    age = st.slider("age",0,100,35, key="age_Option")
    gender = st.selectbox("Select your gender ",gender, key="gender_Option")
    metformin = st.selectbox("Select the relevant check box for metformin",metformin, key="metformin_Option")
    insulin = st.selectbox("Select the relevant check box insulin",insulin, key="insulin_Option")
    
    

    ##############################################################################################
    ok = st.button("Predict Diabetes")
    if ok:
        X = np.array([[race, gender, age, metformin, insulin]])
        race_encoded = race_encoder.transform(X[:, 0])
        X[:, 0] = race_encoded
        gender_encoded = gender_encoder.transform(X[:, 1])
        X[:, 1] = gender_encoded
        metformin_encoded = metformin_encoder.transform(X[:, 3])
        X[:, 3] = metformin_encoded
        insulin_encoded = insulin_encoder.transform(X[:, 4])
        X[:, 4] = insulin_encoded

        X = X.astype(float)
     ###############################################################################################

        data_check = lr.predict(X)
        st.subheader(f"The Probability of having Diabetes is {data_check[0]}")