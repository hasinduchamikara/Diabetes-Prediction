import streamlit as st 
import pickle
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

def load_model():
    
    with open('rf_classifier.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

clf = data["model"]
race = data["race"]
gender = data["gender"]
metformin = data["metformin"]
insulin = data["insulin"]

def show_rf_page():
    # st.image("logo.png", width=435)
    st.title("Random Forest")
    st.subheader("First we need some medical details about you.")

    race_encoder = data["race"]
    gender_encoder = data["gender"]
    metformin_encoder = data["metformin"]
    insulin_encoder = data["insulin"]

    race = ('Caucasian','AfricanAmerican','Asian','Hispanic','Other')
    gender = ('Female','Male')
    metformin = ('No','Steady','Up','Down')
    insulin = ('No','Up','Steady','Down')


    race = st.selectbox("Select your race ",race, key="race_Selector")
    #age = st.selectbox("Select your age ",age)
    age = st.slider("age",0,100,35, key="age_Selector")
    gender = st.selectbox("Select your gender ",gender, key="gender_Selector")
    metformin = st.selectbox("Select the relevant check box for metformin",metformin, key="metformin_Selector")
    insulin = st.selectbox("Select the relevant check box insulin",insulin, key="insulin_Selector")
    
    

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

        db_check = clf.predict(X)
        st.subheader(f"The Probability of having Diabetes is {db_check[0]}")  