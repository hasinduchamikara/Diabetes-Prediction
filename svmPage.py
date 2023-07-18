import streamlit as st 
import pickle
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

def load_model():
    
    with open('svm_classifier.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

classifier = data["model"]
race = data["race"]
gender = data["gender"]
metformin = data["metformin"]
insulin = data["insulin"]

def show_predict_page():
    # st.image("logo.png", width=435)
    st.title("Support Vector Machines")
    st.subheader("First we need some medical details about you.")

    race_encoder = data["race"]
    gender_encoder = data["gender"]
    metformin_encoder = data["metformin"]
    insulin_encoder = data["insulin"]

    race = ('Caucasian','AfricanAmerican','Asian','Hispanic','Other')
    #age = ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100])
    gender = ('Female','Male')
    metformin = ('No','Steady','Up','Down')
    insulin = ('No','Up','Steady','Down')


    race = st.selectbox("Select your race ",race)
    #age = st.selectbox("Select your age ",age)
    age = st.slider("age",0,100,35)
    gender = st.selectbox("Select your gender ",gender)
    metformin = st.selectbox("Select the relevant check box for metformin",metformin)
    insulin = st.selectbox("Select the relevant check box insulin",insulin)
    
    

    ##############################################################################################
    ok = st.button("Predict Diabetes")
    if ok:
        # X = np.array([[race,gender,age,metformin,insulin]])
        # X[:,0] = race.transform(X[:,0]) 
        # X[:,1] = gender.transform(X[:,1])
        # X[:,3] = metformin.transform(X[:,3]) 
        # X[:,4] = insulin.transform(X[:,4])

        # X = X.astype(float)

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

        diabetes_check = classifier.predict(X)
        st.subheader(f"The Probability of having Diabetes is {diabetes_check[0]}")  