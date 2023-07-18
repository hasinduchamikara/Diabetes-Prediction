import streamlit as st
from svmPage import show_predict_page
from rfPage import show_rf_page
from gbPage import show_gb_page
from lrPage import show_lr_page

# show_predict_page()

st.title("Diabetic Predictor")

pages = {
    "Support Vector Machine" : show_predict_page,
    "Random Forest" : show_rf_page,
    "Gradient Boosting" : show_gb_page,
    "Logistic Regression" : show_lr_page
}

selection = st.sidebar.multiselect("Select An Algorithm", list(pages.keys()))

for page in selection:
    pages[page]()