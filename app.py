import streamlit as st
from classification import show_classification_page
from regression import show_regression_page

st.set_page_config(
    page_title="ML Project",
    layout="wide"
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose Page",
    ["Classification", "Regression"]
)

if page == "Classification":
    show_classification_page()
elif page == "Regression":
    show_regression_page()