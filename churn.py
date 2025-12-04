
# ---------------------------------------
# Customer Churn Prediction using PyCaret
# ---------------------------------------

# Import libraries
import pandas as pd
import streamlit as st
#import matplotlib.pyplot as plt
from pycaret.classification import *

st.title("Customer Churn Prediction with PyCaret")
# Load the dataset
file_path = r"WA_Fn-UseC_-Telco-Customer-Churn.csv"   # update path if needed
df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!")
print(df.head())

# ---------------------------------------
# PyCaret Setup
# ---------------------------------------
# target column is "Churn"

clf = setup(
    data=df,
    target='Churn',
    session_id=42,
    normalize=True,
    verbose=True,
    categorical_features=[
        col for col in df.columns
        if df[col].dtype == 'object' and col != 'Churn'
    ]
)
# ---------------------------------------
# BEST MODEL (print it first)
# ---------------------------------------
best_model = compare_models(sort='AUC')
st.write("### Best Model Based on AUC")
st.write(best_model)

plot_model(best_model, plot='auc', display_format='streamlit')


# ---- CONFUSION MATRIX PLOTS ----
st.write("### Confusion Matrix – Best Model")
plot_model(best_model, plot='confusion_matrix', display_format='streamlit')


# Finalize & Save
final_model = finalize_model(best_model)
save_model(final_model, "churn_model")