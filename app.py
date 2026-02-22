import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# ------------------ Page Config ------------------
st.set_page_config(page_title="ID3 Decision Tree", layout="wide")

st.title("🌳 ID3 Decision Tree Classification App")
st.write("This app demonstrates ID3 (Decision Tree) classification.")

# ------------------ Sample Dataset ------------------
data = {
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain",
                "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast"],
    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool",
                    "Mild", "Cool", "Mild", "Mild", "Mild", "Hot"],
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal",
                 "Normal", "High", "Normal", "Normal", "Normal", "High"],
    "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong",
             "Strong", "Weak", "Weak", "Weak", "Strong", "Strong"],
    "PlayTennis": ["No", "No", "Yes", "Yes", "Yes", "No",
                   "Yes", "No", "Yes", "Yes", "Yes", "Yes"]
}

df = pd.DataFrame(data)

# Convert categorical to numeric
df_encoded = pd.get_dummies(df.drop("PlayTennis", axis=1))
y = df["PlayTennis"]

# ------------------ Model ------------------
model = DecisionTreeClassifier(criterion="entropy")
model.fit(df_encoded, y)

# ------------------ Sidebar Input ------------------
st.sidebar.header("Input Features")

outlook = st.sidebar.selectbox("Outlook", ["Sunny", "Overcast", "Rain"])
temperature = st.sidebar.selectbox("Temperature", ["Hot", "Mild", "Cool"])
humidity = st.sidebar.selectbox("Humidity", ["High", "Normal"])
wind = st.sidebar.selectbox("Wind", ["Weak", "Strong"])

# Create input dataframe
input_data = pd.DataFrame({
    "Outlook": [outlook],
    "Temperature": [temperature],
    "Humidity": [humidity],
    "Wind": [wind]
})

input_encoded = pd.get_dummies(input_data)
input_encoded = input_encoded.reindex(columns=df_encoded.columns, fill_value=0)

# Prediction
prediction = model.predict(input_encoded)[0]

st.sidebar.subheader("Prediction Result")
st.sidebar.success(f"Play Tennis: {prediction}")

# ------------------ Show Dataset ------------------
st.subheader("Dataset Used")
st.dataframe(df)

# ------------------ Decision Tree Visualization ------------------
st.subheader("Decision Tree Visualization")

fig, ax = plt.subplots(figsize=(12, 6))
tree.plot_tree(model, feature_names=df_encoded.columns,
               class_names=model.classes_,
               filled=True)
st.pyplot(fig)

# ------------------ Info Section ------------------
with st.expander("ℹ️ What is ID3?"):
    st.write("""
    ID3 (Iterative Dichotomiser 3) is a decision tree algorithm.
    It uses:
    - Entropy
    - Information Gain
    To decide the best attribute to split the data.
    """)
