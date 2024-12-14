import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.title("LinkedIn User Prediction")
st.write("Enter your information below to predict LinkedIn usage.")

# Function to load and clean the dataset
@st.cache_data
def load_and_clean_data():
    s = pd.read_csv("social_media_usage.csv")
    def clean_sm(x):
        return np.where(x == 1, 1, 0)

    columns_ss = ["income", "educ2", "par", "marital", "age", "gender", "web1h"]
    ss = s.loc[:, columns_ss]

    ss = pd.DataFrame({
        "income": np.where(ss["income"] > 10, np.nan, ss["income"]),
        "educ2": np.where(ss["educ2"] > 9, np.nan, ss["educ2"]),
        "par": np.where(ss["par"] == 1, 1, 0),
        "marital": np.where(ss["marital"] == 1, 1, 0),
        "gender": np.where(ss["gender"] == 2, 1, 0),
        "age": np.where(ss["age"] > 98, np.nan, ss["age"]),
        "sm_li": clean_sm(ss["web1h"])
    })

    ss = ss.dropna()
    return ss

# Load the cleaned data
ss = load_and_clean_data()

# Prepare feature set and target
X = ss[["income", "educ2", "par", "marital", "gender", "age"]]
y = ss["sm_li"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=487)

# Train model
@st.cache_resource
def train_model(X_train, y_train):
    model = LogisticRegression(class_weight="balanced", random_state=487)
    model.fit(X_train, y_train)
    return model

lr = train_model(X_train, y_train)

# Evaluate model
accuracy = lr.score(X_test, y_test)
st.write(f"Model Accuracy: {accuracy:.2f}")

# User input
st.write("### Enter Your Data")
income = st.slider("Income Level (1-10)", 1, 10, value=5)
education = st.slider("Education Level (1-8)", 1, 8, value=4)
parent = st.selectbox("Are you a parent?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
married = st.selectbox("Are you married?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
female = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 1 else "Male")
age = st.slider("Age", 1, 98, value=30)

# Prediction
if st.button("Predict"):
    input_data = np.array([[income, education, parent, married, female, age]])
    prediction = lr.predict(input_data)[0]
    probability = lr.predict_proba(input_data)[0][1]

    st.write(f"Prediction: {'LinkedIn User' if prediction == 1 else 'Not a LinkedIn User'}")
    st.write(f"Probability of being a LinkedIn user: {probability:.2f}")
