import streamlit as st
st.title("LinkedIn User Prediction")
st.write("Enter your information below to predict LinkedIn usage")

#import necessary libraries
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#read in data and assign dataframe "s"
s = pd.read_csv("C:/Users/deeha/OneDrive/Desktop/Project/social_media_usage.csv")


#import the numpy library to check fo null values and convert them into NaN
import numpy as np

#define function clean_sm that takes one input x and uses np.where to return 0 or 1. 
def clean_sm(x): 
    return np.where(x == 1, 1, 0)

# test column_ss
columns_ss = ["income", "educ2", "par", "marital", "age", "gender", "web1h"]
ss = s.loc[:, columns_ss]

print(columns_ss)

#3. Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

columns_ss = ["income", "educ2", "par", "marital", "age", "gender", "web1h"]
ss = s.loc[:, columns_ss]


ss = pd.DataFrame({
    "income":np.where(ss["income"] > 10, np.nan, ss["income"]),
    "educ2":np.where(ss["educ2"] > 9,  np.nan, ss["educ2"]),
    "par":np.where(ss["par"] == 1, 1, 0),  # Parent binary
    "marital":np.where(ss["marital"] == 1, 1, 0),  # Marital binary
    "gender":np.where(ss["gender"] == 2, 1, 0), # gender binary
    "age":np.where(ss["age"] > 98, np.nan, ss["age"]),
    "sm_li":clean_sm(ss["web1h"])})


# Drop unnecessary columns and rows with missing values
ss = ss.dropna()

# Check the cleaned DataFrame
print("Cleaned DataFrame SS:")
print(ss.head())

#create target vector y
y = ss["sm_li"]

#create feature set X
X = ss[["income", "educ2", "par", "marital", "gender", "age"]]

#check
print("Shape of Feature Set (X):", X.shape)
print("Shape of Target Vector (y):", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y, #same number of target in training and test set
                                                    test_size=0.2, #  20% of data is for testing
                                                    random_state=487) # set for reproducibility

#initialize algorithm for logistic regression model
lr = LogisticRegression(class_weight = "balanced", random_state = 487)

#fit algorithm to training data
lr.fit(X_train, y_train)

#Evaluate the model's accuracy using the testing data
accuracy = lr.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

#Make predictions using the model and the testing data
y_pred = lr.predict(X_test)

st.write(f"Model Accuracy: {accuracy:.2f}")
