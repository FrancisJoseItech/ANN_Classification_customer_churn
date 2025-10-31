import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Loading the trained model
model = tf.keras.models.load_model('model.h5')

#Load the encoders and Scaler
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with  open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


## Streamlit App
st.title("Customer Churn Prediction")

#User Inputs

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 99, 40)
balance = st.number_input('Balance', 0.0)
credit_score = st.number_input('Credit Score', 0)
estimated_salary = st.number_input('Estimated Salary', 0.0)
tenure = st.select_slider('Tenure', options=list(range(0, 11)), value=3)
num_of_products = st.slider('Number of Products', 1, 5, 1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Preparing the input data
input_df = pd.DataFrame([{
    "CreditScore": credit_score,
    "Geography": geography,          # raw text
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary
}])

# one hot encode "Geography"
geo_encoded = onehot_encoder_geo.transform(input_df[['Geography']])
geo_cols = onehot_encoder_geo.get_feature_names_out()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_cols, index=input_df.index)

# concat EXACTLY like notebook: drop 'Geography', then append OHE
input_df = pd.concat(
    [input_df.drop(columns=["Geography"]), geo_encoded_df],
    axis=1
)

# label-encode Gender IN-PLACE (so column name stays 'Gender')
input_df["Gender"] = label_encoder_gender.transform(input_df["Gender"])


# Scaling the input data
full_input_df_scaled = scaler.transform(input_df)

# Predit churn
prediction = model.predict(full_input_df_scaled)
prediction_proba = prediction [0][0]

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')