

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
from pickle import dump
from pickle import load


data = pd.read_csv("df.csv")


st.title('Model Deployment: Random forest Regression')

st.sidebar.header('User Input Parameters')
cols1 = list(data["Item_Type"].unique())

cols2 = list(data["Item_Fat_Content"].unique())
cols3 = list(data["Outlet_Size"].unique())
cols4 = list(data["Outlet_Location_Type"].unique())
cols5 = list(data["Outlet_Type"].unique())


def user_input_features():
    Item_Weight = st.number_input('item weight')
    Item_Visibility = st.number_input('Visibility of item')
    Item_Type = st.sidebar.selectbox('select item type',cols1)
    Item_MRP = st.number_input("item mrp")
    Outlet_Years = st.number_input("Age")
    
    Item_Fat_Content = st.sidebar.selectbox("select fat content",cols2)
    Outlet_Size = st.sidebar.selectbox('select size of outlet',cols3)
    Outlet_Location_Type = st.sidebar.selectbox('select location type',cols4)
    Outlet_Type = st.sidebar.selectbox('select the outlet type',cols5)
      # Create the "New_Item_Type" column based on your logic
    New_Item_Type = st.sidebar.selectbox('select drinks or food or NC', ['Food', 'Drinks', 'Non-Consumable'])
    
    
    
    
    
    data = {'Item_Weight':Item_Weight,
            'Item_Fat_Content':Item_Fat_Content,
           'Item_Visibility':Item_Visibility,
            'Item_Type':Item_Type,
            'Item_MRP':Item_MRP,
            'Outlet_Size':Outlet_Size,
            'Outlet_Location_Type':Outlet_Location_Type,           
            'Outlet_Type':Outlet_Type,
            'New_Item_Type':New_Item_Type,
            'Outlet_Years':Outlet_Years,                                                       

           }
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

#encoder
# Load the label encoders
loaded_label_encoders = {}
cat_cols = ["Item_Fat_Content", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type", "New_Item_Type", "Item_Type"]

for col in cat_cols:
    le_filename = f"{col}_label_encoder.sav"
    with open(le_filename, 'rb') as le_file:
        loaded_label_encoders[col] = load(le_file)

# Transform the new data using the loaded LabelEncoders
for col, le in loaded_label_encoders.items():
    if col in df.columns:
        df[col] = le.transform(df[col])



# scaler the features 
scaler = load(open('s_c.sav', 'rb'))
df = scaler.transform(df)


#features 


# load the model from disk
loaded_model = load(open('RF_model.sav', 'rb'))


prediction = loaded_model.predict(df)
original_prediction = np.exp(prediction) - 1

submit = st.button("Predict")
if submit:
    st.subheader('Predicted Result')
    st.write(original_prediction)

