import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained modelrandom
def load_model():
    model = joblib.load(r'C:\Users\LENOVO\OneDrive\Desktop\mltask6\random_forest_model.pkl')
    return model


# Load the features used by the model
def load_features():
    features = joblib.load(r'C:\Users\LENOVO\OneDrive\Desktop\mltask6\model_features.pkl')
    return features


st.title("Student Dropout Prediction App")

# User inputs for predicting student dropout
school = st.selectbox('School', ['GP', 'MS'])
sex = st.selectbox('Gender', ['male', 'female'])
age = st.slider('Age', 15, 22, 18)
address = st.selectbox('Home Address', ['urban', 'rural'])
famsize = st.selectbox('Family Size', ['GT3', 'LE3'])
pstatus = st.selectbox('Parental Status', ['together', 'apart'])
traveltime = st.slider('Travel Time to School (in minutes)', 1, 4, 1)
studytime = st.slider('Study Time (hours per week)', 1, 4, 2)
failures = st.number_input('Number of Past Class Failures', min_value=0, max_value=4, value=0)
famsup = st.selectbox('Family Support', ['yes', 'no'])
schoolsup = st.selectbox('School Support', ['yes', 'no'])
activities = st.selectbox('Extra-curricular Activities', ['yes', 'no'])
higher = st.selectbox('Wants Higher Education', ['yes', 'no'])
internet = st.selectbox('Internet Access at Home', ['yes', 'no'])
romantic = st.selectbox('In a Romantic Relationship', ['yes', 'no'])
absences = st.number_input('Number of School Absences', min_value=0, max_value=93, value=0)

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'School_GP': [1 if school == 'GP' else 0],
    'Sex_male': [1 if sex == 'male' else 0],
    'Age': [age],
    'Address': [1 if address == 'urban' else 0],
    'Family_Size_GT3': [1 if famsize == 'GT3' else 0],
    'Parental_Status_together': [1 if pstatus == 'together' else 0],
    'Travel_Time': [traveltime],
    'Study_Time': [studytime],
    'Failures': [failures],
    'Family_Support_yes': [1 if famsup == 'yes' else 0],
    'School_Support_yes': [1 if schoolsup == 'yes' else 0],
    'Activities_yes': [1 if activities == 'yes' else 0],
    'Wants_Higher_Education_yes': [1 if higher == 'yes' else 0],
    'Internet_Access_yes': [1 if internet == 'yes' else 0],
    'In_Relationship_yes': [1 if romantic == 'yes' else 0],
    'Absences': [absences],
})

# Load the model and features
model = load_model()
features = load_features()

# Align input data with model's features
input_data = input_data.reindex(columns=features, fill_value=0)

# Predict and display results
if st.button('Predict'):
    prediction = model.predict(input_data)
    if prediction == 1:
        st.success("The student is at risk of dropping out.")
    else:
        st.success("The student is not at risk of dropping out.")
