import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Heart Disease Prediction",page_icon="⚕️",layout="centered",initial_sidebar_state="expanded")
loaded_model = pickle.load(open('Random_forest_model.sav', 'rb'))

df = pd.read_csv('heart.csv')
X = df[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
       'FastingBS', 'RestingECG', 'MaxHR','ExerciseAngina','Oldpeak','ST_Slope',]]
y = df['HeartDisease']


def disease_pred(input_data):
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return "The person hasn't heart disease"
    else:
        return "Person has heart disease"
def main():
    st.write("""
    # Heart disease Prediction App
    This app predicts If a patient has a heart disease or not.
    """)
    Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope
    Age = st.slider('Enter your age: ',5,90)
    Sex  = st.selectbox('Sex',(0,1))
    ChestPainType = st.selectbox('Chest pain type',(0,1,2,3))
    RestingBP = st.slider('Resting blood pressure: ',100,160)
    Cholesterol = st.slider('Serum cholestoral in mg/dl: ',100,320)
    FastingBS = st.selectbox('Fasting blood sugar',(0,1))
    RestingECG = st.selectbox('Resting electrocardiographic results: ',(0,1,2))
    MaxHR = st.slider('Maximum heart rate achieved: ',72,200)
    ExerciseAngina = st.selectbox('Exercise induced angina: ',(0,1))
    Oldpeak = st.number_input('oldpeak ')
    ST_Slope = st.selectbox('Slope of the peak exercise ST segmen: ',(0,1,2))

    diagnosis = ''

    if st.button('Heart Disease Prediction result'):
        diagnosis = disease_pred([Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope])

        st.success(diagnosis)

if __name__=='__main__':
    main()


