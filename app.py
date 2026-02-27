import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import streamlit as st
import base64

# ---------------- LOGIN CONFIG ---------------- #

USER_CREDENTIALS = {
    "doctor": {
        "password": st.secrets["auth"]["doctor_password"],
        "role": "Doctor"
    },
    "staff": {
        "password": st.secrets["auth"]["staff_password"],
        "role": "Healthcare Staff"
    }
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

st.markdown("""
<style>

/* Make headings and labels white */
h1, h2, h3, h4, h5, h6, label, p, span {
    color: white !important;
}

/* Make number input text white*/
input[type="number] {
    color: white !important;
}

/*keep checkbox text unchanged*/
input[type="checkbox"] {
    accent-color: #00BFFF;
}

</style>
""", unsafe_allow_html=True)

# Function to set background
def set_bg(image_file):

    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    css = f"""
    <style>

    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Make title bold and colored */
    h1 {{
        color: #ffffff;
        text-align: center;
        font-size: 50px;
        font-weight: bold;
    }}

    /* Make labels bold */
    label {{
        font-weight: bold;
        color: white;
        font-size: 18px;
    }}

    /* Button styling */
    .stButton>button {{
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
    }}

    </style>
    """

    st.markdown(css, unsafe_allow_html=True)


# call function
set_bg("medical_bg.jpg")

# Load Models (Fusion Logic)

mri_model = load_model("mri_model.h5")

clinical_model = joblib.load("clinical_model.pkl")

scaler = joblib.load("scaler.pkl")

if not st.session_state.logged_in:

    st.title("Healthcare Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username in USER_CREDENTIALS and password == USER_CREDENTIALS[username]["password"]:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()

# Title

st.markdown("<h1 style= 'color:White;'> Alzheimer Detection System</h1>", unsafe_allow_html=True)


# Mode Selection

mode = st.radio("Select Mode",

                ["MRI Mode", "Clinical Mode", "Fusion Mode"])


# MRI Mode

if mode == "MRI Mode":

    file = st.file_uploader("Upload MRI Image", type=["jpg","png"])

    if file:

        img = image.load_img(file,

                             target_size=(224,224),

                             color_mode="grayscale")

        img = image.img_to_array(img)/255.0

        img = np.expand_dims(img, axis=0)

        pred = mri_model.predict(img)

        classes = ["Non Demented",

                   "Very Mild Demented",

                   "Mild Demented",

                   "Moderate Demented"]

        result = classes[np.argmax(pred)]

        confidence_stage = np.max(pred) * 100

        st.success(result)

        st.info(f"Confidence: {confidence_stage:.2f}%")


# Clinical Mode

elif mode == "Clinical Mode":

    st.header("Clinical Based Alzheimer Risk Assessment")

    age = st.number_input("Age", min_value=0)

    gender = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender == "Male" else 0

    education = st.number_input("Education Level", min_value=0)

    smoking = st.selectbox("Smoking", ["No", "Yes"])
    smoking = 1 if smoking == "Yes" else 0

    alcohol = st.number_input("Alcohol Consumption", min_value=0.0)

    physical = st.number_input("Physical Activity", min_value=0.0)

    diet = st.number_input("Diet Quality", min_value=0.0)

    sleep = st.number_input("Sleep Quality", min_value=0.0)

    family = st.selectbox("Family History of Alzheimer’s", ["No", "Yes"])
    family = 1 if family == "Yes" else 0

    cardio = st.selectbox("Cardiovascular Disease", ["No", "Yes"])
    cardio = 1 if cardio == "Yes" else 0

    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    diabetes = 1 if diabetes == "Yes" else 0

    depression = st.selectbox("Depression", ["No", "Yes"])
    depression = 1 if depression == "Yes" else 0

    headinjury = st.selectbox("Head Injury", ["No", "Yes"])
    headinjury = 1 if headinjury == "Yes" else 0

    hyper = st.selectbox("Hypertension", ["No", "Yes"])
    hyper = 1 if hyper == "Yes" else 0

    mmse = st.number_input("MMSE Score", min_value=0.0)

    functional = st.number_input("Functional Assessment", min_value=0.0)

    memory = st.selectbox("Memory Complaints", ["No", "Yes"])
    memory = 1 if memory == "Yes" else 0

    behavior = st.selectbox("Behavioral Problems", ["No", "Yes"])
    behavior = 1 if behavior == "Yes" else 0

    personality = st.selectbox("Personality Changes", ["No", "Yes"])
    personality = 1 if personality == "Yes" else 0

    tasks = st.selectbox("Difficulty Completing Tasks", ["No", "Yes"])
    tasks = 1 if tasks == "Yes" else 0

    forget = st.selectbox("Forgetfulness", ["No", "Yes"])
    forget = 1 if forget == "Yes" else 0


    if st.button("Predict Risk"):

        data = scaler.transform([[

            age, gender, education, smoking, alcohol,

            physical, diet, sleep, family, cardio,

            diabetes, depression, headinjury, hyper,

            mmse, functional, memory, behavior,

            personality, tasks, forget

        ]])

        prediction = clinical_model.predict(data)

        prob = clinical_model.predict_proba(data)

        confidence = max(prob[0]) * 100

        if prediction[0] == 1:

            st.error("High Risk of Alzheimer’s")

        else:

            st.success("Low Risk of Alzheimer’s")

        st.info(f"Confidence: {confidence:.2f}%")

# Fusion Mode

elif mode == "Fusion Mode":

    st.header("Fusion Mode: Stage + Risk Prediction")

    col1, col2 = st.columns(2)

    # MRI INPUT
    with col1:
        st.subheader("MRI Analysis")
        file = st.file_uploader("Upload MRI Image", type=["jpg","png"])


    # CLINICAL INPUTS

    with col2:
        st.subheader("Clinical Assessment")    
        age = st.number_input("Age")
        gender = st.selectbox("Gender", ["Male", "Female"])
        gender = 1 if gender == "Male" else 0
        education = st.number_input("Education Level")
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        smoking = 1 if smoking == "Yes" else 0
        alcohol = st.number_input("Alcohol Consumption")
        physical = st.number_input("Physical Activity")
        diet = st.number_input("Diet Quality")
        sleep = st.number_input("Sleep Quality")
        family = st.selectbox("Family History", ["No", "Yes"])
        family = 1 if family == "Yes" else 0
        cardio = st.selectbox("Cardiovascular Disease", ["No", "Yes"])
        cardio = 1 if cardio == "Yes" else 0
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        diabetes = 1 if diabetes == "Yes" else 0
        depression = st.selectbox("Depression", ["No", "Yes"])
        depression = 1 if depression == "Yes" else 0
        headinjury = st.selectbox("Head Injury", ["No", "Yes"])
        headinjury = 1 if headinjury == "Yes" else 0
        hyper = st.selectbox("Hypertension", ["No", "Yes"])
        hyper = 1 if hyper == "Yes" else 0
        mmse = st.number_input("MMSE")
        functional = st.number_input("Functional Assessment")
        memory = st.selectbox("Memory Complaints", ["No", "Yes"])
        memory = 1 if memory == "Yes" else 0
        behavior = st.selectbox("Behavioral Problems", ["No", "Yes"])
        behavior = 1 if behavior == "Yes" else 0
        personality = st.selectbox("Personality Changes", ["No", "Yes"])
        personality = 1 if personality == "Yes" else 0
        tasks = st.selectbox("Difficulty Completing Tasks", ["No", "Yes"])
        tasks = 1 if tasks == "Yes" else 0
        forget = st.selectbox("Forgetfulness", ["No", "Yes"])
        forget = 1 if forget == "Yes" else 0


    if st.button("Predict Both"):


        # MRI Prediction

        if file:

            img = image.load_img(file, target_size=(224,224), color_mode="grayscale")

            img = image.img_to_array(img)/255.0

            img = np.expand_dims(img, axis=0)

            pred = mri_model.predict(img)

            classes = ["Non Demented",

                       "Very Mild Demented",

                       "Mild Demented",

                       "Moderate Demented"]

            stage = classes[np.argmax(pred)]

            st.success("Alzheimer Stage: " + stage)


        # Clinical Prediction

        data = scaler.transform([[

            age, gender, education, smoking, alcohol,

            physical, diet, sleep, family, cardio,

            diabetes, depression, headinjury, hyper,

            mmse, functional, memory, behavior,

            personality, tasks, forget

        ]])

        risk = clinical_model.predict(data)

        if risk[0] == 1:

            st.error("Risk Level: High Risk")

        else:


            st.success("Risk Level: Low Risk")


