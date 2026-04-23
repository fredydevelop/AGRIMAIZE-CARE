#Importing the dependencies
import numpy as np
import pandas as pd
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import streamlit as st
import base64
import pickle as pk


import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(page_title="Agromaize Care", layout="centered")

MODEL_PATH = "second_maize_model_checkpoint.keras"

class_labels = {
    0: "Maize fall armyworm",
    1: "Maize grasshoper",
    2: "Maize healthy",
    3: "Maize leaf beetle",
    4: "Maize leaf blight",
    5: "Maize streak virus",
    6: "Maize_cercospora Leaf Spot"
}

def get_recommendation(predicted_category):
    if predicted_category == "Maize healthy":
        return "The maize leaf appears healthy. Maintain proper farm monitoring and continue good agronomic practices."
    elif predicted_category == "Maize fall armyworm":
        return "Possible fall armyworm infestation detected. Inspect nearby leaves and stems, and apply appropriate pest control measures."
    elif predicted_category == "Maize grasshoper":
        return "Possible grasshopper damage detected. Inspect the farm environment and consider suitable pest management practices."
    elif predicted_category == "Maize leaf beetle":
        return "Possible maize leaf beetle presence detected. Check for feeding damage and apply suitable control measures where necessary."
    elif predicted_category == "Maize leaf blight":
        return "Possible leaf blight detected. Remove heavily affected leaves and consider recommended disease management practices."
    elif predicted_category == "Maize streak virus":
        return "Possible maize streak virus detected. Monitor spread, remove severely affected plants, and control insect vectors."
    elif predicted_category == "Maize_cercospora Leaf Spot":
        return "Possible cercospora leaf spot detected. Improve field hygiene and apply appropriate disease management practices."
    else:
        return "No recommendation available."

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

def predict_maize(img_array):
    loaded_model = load_model(MODEL_PATH)
    prediction = loaded_model.predict(img_array)
    predicted_class = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction) * 100)
    predicted_category = class_labels[predicted_class]
    return predicted_category, confidence

with st.sidebar:
    st.image("maizelogo.png", width=120)
    st.header("AGRIMAIZE CARE")
    uploaded_file = st.file_uploader(
        "Upload a maize leaf image",
        type=["jpg", "jpeg", "png", "bmp"]
    )
    predict_clicked = st.button("Predict")

st.title("Maize Leaf Classification")

if uploaded_file is not None:
    try:
        img, img_array = preprocess_image(uploaded_file)

        st.subheader("Uploaded Image")
        st.image(img, width=120)

        if predict_clicked:
            predicted_category, confidence = predict_maize(img_array)
            recommendation = get_recommendation(predicted_category)

            st.subheader("Prediction Result")
            if predicted_category == "Maize healthy":
                st.success(f"Predicted Class: {predicted_category}")
            else:
                st.warning(f"Predicted Class: {predicted_category}")

            st.info(f"Confidence Score: {confidence:.2f}%")

            st.subheader("Recommendation")
            st.write(recommendation)

    except UnidentifiedImageError:
        st.error("Invalid file uploaded. Please upload a valid image file only.")
    except Exception as e:
        st.error(f"An error occurred while processing the image: {str(e)}")
else:
    st.info("Please upload a maize leaf image from the sidebar.")
