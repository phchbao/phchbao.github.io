import streamlit as st
import numpy as np
import cv2 as cv

def app():
    st.title("Face Detection 🖼️")
    uploaded_file = st.file_uploader("Choose image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, 1)
        detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        st.image(img, channels="BGR")
        st.write("Number of faces detected: ", len(faces))
        for (x,y,w,h) in faces:
            cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        result_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        st.image(result_img, channels="RGB")
