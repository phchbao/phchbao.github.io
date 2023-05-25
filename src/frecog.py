import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

def app(): 
    np.set_printoptions(suppress=True)
    model = load_model("C:/Users/laptops.vn/Desktop/fdttrained.h5", compile=False)
    with open("C:/Users/laptops.vn/Desktop/person.txt", "r") as f:
        class_names = f.readlines()
    camera = cv2.VideoCapture(0)
    st.title("Face Recognition 📹")
    image_placeholder = st.empty()
    while True:
        ret, image = camera.read()
        if not ret:
            continue
        image_with_boxes = image.copy()       
        image = np.asarray(image, dtype=np.float32)
        image = image / 255.0
        image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image_expanded = np.expand_dims(image_resized, axis=0)
        prediction = model.predict(image_expanded)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]
        cv2.rectangle(image_with_boxes, (0, 0), (image.shape[1], image.shape[0]), (0, 255, 0), 2)
        cv2.putText(image_with_boxes, f"{class_name}: {np.round(confidence_score * 100, 2)}%",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        image_placeholder.image(image_with_boxes, channels="BGR")
        keyboard_input = cv2.waitKey(1)
        if keyboard_input == 27:
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
   app()
