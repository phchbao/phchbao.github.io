import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import model_from_json 
from tensorflow.keras.optimizers import SGD 
import cv2

model_architecture = "C:/Users/laptops.vn/Desktop/NhanDangChuSoVietTayTiengAnhMNIST_GUI/digit_config.json"
model_weights = "C:/Users/laptops.vn/Desktop/NhanDangChuSoVietTayTiengAnhMNIST_GUI/digit_weight.h5"
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights) 
optim = SGD()
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"]) 
mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_test_image = X_test
RESHAPED = 784
X_test = X_test.reshape(10000, RESHAPED)
X_test = X_test.astype('float32')
X_test /= 255

def app():
    st.title('Handwritten Digit Recognition 🔢✍️')
    cvs_digit = st.empty()
    lbl_ket_qua = st.empty()
    btn_action = st.button('Generate image and result')
    if btn_action:
        index = np.random.randint(0, 9999, 150)
        digit_random = np.zeros((10*28, 15*28), dtype=np.uint8)
        for i in range(0, 150):
            m = i // 15
            n = i % 15
            digit_random[m*28:(m+1)*28, n*28:(n+1)*28] = X_test_image[index[i]] 
        cv2.imwrite('C:/Users/laptops.vn/Desktop/NhanDangChuSoVietTayTiengAnhMNIST_GUI/digit_random.jpg', digit_random)
        image = Image.open('C:/Users/laptops.vn/Desktop/NhanDangChuSoVietTayTiengAnhMNIST_GUI/digit_random.jpg')
        cvs_digit.image(image, width=421)
        X_test_sample = np.zeros((150, 784), dtype=np.float32)
        for i in range(0, 150):
            X_test_sample[i] = X_test[index[i]]
        prediction = model.predict(X_test_sample)
        s = ''
        for i in range(0, 150):
            ket_qua = np.argmax(prediction[i])
            s = s + str(ket_qua) + ' '
            if (i+1) % 15 == 0:
                s = s + '\n'
        lbl_ket_qua.text(s)

if __name__ == "__main__":
    app()
