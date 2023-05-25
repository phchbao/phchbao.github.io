import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import random

def determine_winner(player_choice, computer_choice):
    if (player_choice == "Scissor" and computer_choice == "Rock") or \
            (player_choice == "Paper" and computer_choice == "Scissor") or \
            (player_choice == "Rock" and computer_choice == "Paper"):
        return "🤖 Computer wins!"
    elif (player_choice == "Rock" and computer_choice == "Scissor") or \
            (player_choice == "Scissor" and computer_choice == "Paper") or \
            (player_choice == "Paper" and computer_choice == "Rock"):
        return "🍀 Player wins!"
    else:
        return "TIE ✖️"

def app(): 
    np.set_printoptions(suppress=True)
    model = load_model("C:/Users/laptops.vn/Desktop/odttrained.h5", compile=False)
    with open("C:/Users/laptops.vn/Desktop/ingame.txt", "r") as f:
        class_names = f.readlines()
    camera = cv2.VideoCapture(0)
    st.title("Rock Paper Scissor 🎲")
    image_placeholder = st.empty()
    player_choice = None
    computer_choice = None
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
        if player_choice is None:
            player_choice = class_name
            computer_choice = random.choice(["Rock", "Paper", "Scissor"])
            st.write("Player chooses:", player_choice)
            st.write("Computer chooses:", computer_choice)
            winner = determine_winner(player_choice, computer_choice)
            st.write(winner)
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app()
