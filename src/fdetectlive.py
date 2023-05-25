import streamlit as st
import cv2

def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def app():
    st.title("Face Detection 📹")
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        st.error('Could not open video capture device')
        return
    stframe = st.empty()
    while True:
        ret, frame = video_capture.read()
        if not ret:
            st.warning('Error while receiving video frame. Please try again.')
            break
        frame = detect_faces(frame)
        stframe.image(frame, channels="BGR")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    app()
