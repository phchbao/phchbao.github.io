import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import os

def app():
    def speech_to_text(file_path):
        audio = AudioSegment.from_file(file_path)
        if audio.channels > 1:
            audio = audio.set_channels(1)
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000)
                audio.export("temp.wav", format="wav")
    
        r = sr.Recognizer()
        with sr.AudioFile("temp.wav") as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data, language="vi-VN")
    
        os.remove("temp.wav")
    
        return text

    st.title("Speech To Text 🎵👉🔠")
    uploaded_file = st.file_uploader("Choose audio file", type=["wav"])

    if uploaded_file is not None:
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        result = speech_to_text("temp.wav")
        st.subheader("Result:")
        st.write(result)
        st.subheader("Check result☑️")
        audio_bytes = uploaded_file.getvalue()
        st.audio(audio_bytes, format="audio/wav")
        
if __name__ == '__main__':
    app()
