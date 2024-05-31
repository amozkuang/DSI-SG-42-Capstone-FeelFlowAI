import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import tempfile
import soundfile as sf
import os
import h5py
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import StandardScaler
import whisper
import streamlit as st
import librosa
import librosa.display
import matplotlib.colors as mcolors

# Set the page config to load the favicon and set the page title
st.set_page_config(page_title="FeelFlow AI", page_icon="img/ff.png")

st.markdown("""
<style>
    header {visibility: hidden;}
    .css-18e3th9 {padding-top: 0 !important;}
    .css-1d391kg {background-color: #f5f5f5;}
    .css-1v3fvcr {font-family: 'Archivo'; font-size: 22px; color: black;}
    .stButton>button {font-size: 16px;}
</style>
""", unsafe_allow_html=True)

# Load model and other components
def load_components(model_path):
    try:
        model = load_model(model_path)
        encoder = pickle.load(open('encoders/label_encoder.pkl', 'rb'))
        scaler = StandardScaler()
        scaler.fit(np.random.rand(10, 164))  # Dummy initialization
        return model, encoder, scaler
    except Exception as e:
        st.error(f"Failed to load components: {e}")
        st.stop()

# Record audio from microphone
def record_audio(duration=5, sample_rate=16000):
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

# Placeholder feature extraction
def extract_features(audio_data):
    return np.random.rand(164)  # This should be replaced with real feature extraction logic

# Process audio data
def process_audio(audio_data, model, scaler, encoder):
    features = extract_features(audio_data)
    features_scaled = scaler.transform([features])
    features_reshaped = features_scaled.reshape(1, -1, 1)
    predictions = model.predict(features_reshaped)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = encoder.inverse_transform(predicted_class)
    prediction_probability = np.max(predictions, axis=1)
    return predicted_label[0], prediction_probability[0]

def transcribe_audio(audio_data):
    model = whisper.load_model("base")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, audio_data, 16000)
    result = model.transcribe(temp_file.name)
    os.unlink(temp_file.name)
    return result['text']

# Save audio to file
def save_audio(audio_data, file_path):
    sf.write(file_path, audio_data, 16000)

# Save transcription to file
def save_transcription(transcription, file_path):
    with open(file_path, 'w') as file:
        file.write(transcription)

def visualize_audio(audio_data, predicted_emotion, probability):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # Waveplot
    librosa.display.waveshow(audio_data, sr=16000, ax=axs[0])
    axs[0].set_title("Waveplot")
    
    # Spectrogram
    X = librosa.stft(audio_data)
    Xdb = librosa.amplitude_to_db(abs(X))
    img = librosa.display.specshow(Xdb, sr=16000, x_axis='time',
    y_axis='hz', ax=axs[1])
    axs[1].set_title("Spectrogram")
    fig.colorbar(img, ax=axs[1], format='%+2.0f dB')

    # Dashboard meter for emotions
    colors = {'Angry': 'red', 'Sad': 'amber', 'Fear': 'orange', 'Disgust': 'yellow', 'Happy': 'lightgreen'}
    emotion_color = colors.get(predicted_emotion, 'grey')
    axs[2].barh([predicted_emotion], [probability], color=emotion_color)
    axs[2].set_xlim(0, 1)
    axs[2].set_title("Emotion Probability Meter")
    
    st.pyplot(fig)

# Main app
def main():
    logo_path = 'img/ff.png'
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image(logo_path, width=90)
    with col2:
        st.markdown("""
        <h1 style='margin-left: 0;'>FeelFlow AI</h1>
        <h2 style='margin-left: 1; font-size: 25px;'>Decoding Emotions, Advancing Patient Support</h3>
        <p style='margin-left: 0; font-size: 20px; width: 100%;'>An AI-powered tool to analyse emotions from voice recordings.</p>
        """, unsafe_allow_html=True)
    

    model_path = 'models/wavenet_model.h5'
    model, encoder, scaler = load_components(model_path)

    if st.button("Record and Analyse"):
        audio_data = record_audio()
        predicted_emotion, probability = process_audio(audio_data, model, scaler, encoder)
        transcription = transcribe_audio(audio_data)
        
        st.write(f"Predicted Emotion: {predicted_emotion} ({probability * 100:.2f}%)")
        st.write(f"Transcription: {transcription}")
        
        visualize_audio(audio_data, predicted_emotion, probability)

        if st.button("Download Audio"):
            audio_file_path = "audio/audio.wav"
            save_audio(audio_data, audio_file_path)
            st.success("Audio saved successfully!")

        if st.button("Download Transcription"):
            transcription_file_path = "transcriptions/transcription.txt"
            save_transcription(transcription, transcription_file_path)
            st.success("Transcription saved successfully!")

        # Footer
        st.markdown("""
        <p style='text-align: center; color: gray; font-size: 14px;'>*Disclaimer: Please note that this is still in Phase 1 beta stage (as of 14 May 2024). Thank you for being patient with us.*</p>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
