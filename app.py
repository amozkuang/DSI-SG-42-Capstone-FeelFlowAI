import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import tempfile
import soundfile as sf
import os
import pickle
from sklearn.preprocessing import StandardScaler
import whisper
import streamlit as st
import librosa
import librosa.display
import matplotlib.colors as mcolors
from tensorflow.keras.models import Model, Input
from tensorflow.keras.layers import Conv1D, Add, Activation, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set the page config to load the favicon and set the page title
st.set_page_config(page_title="FeelFlow AI", page_icon="favicon.ico")

st.markdown("""
<style>
    header {visibility: hidden;}
    .css-18e3th9 {padding-top: 0 !important;}
    .css-1d391kg {background-color: #f5f5f5;}
    .css-1v3fvcr {font-family: 'Archivo'; font-size: 22px; color: black;}
    .stButton>button {font-size: 16px;}
</style>
""", unsafe_allow_html=True)

def WaveNetResidualConv1D(num_filters, kernel_size, dilation_rate):
    def build_residual_block(l_input):
        l_conv_dilated = Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal', activation='relu')(l_input)
        l_conv_out = Conv1D(num_filters, 1, padding='same', activation='relu')(l_conv_dilated)
        l_out = Add()([l_input, l_conv_out])
        return l_out, l_conv_out
    return build_residual_block

def build_wavenet(input_shape, num_filters, kernel_size):
    input_layer = Input(shape=input_shape)
    x = input_layer
    skip_connections = []
    for dilation_rate in [1, 2, 4, 8, 16, 32]:
        x, skip = WaveNetResidualConv1D(num_filters, kernel_size, dilation_rate)(x)
        x = Dropout(0.2)(x)  # Add dropout to model
        skip_connections.append(skip)
    out = Add()(skip_connections)
    out = Activation('relu')(out)
    out = Conv1D(num_filters, 1, activation='relu')(out)
    out = GlobalAveragePooling1D()(out)
    out = Dense(6, activation='softmax')(out)  # Number of classes assumed to be 6
    model = Model(inputs=input_layer, outputs=out)
    model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load components
encoder = pickle.load(open('encoders/label_encoder.pkl', 'rb'))
scaler = StandardScaler()
scaler.fit(np.random.rand(10, 164))  # Dummy initialization
model = build_wavenet((164, 1), 64, 3)  # Adjust the input_shape as needed

def process_audio(audio_data, model, scaler, encoder):
    features = np.random.rand(164).reshape(1, -1)  # Placeholder feature extraction
    features_scaled = scaler.transform(features)
    features_reshaped = np.expand_dims(features_scaled, axis=-1)
    predictions = model.predict(features_reshaped)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = encoder.inverse_transform(predicted_class)
    prediction_probability = np.max(predictions)
    return predicted_label[0], prediction_probability

# Main app
def main():
    st.title('FeelFlow AI: Emotion Recognition')
    logo_path = 'img/ff.png'
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(logo_path, width=120)
    with col2:
        st.markdown("""
        <h1 style='margin-left: 0;'>Decoding Emotions, Advancing Patient Support</h1>
        <p style='margin-left: 0; font-size: 18px;'>An AI-powered tool to analyse emotions from voice recordings.</p>
        """, unsafe_allow_html=True)

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

        st.markdown("""
        <p style='text-align: center; color: gray; font-size: 14px;'>*Disclaimer: Please note that this is still in Phase 1 beta stage. Thank you for being patient with us.*</p>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
