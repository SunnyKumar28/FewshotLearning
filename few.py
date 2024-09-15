import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import io

# Load the model
model = keras.models.load_model('mymodel.keras')

# Define label names (adjust these if necessary)
label_names = np.array(['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes'  ])

def preprocess_audio(file):
    # Create a file-like object from the bytes
    audio_bytes = file.read()
    audio_file = io.BytesIO(audio_bytes)
    
    # Read the audio file
    audio, _ = tf.audio.decode_wav(audio_file.read(), desired_channels=1, desired_samples=16000)
    audio = tf.squeeze(audio, axis=-1)
    spectrogram = get_spectrogram(audio)
    spectrogram = spectrogram[tf.newaxis, ...]  # Add batch dimension
    return spectrogram

def get_spectrogram(waveform):
    # Generate the spectrogram
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]  # Add channel dimension
    return spectrogram

def predict_class(audio_file):
    spectrogram = preprocess_audio(audio_file)
    prediction = model.predict(spectrogram)
    predicted_class = np.argmax(prediction)
    return label_names[predicted_class]

# Streamlit app code
def main():
    st.title('Keyword Spotting with Audio File')
    st.write('Upload an audio file to classify the keyword.')

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        # Make prediction
        predicted_class = predict_class(uploaded_file)
        st.write(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()