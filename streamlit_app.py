import streamlit as st
import librosa
import numpy as np
import joblib

# Load the pretrained Random Forest model and LabelEncoder
MODEL_PATH = "rf_model.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"

@st.cache_resource
def load_model():
    rf_model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return rf_model, label_encoder

rf_model, label_encoder = load_model()

# Feature extraction function
def extract_features(file_path, target_sr=16000):
    try:
        audio_data, fs = librosa.load(file_path, sr=None)
        if audio_data.ndim == 2:
            audio_data = np.mean(audio_data, axis=1)  # Convert stereo to mono

        # Apply pre-emphasis filter
        pre_emphasis = [1, -0.97]
        audio_filtered = np.convolve(audio_data, pre_emphasis, mode="same")

        # Resample audio
        audio_resampled = librosa.resample(audio_filtered, orig_sr=fs, target_sr=target_sr)

        # Frame the audio
        frame_size = int(0.025 * target_sr)
        overlap = int(0.80 * frame_size)
        framed_audio = librosa.util.frame(audio_resampled, frame_length=frame_size, hop_length=frame_size - overlap)

        # Calculate the standard deviation for each frame
        frame_std = np.std(framed_audio, axis=0)
        mean_std = np.mean(frame_std)

        # Threshold to keep high-variance frames
        threshold = 0.9 * mean_std
        non_silent_frames = framed_audio[:, frame_std > threshold]
        processed_audio = non_silent_frames.flatten()

        # Short-Time Fourier Transform
        hop_length = int(0.01 * target_sr)
        n_fft = 512
        S = np.abs(librosa.stft(processed_audio, n_fft=n_fft, hop_length=hop_length))

        # MFCCs and deltas
        mfccs = librosa.feature.mfcc(y=processed_audio, sr=target_sr, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta_delta_mfccs = librosa.feature.delta(mfccs, order=2)

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=target_sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=target_sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=target_sr)
        spectral_flux = np.mean(np.diff(S, axis=1) ** 2, axis=0)
        spectral_contrast = librosa.feature.spectral_contrast(S=S, sr=target_sr)
        spectral_flatness = librosa.feature.spectral_flatness(S=S)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(processed_audio, hop_length=hop_length)

        # Pad or truncate spectral flux for dimension matching
        spectral_flux = np.pad(spectral_flux, (0, S.shape[1] - spectral_flux.shape[0]), mode="edge")
        num_frames = S.shape[1]
        spectral_centroid = spectral_centroid[:, :num_frames]
        spectral_bandwidth = spectral_bandwidth[:, :num_frames]
        spectral_rolloff = spectral_rolloff[:, :num_frames]
        spectral_contrast = spectral_contrast[:, :num_frames]
        spectral_flatness = spectral_flatness[:, :num_frames]
        zero_crossing_rate = zero_crossing_rate[:, :num_frames]

        # Combine spectral features
        spectral_features = np.concatenate([
            spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flux.reshape(1, -1),
            spectral_contrast, spectral_flatness, zero_crossing_rate
        ], axis=0)

        # Combine MFCCs, deltas, and spectral features
        combined_features = np.concatenate([mfccs, delta_mfccs, delta_delta_mfccs, spectral_features], axis=0)

        # Average features across frames
        avg_features = np.mean(combined_features, axis=1)
        return avg_features
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None

# Add background image using CSS
def add_background_image(image_file):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/png;base64,{image_file});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        /* Customizing Fonts */
        h1, h2, h3, h4, h5, h6 {{
            color: #DFFF00;  /* Darker blue for headings */
            font-size: 2.2em;  /* Larger headings */
            font-weight: bold;
        }}
        p, label, span {{
            color: #DFFF00;  /* Slightly lighter blue for other text */
            font-size: 1.2em;  /* Bigger font for paragraph and labels */
        }}
        .stFileUploader {{
            color: #DFFF00;  /* Blue shade for upload widget */
            font-size: 1.1em;
        }}
        .stAudio {{
            margin-top: 20px;
            margin-bottom: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Load and encode the background image
import base64
with open("background.jpg", "rb") as img_file:
    bg_image = base64.b64encode(img_file.read()).decode()

add_background_image(bg_image)
# Streamlit app interface
st.title("****Cough Type Classification****")
st.write("**Upload an audio file to classify cough type.**")
st.write("**(আপনার কাশির ফাইলটি নিচে আপলোড করুন।)**")

# Allow MP3, WAV, and M4A files
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

st.write("\n📜📜Instruction: (দিকনির্দেশনা:) ")
st.write("**Please record your cough in a noiseless background.**")
st.write("**(অনুগ্রহ করে কাশির শব্দ নীরব পরিবেশে রেকর্ড করুন।)**")

if uploaded_file:
    st.audio(uploaded_file, format="audio/mp3")  # Update to show MP3/WAV/M4A
    st.write("Processing the audio file...")

    # Save the uploaded file temporarily
    temp_file_path = f"temp_audio_file.{uploaded_file.name.split('.')[-1]}"  # Preserve file extension
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract features and make predictions
    features = extract_features(temp_file_path)

    if features is not None:
        # Predict the class
        features = features.reshape(1, -1)
        prediction = rf_model.predict(features)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # Display prediction result
        if predicted_label.lower() == "p":
            st.success("Prediction: Pertussis cough detected.")
        else:
            st.info("Prediction: Non-Pertussis cough detected.")
    else:
        st.error("Feature extraction failed. Please try again.")
