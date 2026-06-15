import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Add current dir to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.preprocessing import load_and_preprocess_audio, extract_features

# Set Page Config
st.set_page_config(
    page_title="VoiceShield - Deepfake Audio Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #020617 100%);
        color: #f1f5f9;
    }
    
    /* Gradient Title */
    .title-text {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        background: linear-gradient(to right, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem;
        margin-bottom: 0px;
        text-align: center;
        letter-spacing: -0.05rem;
    }
    
    .subtitle-text {
        font-family: 'Outfit', sans-serif;
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Glassmorphism Card */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }
    
    /* Results Styling */
    .result-badge {
        font-size: 1.8rem;
        font-weight: 700;
        padding: 10px 24px;
        border-radius: 12px;
        display: inline-block;
        margin-top: 10px;
        text-align: center;
    }
    .badge-genuine {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-fake {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Buttons Customization */
    div.stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
        padding: 10px 28px !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #64748b;
        font-size: 0.9rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 4rem;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("<p class='title-text'>VoiceShield AI</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Deepfake Voice Detection & Anti-Spoofing Guard</p>", unsafe_allow_html=True)

# Setup models paths
MODEL_PATH = "models/detector_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Lazy-load HF pipeline to optimize start speed
@st.cache_resource
def load_hf_model():
    from transformers import pipeline
    return pipeline("audio-classification", model="MelodyMachine/Deepfake-audio-detection")

def predict_custom(file_path):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
        
    y = load_and_preprocess_audio(file_path)
    if y is None:
        return None
    feats = extract_features(y)
    if feats is None:
        return None
        
    feats_scaled = scaler.transform(feats.reshape(1, -1))
    prediction = model.predict(feats_scaled)[0]
    probabilities = model.predict_proba(feats_scaled)[0]
    
    label = "Genuine (Human)" if prediction == 0 else "Deepfake (AI-Generated)"
    confidence = probabilities[prediction]
    return label, confidence, feats

# Layout - Columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🛠️ Input Control Panel")
    
    # Model selection
    model_choice = st.radio(
        "Select Classification Engine:",
        ["Lightweight Feature-Based MLP (Local)", "Deep-Learning Foundation Model (HF WavLM)"],
        help="WavLM runs on pre-trained transformers weights, ideal for real voices. MLP classifier is trained on spectral attributes."
    )
    
    st.markdown("---")
    
    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload speech recording (.wav, .mp3, .flac)",
        type=["wav", "mp3", "flac"]
    )
    
    st.markdown("<h5 style='text-align: center; color: #64748b;'>-- OR --</h5>", unsafe_allow_html=True)
    
    # Select pre-generated samples
    sample_choice = st.selectbox(
        "Test with simulated audio samples:",
        ["-- Select a Sample --", "Genuine voice sample 1", "Genuine voice sample 2", "AI Deepfake sample 1", "AI Deepfake sample 2"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Run classification trigger
run_classification = False
audio_file_path = None

# Set path depending on upload vs sample selection
if uploaded_file is not None:
    # Save temporary file
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    audio_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(audio_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Play Audio
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🎧 Play Uploaded Audio")
        st.audio(uploaded_file)
        if st.button("Run Shield Analysis"):
            run_classification = True
        st.markdown('</div>', unsafe_allow_html=True)

elif sample_choice != "-- Select a Sample --":
    sample_mapping = {
        "Genuine voice sample 1": "data/simulated/genuine_sample_1.wav",
        "Genuine voice sample 2": "data/simulated/genuine_sample_2.wav",
        "AI Deepfake sample 1": "data/simulated/deepfake_sample_1.wav",
        "AI Deepfake sample 2": "data/simulated/deepfake_sample_2.wav"
    }
    audio_file_path = sample_mapping[sample_choice]
    
    # Play Audio
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🎧 Play Simulated Sample")
        st.audio(audio_file_path)
        if st.button("Run Sample Analysis"):
            run_classification = True
        st.markdown('</div>', unsafe_allow_html=True)

# Output results section
with col2:
    st.markdown('<div class="glass-card" style="min-height: 480px;">', unsafe_allow_html=True)
    st.markdown("### 📊 Shield Diagnostics Output")
    
    if run_classification and audio_file_path:
        with st.spinner("Analyzing spectral dynamics..."):
            if "MLP" in model_choice:
                # Custom MLP Model inference
                res = predict_custom(audio_file_path)
                if res:
                    label, confidence, features = res
                    
                    # Display Badge
                    if "Genuine" in label:
                        st.markdown(f'<div class="result-badge badge-genuine">🛡️ {label}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-badge badge-fake">⚠️ {label}</div>', unsafe_allow_html=True)
                        
                    st.markdown(f"#### Classification Confidence: **{confidence*100:.2f}%**")
                    st.progress(confidence)
                    
                    st.markdown("---")
                    st.markdown("#### 📉 Audio Spectral Features")
                    # Plot MFCC values
                    mfcc_vals = features[:20]
                    fig, ax = plt.subplots(figsize=(6, 2.5), facecolor='none')
                    ax.set_facecolor('none')
                    ax.bar(range(1, 21), mfcc_vals, color='#6366f1', edgecolor='none')
                    ax.set_xlabel('MFCC Coefficients', color='#94a3b8', fontsize=8)
                    ax.set_ylabel('Mean Amplitude', color='#94a3b8', fontsize=8)
                    ax.tick_params(colors='#94a3b8', labelsize=7)
                    ax.grid(color=(1.0, 1.0, 1.0, 0.05), linestyle='--')
                    fig.patch.set_alpha(0.0)
                    st.pyplot(fig)
                else:
                    st.error("Audio processing failed. Please ensure the file is a valid 16kHz audio recording.")
                    
            else:
                # Foundation Hugging Face WavLM model inference
                try:
                    classifier = load_hf_model()
                    results = classifier(audio_file_path)
                    
                    # Parse labels
                    top_result = results[0]
                    score = top_result['score']
                    hf_label = top_result['label'].lower()
                    
                    label = "Genuine (Human)" if 'real' in hf_label or 'bonafide' in hf_label else "Deepfake (AI-Generated)"
                    
                    if "Genuine" in label:
                        st.markdown(f'<div class="result-badge badge-genuine">🛡️ {label}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-badge badge-fake">⚠️ {label}</div>', unsafe_allow_html=True)
                        
                    st.markdown(f"#### Classification Confidence: **{score*100:.2f}%**")
                    st.progress(score)
                    
                    st.markdown("---")
                    st.markdown("#### 🔄 Class Probabilities")
                    # Show probabilities in dataframe
                    prob_df = pd.DataFrame(results)
                    prob_df.columns = ["Class Label", "Probability"]
                    prob_df["Class Label"] = prob_df["Class Label"].apply(lambda x: "Genuine (Human)" if "real" in x.lower() else "Deepfake (AI-Generated)")
                    st.dataframe(prob_df, hide_index=True, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Failed to load foundation model: {e}")
                    st.info("Check internet connectivity. You can fall back to the local MLP classifier.")
    else:
        st.markdown(
            "<div style='text-align: center; padding-top: 100px; color: #64748b;'>"
            "🚀 Upload an audio file or select a simulated sample and click 'Run Analysis' to show diagnostics."
            "</div>",
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(
    "<div class='footer'>"
    "VoiceShield AI • Machine Learning for Audio Forensics & Synthetic Speech Analysis • Accuracy: 87.9% | EER: 11.6%"
    "</div>",
    unsafe_allow_html=True
)
