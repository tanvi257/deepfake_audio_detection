import os
import sys
import argparse
import pickle
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import custom modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.preprocessing import load_and_preprocess_audio, extract_features

def test_custom_model(audio_path, model_path, scaler_path):
    """
    Inference using the custom trained MLP classifier.
    """
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Error: Custom model weights ({model_path}) or scaler ({scaler_path}) are missing.", file=sys.stderr)
        print("Please run 'python train.py' first to train and save the custom model.", file=sys.stderr)
        sys.exit(1)
        
    # Load model and scaler
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    # Load and preprocess audio
    print(f"[*] Preprocessing audio file: {audio_path}")
    y = load_and_preprocess_audio(audio_path)
    if y is None:
        print("Error: Could not load or preprocess audio.", file=sys.stderr)
        sys.exit(1)
        
    # Extract features
    print("[*] Extracting acoustic features (MFCCs + Spectral statistics)...")
    features = extract_features(y)
    if features is None:
        print("Error: Feature extraction failed.", file=sys.stderr)
        sys.exit(1)
        
    # Scale and predict
    features_scaled = scaler.transform(features.reshape(1, -1))
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Genuine = 0, Deepfake = 1
    label = "Deepfake (AI-Generated)" if prediction == 1 else "Genuine (Human)"
    confidence = probabilities[prediction]
    
    return label, confidence, probabilities

def test_foundation_model(audio_path):
    """
    Inference using state-of-the-art pre-trained deepfake audio detection model on Hugging Face.
    """
    try:
        from transformers import pipeline
    except ImportError:
        print("Error: The 'transformers' library is required for the foundation model. Run 'pip install transformers' first.", file=sys.stderr)
        sys.exit(1)
        
    print("[*] Loading pre-trained foundation model from Hugging Face (MelodyMachine/Deepfake-audio-detection)...")
    print("[*] Note: The first run will download model weights (~300MB).")
    
    try:
        classifier = pipeline("audio-classification", model="MelodyMachine/Deepfake-audio-detection")
        
        print(f"[*] Running inference on: {audio_path}")
        results = classifier(audio_path)
        
        # Parse results
        # Format: [{'label': 'real', 'score': 0.99}, {'label': 'fake', 'score': 0.01}]
        top_result = results[0]
        score = top_result['score']
        hf_label = top_result['label'].lower()
        
        label = "Genuine (Human)" if 'real' in hf_label or 'bonafide' in hf_label else "Deepfake (AI-Generated)"
        
        return label, score, results
    except Exception as e:
        print(f"Error executing foundation model: {e}", file=sys.stderr)
        print("Falling back to local custom model classification...", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description="Test speech recordings for Deepfake synthetic audio")
    parser.add_argument('--audio', type=str, required=True, help="Path to input audio file (.wav, .mp3, .flac)")
    parser.add_argument('--model', type=str, choices=['custom', 'foundation'], default='custom', 
                        help="Model backend to use (custom: local trained MLP, foundation: pre-trained WavLM)")
    parser.add_argument('--model_path', type=str, default="models/detector_model.pkl", help="Path to custom model file")
    parser.add_argument('--scaler_path', type=str, default="models/scaler.pkl", help="Path to custom scaler file")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found at path: {args.audio}", file=sys.stderr)
        sys.exit(1)
        
    print("\n" + "="*40)
    print("      DEEPFAKE AUDIO DETECTION CLI")
    print("="*40)
    print(f"File Path: {args.audio}")
    print(f"Backend:   {args.model.upper()}")
    
    result_label = None
    confidence = 0.0
    
    if args.model == 'foundation':
        res = test_foundation_model(args.audio)
        if res:
            result_label, confidence, raw_res = res
            
    # Fallback to custom if foundation failed or custom selected
    if result_label is None:
        result_label, confidence, probabilities = test_custom_model(args.audio, args.model_path, args.scaler_path)
        print(f"Probabilities - Genuine: {probabilities[0]*100:.2f}%, Deepfake: {probabilities[1]*100:.2f}%")
        
    print("-"*40)
    print(f"Prediction: {result_label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("="*40 + "\n")

if __name__ == '__main__':
    main()
