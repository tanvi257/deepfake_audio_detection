import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# Import our custom modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.preprocessing import load_and_preprocess_audio, extract_features
from src.metrics import evaluate_predictions

def generate_genuine_waveform(t, f0):
    """
    Generates a natural speech-like waveform with pitch modulation and clean harmonics.
    """
    # Pitch modulation (intonation)
    mod = np.sin(2 * np.pi * 4 * t) * 6.0
    y = (
        0.5 * np.sin(2 * np.pi * f0 * t + mod) + 
        0.25 * np.sin(2 * np.pi * 2 * f0 * t + 2 * mod) + 
        0.15 * np.sin(2 * np.pi * 3 * f0 * t + 3 * mod)
    )
    # Add ambient noise (SNR ~ 35dB)
    y += np.random.normal(0, 0.01, len(t))
    max_val = np.max(np.abs(y))
    return y / max_val if max_val > 0 else y

def generate_deepfake_waveform(t, f0, sig_type=None):
    """
    Generates a synthetic speech-like waveform with vocoder buzz, 
    robotic ring modulations, or sawtooth/square distortions.
    """
    if sig_type is None:
        sig_type = np.random.choice(['sawtooth', 'square', 'robotic'])
        
    if sig_type == 'sawtooth':
        y = (t * f0 % 1.0 - 0.5) * 0.4
    elif sig_type == 'square':
        y = np.sign(np.sin(2 * np.pi * f0 * t)) * 0.3
    else: # Metallic/robotic ring modulation
        y = np.sin(2 * np.pi * f0 * t + 8.0 * np.sin(2 * np.pi * 24 * t)) * 0.4
        
    # Add high-frequency vocoder noise/buzz (SNR ~ 15dB)
    y += np.random.normal(0, 0.05, len(t))
    max_val = np.max(np.abs(y))
    return y / max_val if max_val > 0 else y

def generate_mock_wav_files(output_dir):
    """
    Generates dummy wav files for manual testing using our unified generators.
    """
    os.makedirs(output_dir, exist_ok=True)
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # 1. Genuine Sample 1 (Pitch = 200 Hz)
    y_gen1 = generate_genuine_waveform(t, 200.0)
    sf.write(os.path.join(output_dir, 'genuine_sample_1.wav'), y_gen1, sr)
    
    # 2. Genuine Sample 2 (Pitch = 150 Hz)
    y_gen2 = generate_genuine_waveform(t, 150.0)
    sf.write(os.path.join(output_dir, 'genuine_sample_2.wav'), y_gen2, sr)
    
    # 3. Deepfake Sample 1 (Robotic modulation, Pitch = 200 Hz)
    y_fake1 = generate_deepfake_waveform(t, 200.0, sig_type='robotic')
    sf.write(os.path.join(output_dir, 'deepfake_sample_1.wav'), y_fake1, sr)
    
    # 4. Deepfake Sample 2 (Sawtooth vocoder, Pitch = 150 Hz)
    y_fake2 = generate_deepfake_waveform(t, 150.0, sig_type='sawtooth')
    sf.write(os.path.join(output_dir, 'deepfake_sample_2.wav'), y_fake2, sr)
    
    print(f"Generated 4 mock WAV files in {output_dir}")

def load_real_dataset(data_dir):
    """
    Scans data_dir for wav files and extracts features.
    Assumes subdirectory naming or metadata file for labels:
    - Files under 'real' / 'genuine' or labeled 'bonafide' -> 0
    - Files under 'fake' / 'spoof' or labeled 'spoof' -> 1
    """
    X = []
    y = []
    
    print(f"Scanning directory: {data_dir} for audio files...")
    
    # Check for ASVspoof-style protocol file
    protocol_files = [f for f in os.listdir(data_dir) if 'protocol' in f.lower() or 'cm.train' in f.lower()] if os.path.exists(data_dir) else []
    
    if protocol_files:
        protocol_path = os.path.join(data_dir, protocol_files[0])
        print(f"Found protocol file: {protocol_path}")
        # Parse protocol file: filename label
        df = pd.read_csv(protocol_path, sep=r'\s+', header=None)
        # Usually filenames are in col 1 or 2, labels are in the last col ('bonafide'/'spoof')
        for _, row in df.iterrows():
            filename = None
            label = None
            for val in row:
                val_str = str(val)
                if val_str.endswith('.wav') or val_str.endswith('.flac'):
                    filename = val_str
                elif val_str in ['bonafide', 'genuine', 'real']:
                    label = 0
                elif val_str in ['spoof', 'fake', 'deepfake']:
                    label = 1
            
            if filename and label is not None:
                for root, _, files in os.walk(data_dir):
                    if filename in files:
                        file_path = os.path.join(root, filename)
                        features = extract_features(load_and_preprocess_audio(file_path))
                        if features is not None:
                            X.append(features)
                            y.append(label)
                            
    # Directory-based scanning
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac')):
                file_path = os.path.join(root, file)
                path_lower = file_path.lower()
                label = None
                if 'real' in path_lower or 'genuine' in path_lower or 'bonafide' in path_lower:
                    label = 0
                elif 'fake' in path_lower or 'spoof' in path_lower or 'deepfake' in path_lower:
                    label = 1
                    
                if label is not None:
                    features = extract_features(load_and_preprocess_audio(file_path))
                    if features is not None:
                        X.append(features)
                        y.append(label)
                        
    return np.array(X), np.array(y)

def generate_simulated_features(n_samples=400):
    """
    Generates a simulated dataset by synthesizing genuine-like (harmonic) 
    and deepfake-like (synthetic vocoded) waveforms in memory and extracting 
    their actual acoustic features.
    """
    print(f"Synthesizing {n_samples} audio waveforms in memory and extracting features...")
    X = []
    y = []
    
    sr = 16000
    duration = 3.0 # Match the 3.0s duration of training/inference pipeline
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    n_class = n_samples // 2
    
    # 1. Synthesize Genuine (Human-like harmonics)
    print("  -> Processing Genuine samples...")
    for i in range(n_class):
        f0 = np.random.uniform(110, 260)
        y_audio = generate_genuine_waveform(t, f0)
        feats = extract_features(y_audio, sr=sr)
        if feats is not None:
            X.append(feats)
            y.append(0)
            
    # 2. Synthesize Deepfake (Robotic/vocoded artifact signals)
    print("  -> Processing Deepfake samples...")
    for i in range(n_class):
        f0 = np.random.uniform(110, 260)
        y_audio = generate_deepfake_waveform(t, f0)
        feats = extract_features(y_audio, sr=sr)
        if feats is not None:
            X.append(feats)
            y.append(1)

            
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Train the Deepfake Audio Detection Classifier")
    parser.add_argument('--data_dir', type=str, default=r"C:\Users\TANVI\Downloads\LA norm", help="Path to evaluation dataset")
    parser.add_argument('--output_dir', type=str, default="models", help="Directory to save trained model")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Generate Mock WAV files for test runs
    mock_audio_dir = os.path.join("data", "simulated")
    generate_mock_wav_files(mock_audio_dir)
    
    # Step 2: Load or simulate data
    X, y = [], []
    is_simulated = True
    
    if os.path.exists(args.data_dir):
        X, y = load_real_dataset(args.data_dir)
        if len(X) > 10:
            is_simulated = False
            print(f"Loaded {len(X)} samples from real dataset.")
        else:
            print("Real dataset path found, but insufficient audio samples (.wav/.flac) were detected.")
            
    if is_simulated:
        print("Running in SIMULATION MODE to train the classifier...")
        X, y = generate_simulated_features(n_samples=1200)
        
    # Step 3: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20 if is_simulated else 0.30, random_state=42, stratify=y)
    
    # Step 4: Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_path = os.path.join(args.output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    # Step 5: Train MLP Classifier
    # High-performance neural network for tabular features
    print("Training neural network classifier (MLP)...")
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    clf.fit(X_train_scaled, y_train)
    
    # Step 6: Evaluate Model
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    
    metrics = evaluate_predictions(y_test, y_pred, y_prob)
    
    # Print results
    print("\n" + "="*40)
    print("           TRAINING COMPLETE")
    print("="*40)
    print(f"Mode: {'Simulated Data' if is_simulated else 'Real Dataset'}")
    print(f"Overall Accuracy:  {metrics['accuracy']*100:.2f}% (Threshold: >= 80%)")
    print(f"Equal Error Rate:  {metrics['eer']*100:.2f}% (Threshold: <= 12%)")
    print(f"F1 Score:          {metrics['f1_score']*100:.2f}% (Threshold: >= 80%)")
    print(f"Genuine Accuracy:  {metrics['genuine_accuracy']*100:.2f}% (Threshold: >= 75%)")
    print(f"Deepfake Accuracy: {metrics['deepfake_accuracy']*100:.2f}% (Threshold: >= 75%)")
    print("\nConfusion Matrix:")
    print(np.array(metrics['confusion_matrix']))
    print("="*40)
    
    # Step 7: Save Model weights
    model_path = os.path.join(args.output_dir, 'detector_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved to {model_path}")
    
    # Save metrics details for performance report
    metrics_path = os.path.join(args.output_dir, 'metrics_summary.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)

if __name__ == '__main__':
    main()
