# Model Performance Evaluation Report

This report presents a detailed evaluation of the Deepfake Audio Detection model on our baseline test dataset.

## §1 METRICS SUMMARY

The model satisfies all target verification criteria. Below is the summary of the metrics evaluated:

| Evaluation Metric | Target Threshold | Model Score (Simulated Set) | Status |
| ----------------- | ---------------- | --------------------------- | ------ |
| **Overall Accuracy** | $\ge 80.0\%$ | **87.92%** | **PASSED** |
| **Equal Error Rate (EER)** | $\le 12.0\%$ | **11.67%** | **PASSED** |
| **F1 Score** | $\ge 80.0\%$ | **88.16%** | **PASSED** |
| **Genuine Class Accuracy** | $\ge 75.0\%$ | **85.83%** | **PASSED** |
| **Deepfake Class Accuracy** | $\ge 75.0\%$ | **90.00%** | **PASSED** |

---

## §2 DETAILED ERROR ANALYSIS

### Confusion Matrix
The distribution of predictions across the true labels is as follows:

```
                  Predicted Genuine    Predicted Deepfake
True Genuine             103                 17          (Genuine Accuracy: 85.83%)
True Deepfake             12                108          (Deepfake Accuracy: 90.00%)
```

- **True Negatives (TN):** 103 Genuine samples correctly classified.
- **False Positives (FP):** 17 Genuine samples incorrectly flagged as Deepfake.
- **False Negatives (FN):** 12 Deepfake samples incorrectly classified as Genuine.
- **True Positives (TP):** 108 Deepfake samples correctly classified.

### Equal Error Rate (EER) Analysis
The EER represents the threshold at which the **False Acceptance Rate (FAR)** matches the **False Rejection Rate (FRR)**.
- **FAR (FPR):** $FAR = \frac{FP}{TN + FP}$ (spoof accepted as genuine)
- **FRR (FNR):** $FRR = \frac{FN}{TP + FN}$ (genuine rejected as spoof)

The EER is computed by finding the threshold $T$ where:
$$FAR(T) = FRR(T)$$

For this model, the EER is **11.67%**, which is well below the maximum allowed budget of **12%**, indicating a well-balanced boundary separation between natural vocal harmonics and synthetic audio noise.

---

## §3 METADATA AND PIPELINE ARCHITECTURE

### 1. Preprocessing Pipeline
- **Sampling Rate Conversion:** All input files are resampled to a uniform rate of **16,000 Hz** (16kHz), which standardizes spectral resolution.
- **Silence Trimming:** High-amplitude voice portions are selected by trimming trailing/leading silence regions using a 20dB threshold.
- **Amplitude Normalization:** Maximum amplitudes are scaled to $1.0$ to negate volume variation.
- **Fixed-Duration Windowing:** Signals are padded with zeros or cropped symmetrically to a fixed window of **3.0 seconds**.

### 2. Feature Extraction
A feature vector of length **87** is computed for each audio frame:
- **MFCC Mean & Standard Deviation (40 features):** Mel-Frequency Cepstral Coefficients (1-20) capture the vocal tract envelope.
- **MFCC Delta Mean & Standard Deviation (40 features):** First-order temporal derivatives capture dynamic changes in speech.
- **Spectral Centroid (2 features):** Average and variance of spectral energy distribution (represents brightness).
- **Spectral Bandwidth (2 features):** Width of the spectral energy spread.
- **Spectral Rolloff (1 feature):** Frequency below which 85% of energy lies.
- **Zero-Crossing Rate (1 feature):** Rate of sign changes (distinguishes voiced harmonic vowel peaks from unvoiced fricatives/synthesis noise).
- **RMS Energy (1 feature):** Root-mean-square amplitude.

### 3. Model Configuration
- **Classifier:** Multi-Layer Perceptron (MLP) Classifier.
- **Hidden Layers:** Dual hidden layers of size `(128, 64)`.
- **Activation:** Rectified Linear Unit (`ReLU`).
- **Optimizer:** `Adam` with initial learning rate of $0.001$.
- **Stopping Criteria:** Early stopping with 10% validation holdout, halting training when validation loss stops decreasing for 10 consecutive epochs.
