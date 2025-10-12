# Cough-Sound-Classifiction
Digital Signal Processing mini project that classifies cough sounds (Covid, Tuberculosis, Healthy) using MATLAB and DSP tools.


# DSP Mini Project — Cough Classification 🩺🎧

This project was developed as part of my Digital Signal Processing mini-project.  
The main idea is to classify different cough sounds (Covid, Tuberculosis, and Healthy) using basic DSP techniques, MATLAB and Machine Learning. The goal was to apply concepts learned in class to a real-world health-related problem.


## 📌 Project Summary

- Implemented in MATLAB
- Focused on analyzing and classifying cough audio signals
- Used signal processing steps like filtering, feature extraction, and classification
- Categorized cough sounds into:
  - Covid Cough
  - Tuberculosis Cough
  - Healthy Cough

This is a simple but practical project to explore how DSP can be used in the medical field.


## 🧰 Tools & Techniques

- **Language:** MATLAB  
- **Concepts Used:** Digital Signal Processing, Audio Feature Extraction, Random Forest Classifier  
- **Dataset:** Cough audio files collected from public datasets, a website called coughvid


## 📁 Folder Structure
DSP_Project
├── main.m # MATLAB file with full project code
└── README.md # Project documentation
we extracted key DSP features from cough audio signals to classify between COVID, Tuberculosis, and Healthy cases. We computed time-domain features like Zero Crossing Rate (ZCR) to capture signal noisiness, frequency-domain features such as Spectral Centroid and Bandwidth to characterize the energy distribution, and cepstral features using MFCCs to model vocal tract characteristics. These features together provide a compact representation of cough sounds that improves classifier accuracy.

## ⚡ How to Run

1. Open `main.m` in MATLAB.  
2. Set the dataset folder path in the code.  
3. Run the script.  
4. The program will display the predicted class for each cough sample.


## 🚀 What I Learned

- Applying DSP concepts on real audio data  
- Working with MATLAB for classification  
- Understanding basic signal features like energy and zero crossing rate  
- Structuring a mini project from scratch

## Output 
1. Dataset Summary
disp(countEachLabel(ads));


What you see: A table listing each class (Covid, Tuberculosis, Healthy) and the number of WAV files in that class.

Why it matters: It tells you if your dataset is balanced or not, which is important for classifier performance.

Example output:

Label	Count
Covid	50
Tuberculosis	40
Healthy	60
2. Classification Accuracy
disp(['Classification Accuracy: ', num2str(accuracy,'%.2f'), '%']);


What you see: A single number, e.g., Classification Accuracy: 87.50%.

What it means: The percentage of test audio files correctly classified by your Random Forest (TreeBagger) model.

Key point: Accuracy alone doesn’t tell you which classes are misclassified; that’s where the confusion chart comes in.

3. Confusion Chart
confusionchart(YTest, YPred);


What you see: A grid showing true labels vs predicted labels.

Example:

	Pred: Covid	Pred: TB	Pred: Healthy
True: Covid	45	3	2
True: TB	4	35	1
True: Healthy	0	2	58

How to read:

Rows = actual labels

Columns = predicted labels

Diagonal numbers are correct predictions; off-diagonal are misclassifications.

4. Predicted Class for a Single File
disp(['Predicted Class: ', char(YPred)]);


What you see: A single label, e.g., Predicted Class: Covid.

What it means: When you select an audio file in the interactive demo, this tells you how the model classified that specific cough.

5. DSP Visualizations

When you run the interactive demo, MATLAB opens a figure with 3 subplots:

Waveform (Time-domain)

Shows amplitude vs time.

The Zero Crossing Rate (ZCR) is displayed in the title. High ZCR → noisy signal.

FFT (Frequency-domain)

Shows magnitude vs frequency.

Red line = Spectral Centroid (average “center of mass” of frequencies)

Green dashed lines = Spectral Bandwidth (spread of frequencies)

Spectrogram (Time-Frequency)

Shows how frequency content changes over time.

Brighter areas = higher energy.

Title shows the predicted class.

6. Feature Vector Output

Inside the code, each audio file is converted into a feature vector:

feat = [zcr, specCentroid, specBandwidth, mfccMean];


Length: Depends on MFCC coefficients (13 in your code) plus 3 time/frequency features → vector of size 1 × 16.

Purpose: This vector is what the Random Forest uses to classify the cough.

Summary

Dataset summary: Shows class distribution.

Accuracy: Overall model correctness on test set.

Confusion chart: Shows which classes are confused.

Predicted class: For any single audio file.

DSP plots: Waveform, FFT, Spectrogram – visualizes the cough.

Feature vector: Numeric representation used for classification.


## 📈 Future Plans

- Adding more classes and data for better accuracy  
- Trying ML/DL models for comparison  
- Building a small GUI to make the tool more user friendly


⭐ *If you like this project, feel free to check it out, use it, or give it a star.*

