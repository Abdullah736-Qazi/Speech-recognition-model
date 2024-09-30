# Speech-recognition-model
Build a ML model using deep learning model LSTM  during internship from codealpha
Speech Emotion Recognition with LSTM
This project explores the use of Long Short-Term Memory (LSTM) networks for classifying human emotions from speech audio.

Dependencies
This project requires the following Python libraries:

numpy
pandas
os
seaborn
matplotlib.pyplot
librosa
IPython.display
warnings
keras
Getting Started
Install the required libraries using pip install (All Dependencies).
Download the speech emotion recognition dataset.
Modify the script to point to the downloaded dataset location.
Note: This script assumes the dataset is organized with audio files categorized by emotion in subfolders.

Data Loading
The script loads the audio files and their corresponding emotion labels from the dataset directory.

Data Analysis
The script performs some basic data analysis steps:

Creates a DataFrame containing audio file paths and emotion labels.
Visualizes the distribution of emotions using a count plot.
Defines functions for audio visualization using waveforms and spectrograms.
Demonstrates visualization examples for different emotions.
Feature Extraction
The script extracts Mel-frequency cepstral coefficients (MFCCs) as features from the audio data. MFCCs capture the spectral characteristics of the audio signal, which are relevant for emotion recognition.

Function: extract_mfcc(filename)

Loads the audio file with a specified duration and offset.
Calculates the MFCCs using librosa.
Returns the mean of MFCCs across time steps.
Feature Engineering
Applies the extract_mfcc function to all audio files in the dataset.
Reshapes the extracted MFCC features for compatibility with the LSTM model.
Uses One-Hot Encoding to transform categorical emotion labels into numerical vectors.
Model Building
The script defines and trains a deep learning model with an LSTM layer for emotion classification.

Model Architecture:

Sequential model with an LSTM layer with 128 units.
Dense layers with ReLU activation for feature extraction and classification.
Dropout layers for regularization to prevent overfitting.
Softmax activation in the final layer for multi-class classification (7 emotions).
Training:

Compiles the model with categorical cross-entropy loss, Adam optimizer, and accuracy metric.
Trains the model for 100 epochs with a batch size of 512 and validation split of 20%.

Evaluation:

The script plots the training and validation accuracy and loss curves to visualize the model's performance during training.

Plots:

Accuracy vs. Epochs: Shows the training and validation accuracy over the training process.
Loss vs. Epochs: Shows the training and validation loss over the training process.

Future Work:

Experiment with different hyperparameters ( number of epochs, LSTM units, dropout rate).
Explore other feature extraction techniques ( spectrograms, Mel spectrograms).
Evaluate the model performance with different emotion recognition datasets.
Implement model evaluation metrics beyond accuracy ( precision, recall, F1-score).
