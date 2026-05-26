import librosa
import numpy as np

def extract_mfcc(file_path):

    # Load audio
    audio, sr = librosa.load(file_path, sr=16000)

    # Remove silence
    audio, _ = librosa.effects.trim(audio)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    # Transpose shape
    mfcc = mfcc.T

    return mfcc