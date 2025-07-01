# app/analysis.py

import librosa
import numpy as np
from librosa.sequence import dtw
from scipy.spatial.distance import cdist

def extract_features(y, sr):
    """
    Extract relevant audio features for comparison.
    Returns a dictionary of features.
    """
    features = {}

    # --- Pitch (F0 via pyin) ---
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                            fmax=librosa.note_to_hz('C7'))
    features['pitch'] = f0

    # --- MFCCs ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfcc'] = mfccs

    # --- Chroma (harmonic structure) ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma'] = chroma

    # --- Spectral Contrast (timbre & brightness) ---
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['contrast'] = contrast

    return features


def compare_features(ref_feat, stu_feat):
    """
    Compare features using DTW and return similarity scores.
    """
    results = {}

    # --- Pitch DTW ---
    pitch_ref = np.nan_to_num(ref_feat['pitch'])
    pitch_stu = np.nan_to_num(stu_feat['pitch'])
    D_pitch, wp_pitch = dtw(pitch_ref.reshape(1, -1), pitch_stu.reshape(1, -1))
    results['pitch_dtw'] = D_pitch[-1, -1]

    # --- MFCC DTW ---
    D_mfcc, wp_mfcc = dtw(ref_feat['mfcc'], stu_feat['mfcc'])
    results['mfcc_dtw'] = D_mfcc[-1, -1]

    # --- Chroma DTW ---
    D_chroma, _ = dtw(ref_feat['chroma'], stu_feat['chroma'])
    results['chroma_dtw'] = D_chroma[-1, -1]

    # --- Spectral Contrast DTW ---
    D_contrast, _ = dtw(ref_feat['contrast'], stu_feat['contrast'])
    results['contrast_dtw'] = D_contrast[-1, -1]

    return results, wp_pitch
