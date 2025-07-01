import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tempfile
from app.analysis import extract_features, compare_features
from app.feedback import generate_feedback
from pydub import AudioSegment

# ==============================
# 🔧 Helper: Load Any Audio File
# ==============================
def load_audio_file(file):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        audio = AudioSegment.from_file(file)
        audio.export(tmp_wav.name, format="wav")
        y, sr = librosa.load(tmp_wav.name, sr=None)
    return y, sr

# ==========================
# 🎵 Streamlit App Interface
# ==========================
st.set_page_config(page_title="Music LMS", layout="wide")
st.title("🎶 Music LMS - Offline Audio Analyzer")
st.markdown("Upload a reference audio and a student recording to compare pitch, tone, and timing.")

ref_file = st.file_uploader("Upload Reference Audio", type=["wav", "mp3", "opus", "m4a"])
stu_file = st.file_uploader("Upload Student Audio", type=["wav", "mp3", "opus", "m4a"])

if ref_file and stu_file:
    # ========================
    # 🎧 Load & Display Audio
    # ========================
    y_ref, sr_ref = load_audio_file(ref_file)
    y_stu, sr_stu = load_audio_file(stu_file)

    st.subheader("🎧 Waveform Comparison")
    fig, ax = plt.subplots(2, 1, figsize=(10, 4))
    librosa.display.waveshow(y_ref, sr=sr_ref, ax=ax[0])
    ax[0].set(title="Reference")
    librosa.display.waveshow(y_stu, sr=sr_stu, ax=ax[1])
    ax[1].set(title="Student")
    st.pyplot(fig)

    # ========================
    # 🧠 Feature Extraction
    # ========================
    st.subheader("🧠 Feature Extraction")
    ref_feat = extract_features(y_ref, sr_ref)
    stu_feat = extract_features(y_stu, sr_stu)

    # ============================
    # 📈 Pitch Contour Comparison
    # ============================
    st.subheader("📈 Pitch Contour Comparison")
    times_ref = librosa.times_like(ref_feat['pitch'], sr=sr_ref)
    times_stu = librosa.times_like(stu_feat['pitch'], sr=sr_stu)

    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(times_ref, ref_feat['pitch'], label='Reference', color='blue')
    ax2.plot(times_stu, stu_feat['pitch'], label='Student', color='red')
    ax2.set(title="Pitch Contour", xlabel="Time (s)", ylabel="Hz")
    ax2.legend()
    st.pyplot(fig2)

    # ========================
    # 🔁 Feature Comparison
    # ========================
    comparison, _ = compare_features(ref_feat, stu_feat)

    # ========================
    # 📊 Score Metrics
    # ========================
    st.subheader("📊 Comparison Scores")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Pitch DTW", f"{comparison['pitch_dtw']:.2f}")
    col2.metric("🎵 MFCC DTW", f"{comparison['mfcc_dtw']:.2f}")
    col3.metric("🎼 Chroma DTW", f"{comparison['chroma_dtw']:.2f}")
    col4.metric("✨ Contrast DTW", f"{comparison['contrast_dtw']:.2f}")

    # ========================
    # 📝 Feedback
    # ========================
    st.subheader("📝 Feedback")
    feedback = generate_feedback(comparison)
    for line in feedback:
        st.markdown(f"- {line}")

    # ============================
    # 🎼 MFCC Visualization (Optional)
    # ============================
    st.subheader("🎼 MFCC Spectrograms")
    fig3, ax3 = plt.subplots(2, 1, figsize=(10, 6))
    librosa.display.specshow(ref_feat['mfcc'], x_axis='time', sr=sr_ref, ax=ax3[0])
    ax3[0].set(title="Reference MFCCs")
    librosa.display.specshow(stu_feat['mfcc'], x_axis='time', sr=sr_stu, ax=ax3[1])
    ax3[1].set(title="Student MFCCs")
    st.pyplot(fig3)

else:
    st.info("👆 Please upload both reference and student audio files to begin.")
