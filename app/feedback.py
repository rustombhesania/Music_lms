# app/feedback.py

def generate_feedback(results):
    """
    Generate feedback strings based on DTW distances.
    Lower DTW = better match.
    """
    feedback = []

    # --- Pitch ---
    pitch_score = results.get("pitch_dtw", 0)
    if pitch_score < 100:
        feedback.append("✅ Great pitch alignment with the reference.")
    elif pitch_score < 300:
        feedback.append("⚠️ Pitch is somewhat off. Try tuning your notes more carefully.")
    else:
        feedback.append("❌ Pitch mismatch detected. Consider practicing intonation.")

    # --- MFCC (Timbre) ---
    mfcc_score = results.get("mfcc_dtw", 0)
    if mfcc_score < 1000:
        feedback.append("✅ Good timbre and tone quality.")
    elif mfcc_score < 3000:
        feedback.append("⚠️ Timbre is a bit different. Try matching tone quality more closely.")
    else:
        feedback.append("❌ Significant difference in tone. Check your instrument setup or technique.")

    # --- Chroma (Harmony) ---
    chroma_score = results.get("chroma_dtw", 0)
    if chroma_score < 50:
        feedback.append("✅ Harmonic content closely matches the reference.")
    else:
        feedback.append("⚠️ Chords or harmonic structure differ from the reference.")

    # --- Spectral Contrast (Articulation / Brightness) ---
    contrast_score = results.get("contrast_dtw", 0)
    if contrast_score < 50:
        feedback.append("✅ Brightness and articulation are well matched.")
    else:
        feedback.append("⚠️ Articulation or tone balance could be improved.")

    return feedback
