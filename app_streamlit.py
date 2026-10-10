#!/usr/bin/env python3
# app_streamlit.py — UI web alternatif menggunakan Streamlit
# Jalankan: streamlit run app_streamlit.py

import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image

from config import ACTIVE_MODEL, CONFIDENCE_THRESHOLD, SIBI_LABELS
from utils.hand_detector      import get_detector
from utils.feature_extractor   import extract_features, crop_hand_region
from utils.predictor           import GesturePredictor

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "SIBI Recognition",
    page_icon  = "🤟",
    layout     = "wide",
)

st.title("🤟 SIBI Recognition")
st.caption("Bahasa Isyarat Indonesia — tanpa TensorFlow / MediaPipe / Pygame")

# ── Sidebar kontrol ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan")
    model_choice = st.selectbox(
        "Model", ["random_forest", "cnn", "lstm"],
        index=["random_forest", "cnn", "lstm"].index(ACTIVE_MODEL))
    threshold = st.slider(
        "Confidence Threshold", 0.3, 0.99,
        float(CONFIDENCE_THRESHOLD), 0.05)
    use_tts = st.checkbox("Aktifkan TTS (pyttsx3)", value=False)
    st.markdown("---")
    st.markdown("**Kontrol kata:**")
    col_a, col_b = st.columns(2)
    clear_btn  = col_a.button("🗑️ Bersihkan")
    speak_btn  = col_b.button("🔊 Ucapkan")

# ── State ─────────────────────────────────────────────────────────────────────
if "word" not in st.session_state:
    st.session_state.word = ""
if "detector" not in st.session_state:
    st.session_state.detector  = get_detector()
if "predictor" not in st.session_state:
    st.session_state.predictor = GesturePredictor(model_choice)
if "last_label" not in st.session_state:
    st.session_state.last_label = ""
if "hold_count" not in st.session_state:
    st.session_state.hold_count = 0

# ── Clear / Speak ─────────────────────────────────────────────────────────────
if clear_btn:
    st.session_state.word = ""
if speak_btn and use_tts and st.session_state.word.strip():
    from utils.tts_engine import get_tts
    tts = get_tts()
    tts.speak(st.session_state.word.strip())

# ── Layout ────────────────────────────────────────────────────────────────────
col_cam, col_info = st.columns([2, 1])

with col_cam:
    st.subheader("Kamera")
    frame_placeholder = st.empty()

with col_info:
    st.subheader("Prediksi")
    pred_placeholder = st.empty()
    conf_placeholder = st.empty()
    word_placeholder = st.empty()
    st.markdown("---")
    st.subheader("Referensi Huruf SIBI")
    st.info(
        "A B C D E F G H I J K L M\n"
        "N O P Q R S T U V W X Y Z"
    )

# ── Kamera loop ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

HOLD_NEEDED = 10
last_stable = ""

run = st.checkbox("▶️ Jalankan Kamera", value=True)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Kamera tidak terdeteksi.")
        break

    frame = cv2.flip(frame, 1)
    hands = st.session_state.detector.detect(frame)
    frame = st.session_state.detector.draw(frame, hands)

    label, conf = None, 0.0

    if hands:
        hand     = hands[0]
        lms      = hand["landmarks"]
        bbox     = hand["bbox"]
        features = extract_features(lms)
        label, conf = st.session_state.predictor.predict(features)

        if label == st.session_state.last_label:
            st.session_state.hold_count += 1
        else:
            st.session_state.hold_count = 0
            st.session_state.last_label = label or ""

        if (label and
                st.session_state.hold_count >= HOLD_NEEDED and
                label != last_stable):
            st.session_state.word += label
            last_stable = label
            st.session_state.hold_count = 0
    else:
        st.session_state.last_label = ""
        st.session_state.hold_count = 0
        last_stable = ""

    # Tampilkan frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

    # Panel info
    if label:
        pred_placeholder.metric("Huruf Terdeteksi", label)
        conf_placeholder.progress(float(conf),
                                   text=f"Confidence: {conf:.1%}")
    else:
        pred_placeholder.metric("Huruf Terdeteksi", "—")
        conf_placeholder.progress(0.0, text="Menunggu tangan...")

    word_placeholder.markdown(f"### 📝 Kata:\n`{st.session_state.word or '(kosong)'}`")

    time.sleep(1 / 30)

cap.release()
