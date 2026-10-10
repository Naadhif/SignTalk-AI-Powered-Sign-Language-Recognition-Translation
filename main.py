#!/usr/bin/env python3
# main.py — Aplikasi utama SIBI Recognition (OpenCV)
#
# Alur sesuai diagram:
#   Kamera → Deteksi tangan → Ekstraksi fitur → Pilih model
#   → Prediksi gesture → Confidence filter → Teks + TTS

import cv2
import time
import numpy as np
from collections import deque

from config import (ACTIVE_MODEL, FRAME_WIDTH, FRAME_HEIGHT, FRAME_FPS,
                    CONFIDENCE_THRESHOLD, HOLD_FRAMES,
                    COLOR_GREEN, COLOR_ORANGE, COLOR_WHITE, COLOR_RED,
                    FONT, FONT_SCALE)
from utils.hand_detector     import get_detector
from utils.feature_extractor  import extract_features, crop_hand_region
from utils.predictor          import GesturePredictor
from utils.tts_engine         import get_tts


def draw_info_panel(frame: np.ndarray,
                    label: str | None, conf: float,
                    word_buffer: str,
                    fps: float,
                    model_name: str) -> np.ndarray:
    """Gambar panel informasi di bagian bawah frame."""
    h, w = frame.shape[:2]

    # Background semi-transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 110), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Prediksi huruf saat ini
    if label:
        color = COLOR_GREEN
        txt   = f"Huruf: {label}  ({conf:.0%})"
    else:
        color = COLOR_ORANGE
        txt   = f"Menunggu...  (conf < {CONFIDENCE_THRESHOLD:.0%})"

    cv2.putText(frame, txt, (15, h - 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    # Kata yang terbentuk
    cv2.putText(frame, f"Kata : {word_buffer}",
                (15, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                COLOR_WHITE, 2)

    # FPS + model
    cv2.putText(frame,
                f"FPS:{fps:.0f}  Model:{model_name.upper()}",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (150, 150, 150), 1)

    return frame


def draw_confidence_bar(frame: np.ndarray, conf: float,
                        x: int, y: int, w: int = 150):
    """Bar confidence di samping tangan."""
    bar_h  = 12
    filled = int(w * conf)
    color  = COLOR_GREEN if conf >= CONFIDENCE_THRESHOLD else COLOR_ORANGE
    cv2.rectangle(frame, (x, y), (x + w, y + bar_h), (80, 80, 80), -1)
    cv2.rectangle(frame, (x, y), (x + filled, y + bar_h), color, -1)
    cv2.putText(frame, f"{conf:.0%}", (x + w + 5, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def main():
    detector  = get_detector()
    predictor = GesturePredictor(ACTIVE_MODEL)
    tts       = get_tts()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FRAME_FPS)

    word_buffer   : str   = ""
    hold_counter  : int   = 0
    last_label    : str   = ""
    last_conf     : float = 0.0
    last_stable   : str   = ""

    fps_deque     = deque(maxlen=30)
    prev_time     = time.time()

    print("=" * 50)
    print(" SIBI RECOGNITION — OpenCV")
    print("=" * 50)
    print(f" Model aktif : {ACTIVE_MODEL.upper()}")
    print(f" Threshold   : {CONFIDENCE_THRESHOLD:.0%}")
    print()
    print(" Kontrol:")
    print("   [SPACE]  → tambah spasi ke kata")
    print("   [BACKSP] → hapus huruf terakhir")
    print("   [ENTER]  → ucapkan kata (TTS)")
    print("   [c]      → bersihkan kata")
    print("   [m]      → ganti model (RF/CNN/LSTM)")
    print("   [q]      → keluar")
    print("=" * 50 + "\n")

    active_model = ACTIVE_MODEL

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # ── FPS ─────────────────────────────────────────────────────────
        now       = time.time()
        fps_deque.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        fps       = sum(fps_deque) / len(fps_deque)

        # ── Deteksi Tangan ──────────────────────────────────────────────
        hands = detector.detect(frame)
        frame = detector.draw(frame, hands)

        label, conf = None, 0.0

        if hands:
            hand = hands[0]
            lms  = hand["landmarks"]
            bbox = hand["bbox"]

            # ── Ekstraksi Fitur ─────────────────────────────────────────
            if active_model == "cnn":
                features = crop_hand_region(frame, bbox)
            else:
                features = extract_features(lms)

            # ── Prediksi + Confidence Filter ────────────────────────────
            label, conf = predictor.predict(features)
            last_conf   = conf

            # Gambar confidence bar di atas tangan
            x, y, w, h = bbox
            draw_confidence_bar(frame, conf, x, max(0, y - 20))

            # ── Stabilisasi HOLD_FRAMES ──────────────────────────────────
            if label == last_label:
                hold_counter += 1
            else:
                hold_counter  = 0
                last_label    = label or ""

            if label and hold_counter >= HOLD_FRAMES and label != last_stable:
                last_stable  = label
                word_buffer += label
                hold_counter = 0
                print(f"  + '{label}'  (kata: {word_buffer})")

        else:
            hold_counter = 0
            last_label   = ""
            last_stable  = ""
            predictor.reset_buffer()

        # ── Tampilan ─────────────────────────────────────────────────────
        frame = draw_info_panel(frame, label, last_conf,
                                word_buffer, fps, active_model)

        cv2.imshow("SIBI Recognition", frame)

        # ── Input Keyboard ───────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == ord(" "):
            word_buffer += " "
        elif key == 8:                        # Backspace
            word_buffer = word_buffer[:-1]
        elif key == 13:                       # Enter → TTS
            if word_buffer.strip():
                print(f"[TTS] Mengucapkan: '{word_buffer.strip()}'")
                tts.speak(word_buffer.strip())
        elif key == ord("c"):
            word_buffer  = ""
            last_stable  = ""
        elif key == ord("m"):
            # Ganti model secara siklik
            models      = ["random_forest", "cnn", "lstm"]
            idx         = models.index(active_model)
            active_model = models[(idx + 1) % len(models)]
            predictor   = GesturePredictor(active_model)
            print(f"[Model] Beralih ke: {active_model.upper()}")

    cap.release()
    tts.stop()
    cv2.destroyAllWindows()
    print("\nAplikasi ditutup.")


if __name__ == "__main__":
    main()
