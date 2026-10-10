#!/usr/bin/env python3
# collect_data.py
# Kumpulkan dataset SIBI dari kamera — simpan fitur ke CSV
# Gunakan ini SEBELUM melatih model.

import cv2
import csv
import os
import time
import numpy as np
from config import (SIBI_LABELS, PROC_DIR, FRAME_WIDTH, FRAME_HEIGHT,
                    COLOR_GREEN, COLOR_WHITE, COLOR_ORANGE)
from utils.hand_detector    import get_detector
from utils.feature_extractor import extract_features

OUTPUT_CSV  = os.path.join(PROC_DIR, "sibi_dataset.csv")
SAMPLES_PER_LABEL = 200   # Jumlah sampel per huruf


def main():
    detector = get_detector()
    cap      = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Siapkan file CSV
    file_exists = os.path.isfile(OUTPUT_CSV)
    csv_file    = open(OUTPUT_CSV, "a", newline="")
    writer      = csv.writer(csv_file)
    if not file_exists:
        # Header: 78 fitur + label
        header = [f"f{i}" for i in range(78)] + ["label"]
        writer.writerow(header)

    print("=" * 50)
    print(" PENGUMPULAN DATA SIBI")
    print("=" * 50)
    print("Tekan huruf A-Z untuk mulai rekam gesture tsb.")
    print("Tekan 'ESC' untuk keluar.\n")

    current_label  : str | None = None
    count          : int        = 0
    recording      : bool       = False
    countdown      : int        = 0
    countdown_start: float      = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hands = detector.detect(frame)
        frame = detector.draw(frame, hands)

        # ── UI ──────────────────────────────────────────────────────────
        if current_label:
            status = f"Merekam: {current_label} ({count}/{SAMPLES_PER_LABEL})"
            color  = COLOR_GREEN if recording else COLOR_ORANGE
        else:
            status = "Tekan A-Z untuk mulai"
            color  = COLOR_WHITE

        cv2.putText(frame, status, (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Countdown sebelum rekam
        if countdown > 0:
            elapsed = time.time() - countdown_start
            remain  = max(0, countdown - int(elapsed))
            cv2.putText(frame, f"Siapkan tangan... {remain}",
                        (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        COLOR_ORANGE, 2)
            if remain == 0:
                countdown  = 0
                recording  = True

        # ── Rekam ──────────────────────────────────────────────────────
        if recording and hands and count < SAMPLES_PER_LABEL:
            feats = extract_features(hands[0]["landmarks"])
            writer.writerow(list(feats) + [current_label])
            csv_file.flush()
            count += 1

            if count >= SAMPLES_PER_LABEL:
                recording     = False
                current_label = None
                count         = 0
                print(f"  ✓ Selesai {SAMPLES_PER_LABEL} sampel.\n")

        # Progress bar
        if current_label and SAMPLES_PER_LABEL > 0:
            pct = count / SAMPLES_PER_LABEL
            w   = int(FRAME_WIDTH * pct)
            cv2.rectangle(frame, (0, FRAME_HEIGHT - 12),
                          (w, FRAME_HEIGHT), COLOR_GREEN, -1)

        cv2.imshow("SIBI Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif chr(key).upper() in SIBI_LABELS and not recording:
            current_label  = chr(key).upper()
            count          = 0
            recording      = False
            countdown      = 3
            countdown_start = time.time()
            print(f"[+] Huruf: {current_label} — bersiap...")

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"\nDataset disimpan: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
