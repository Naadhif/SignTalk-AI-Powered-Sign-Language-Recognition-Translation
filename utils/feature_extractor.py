# utils/feature_extractor.py
# Ekstraksi fitur: posisi landmark (x,y,z) + sudut antar ruas jari

import numpy as np
from config import N_LANDMARKS, ANGLE_PAIRS, N_FEATURES, N_ANGLE_FEATURES


def normalize_landmarks(landmarks: list[tuple]) -> np.ndarray:
    """
    Normalisasi 21 landmark relatif terhadap:
      - Posisi  : geser sehingga titik 0 (pergelangan) jadi origin
      - Skala   : bagi dengan jarak max antar landmark (skala-invariant)

    Input : list of (x, y, z) — 21 titik
    Output: array shape (63,) — 21*3, dtype float32
    """
    pts = np.array(landmarks, dtype=np.float32)   # (21, 3)

    # Geser ke origin
    pts -= pts[0]

    # Normalisasi skala
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 1e-6:
        pts /= scale

    return pts.flatten()   # (63,)


def compute_angles(landmarks: list[tuple]) -> np.ndarray:
    """
    Hitung sudut (radian) pada tiap ruas jari dari ANGLE_PAIRS.
    Setiap triplet (a, b, c) → sudut di titik b.

    Output: array shape (15,), dtype float32
    """
    pts    = np.array(landmarks, dtype=np.float32)   # (21, 3)
    angles = []

    for a_idx, b_idx, c_idx in ANGLE_PAIRS:
        ba = pts[a_idx] - pts[b_idx]
        bc = pts[c_idx] - pts[b_idx]

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba < 1e-6 or norm_bc < 1e-6:
            angles.append(0.0)
            continue

        cos_ang = np.dot(ba, bc) / (norm_ba * norm_bc)
        cos_ang = np.clip(cos_ang, -1.0, 1.0)
        angles.append(float(np.arccos(cos_ang)))

    return np.array(angles, dtype=np.float32)   # (15,)


def extract_features(landmarks: list[tuple],
                     include_angles: bool = True) -> np.ndarray:
    """
    Gabungkan posisi ternormalisasi + sudut jari.

    Output: array shape (78,) jika include_angles else (63,)
    """
    pos = normalize_landmarks(landmarks)           # (63,)
    if include_angles:
        ang = compute_angles(landmarks)            # (15,)
        return np.concatenate([pos, ang])          # (78,)
    return pos


def extract_sequence_features(seq_landmarks: list[list[tuple]],
                               seq_len: int) -> np.ndarray:
    """
    Untuk model LSTM: ekstraksi fitur dari sekuens frame.

    Input : list of (21 landmark per frame), panjang bervariasi
    Output: array shape (seq_len, 78) — padding atau cropping ke seq_len
    """
    feats = [extract_features(lm) for lm in seq_landmarks]
    n     = len(feats)
    feat_dim = feats[0].shape[0] if feats else 78

    out = np.zeros((seq_len, feat_dim), dtype=np.float32)

    if n >= seq_len:
        out[:] = feats[-seq_len:]          # ambil frame terbaru
    else:
        out[seq_len - n:] = feats          # padding awal dengan nol

    return out   # (seq_len, 78)


def crop_hand_region(frame, bbox: tuple, size: tuple = (64, 64)):
    """
    Crop dan resize area tangan untuk input CNN.

    Input : frame BGR, bbox=(x,y,w,h)
    Output: array (size[1], size[0], 3) uint8
    """
    import cv2
    x, y, w, h = bbox
    margin = int(max(w, h) * 0.15)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame.shape[1], x + w + margin)
    y2 = min(frame.shape[0], y + h + margin)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((*size[::-1], 3), dtype=np.uint8)

    return cv2.resize(crop, size)
