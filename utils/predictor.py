# utils/predictor.py
# Prediksi gesture SIBI + Confidence Threshold Filter
# Semua model sekarang berbasis scikit-learn (tanpa PyTorch)

import numpy as np
from collections import deque, Counter
from config import (ACTIVE_MODEL, MODEL_PATHS, CONFIDENCE_THRESHOLD,
                    PREDICTION_BUFFER, SIBI_LABELS, LSTM_SEQ_LEN)


class GesturePredictor:
    """
    Wrapper prediksi yang:
      1. Memuat model sesuai ACTIVE_MODEL (RF / SVM+HOG / GBM)
      2. Menerapkan confidence threshold filter
      3. Menghaluskan prediksi dengan majority-vote buffer
    """

    def __init__(self, model_type: str = ACTIVE_MODEL):
        self.model_type = model_type
        self.model      = None
        self.le         = None
        self._buffer: deque = deque(maxlen=PREDICTION_BUFFER)
        self._seq_buf: deque = deque(maxlen=LSTM_SEQ_LEN)
        self._load()

    # ── Load ────────────────────────────────────────────────────────────────
    def _load(self):
        import os, joblib
        path = MODEL_PATHS.get(self.model_type)

        if not path or not os.path.exists(path):
            print(f"[Predictor] Model '{self.model_type}' belum ada di {path}")
            print("  → Jalankan train_rf.py / train_cnn.py / train_lstm.py dulu.")
            return

        self.model = joblib.load(path)

        le_path = MODEL_PATHS["label_encoder"]
        if os.path.exists(le_path):
            self.le = joblib.load(le_path)

        print(f"[Predictor] Model '{self.model_type}' dimuat dari {path}")

    # ── Predict ─────────────────────────────────────────────────────────────
    def predict(self, features: np.ndarray) -> tuple[str | None, float]:
        """
        features:
          - model 'random_forest' / 'lstm' : array (78,) fitur posisi+sudut
          - model 'cnn'                    : array HOG dari crop_hand_region,
                                             atau array (78,) jika HOG belum dihitung

        Mengembalikan (label, confidence) atau (None, conf) jika di bawah threshold.
        """
        if self.model is None:
            return None, 0.0

        if self.model_type == "lstm":
            return self._predict_seq(features)
        else:
            return self._predict_static(features)

    def _predict_static(self, features: np.ndarray):
        """RF dan SVM — input fitur 1D."""
        X = features.reshape(1, -1)
        try:
            proba = self.model.predict_proba(X)[0]
        except Exception:
            pred  = self.model.predict(X)[0]
            label = self._decode(pred)
            return self._apply_buffer(label, 0.8)

        idx   = int(np.argmax(proba))
        conf  = float(proba[idx])
        label = self._decode(idx)
        return self._apply_buffer(label, conf)

    def _predict_seq(self, features: np.ndarray):
        """GBM sekuens — akumulasi frame lalu prediksi."""
        from models.lstm_model import build_sequence_features

        self._seq_buf.append(features)
        if len(self._seq_buf) < 5:          # tunggu minimal 5 frame
            return None, 0.0

        seq   = np.stack(list(self._seq_buf))
        X     = build_sequence_features(seq).reshape(1, -1)
        try:
            proba = self.model.predict_proba(X)[0]
        except Exception:
            pred  = self.model.predict(X)[0]
            return self._apply_buffer(self._decode(pred), 0.8)

        idx   = int(np.argmax(proba))
        conf  = float(proba[idx])
        return self._apply_buffer(self._decode(idx), conf)

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _decode(self, idx) -> str:
        if self.le is not None:
            try:
                return str(self.le.inverse_transform([int(idx)])[0])
            except Exception:
                pass
        return SIBI_LABELS[int(idx)] if int(idx) < len(SIBI_LABELS) else "?"

    def _apply_buffer(self, label: str, conf: float):
        if conf < CONFIDENCE_THRESHOLD:
            return None, conf

        self._buffer.append(label)
        most_common, count = Counter(self._buffer).most_common(1)[0]
        smooth_conf = conf * (count / len(self._buffer))
        return most_common, smooth_conf

    def reset_buffer(self):
        self._buffer.clear()
        self._seq_buf.clear()
