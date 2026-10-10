# models/lstm_model.py
# Pengganti LSTM — Gradient Boosting dengan fitur sliding window
# Tidak butuh PyTorch/TensorFlow, kompatibel Python 3.15

import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import MODEL_PATHS, SIBI_LABELS, LSTM_SEQ_LEN

FEATURE_DIM = 78   # 63 posisi + 15 sudut per frame


def build_sequence_features(seq: np.ndarray) -> np.ndarray:
    """
    Ubah sekuens frame menjadi fitur statistik temporal.
    Menangkap pola gerakan tanpa membutuhkan LSTM.

    Input : array (seq_len, 78)
    Output: array 1D — statistik per fitur × frame
    """
    if seq.ndim == 1:
        # Satu frame saja — tile jadi sekuens
        seq = np.tile(seq, (LSTM_SEQ_LEN, 1))

    # Statistik per dimensi fitur
    mean_f = seq.mean(axis=0)          # rata-rata
    std_f  = seq.std(axis=0)           # variabilitas
    diff_f = np.diff(seq, axis=0).mean(axis=0)  # kecepatan rata-rata
    rng_f  = seq.max(axis=0) - seq.min(axis=0)  # jangkauan gerakan

    # Frame pertama, tengah, terakhir (posisi kunci)
    mid    = len(seq) // 2
    f_first = seq[0]
    f_mid   = seq[mid]
    f_last  = seq[-1]

    return np.concatenate([
        mean_f, std_f, diff_f, rng_f,
        f_first, f_mid, f_last
    ]).astype(np.float32)
    # Output dim: 78*7 = 546


def build_gradient_boost_pipeline() -> Pipeline:
    """
    Pipeline: StandardScaler → HistGradientBoosting
    HistGradientBoosting adalah versi sklearn dari LightGBM —
    cepat, akurat, dan menangani urutan temporal melalui
    fitur statistik sliding window.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("hgb",    HistGradientBoostingClassifier(
            max_iter          = 300,
            max_depth         = 6,
            learning_rate     = 0.05,
            min_samples_leaf  = 5,
            l2_regularization = 0.1,
            early_stopping    = True,
            validation_fraction = 0.1,
            n_iter_no_change  = 20,
            class_weight      = "balanced",
            random_state      = 42,
        )),
    ])


def train(X_train: np.ndarray, y_train: np.ndarray,
          X_val:   np.ndarray | None = None,
          y_val:   np.ndarray | None = None) -> Pipeline:
    """
    X: (N, 546) — fitur sekuens (build_sequence_features per sampel)
    y: (N,)     — label string
    """
    le = LabelEncoder()
    le.fit(SIBI_LABELS)
    y_enc = le.transform(y_train)

    pipeline = build_gradient_boost_pipeline()
    print(f"[GBM] Melatih dengan {len(X_train)} sampel...")
    pipeline.fit(X_train, y_enc)

    joblib.dump(pipeline, MODEL_PATHS["lstm"])
    joblib.dump(le,       MODEL_PATHS["label_encoder"])
    print(f"[GBM] Model disimpan: {MODEL_PATHS['lstm']}")

    if X_val is not None and y_val is not None:
        y_val_enc = le.transform(y_val)
        acc = pipeline.score(X_val, y_val_enc)
        print(f"[GBM] Akurasi validasi: {acc:.4f}")

    return pipeline


def evaluate(pipeline: Pipeline,
             X_test: np.ndarray, y_test: np.ndarray) -> dict:
    from sklearn.metrics import classification_report, accuracy_score
    le     = joblib.load(MODEL_PATHS["label_encoder"])
    y_enc  = le.transform(y_test)
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_enc, y_pred)
    report = classification_report(y_test, le.inverse_transform(y_pred))
    print(f"\n[GBM] Akurasi Test: {acc:.4f}")
    print(report)
    return {"accuracy": acc, "report": report}
