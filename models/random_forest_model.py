# models/random_forest_model.py
# Model Random Forest untuk klasifikasi gesture statis SIBI

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os
from config import MODEL_PATHS, N_CLASSES, SIBI_LABELS


def build_rf_pipeline(n_estimators: int = 200,
                      max_depth: int | None = None) -> Pipeline:
    """
    Pipeline: StandardScaler → RandomForestClassifier
    Random Forest bagus untuk gesture statis (fitur posisi + sudut).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators = n_estimators,
            max_depth    = max_depth,
            min_samples_split = 4,
            min_samples_leaf  = 2,
            max_features = "sqrt",
            class_weight = "balanced",   # tangani ketidakseimbangan kelas
            random_state = 42,
            n_jobs       = -1,           # semua core CPU
        )),
    ])


def train(X_train: np.ndarray, y_train: np.ndarray,
          X_val:   np.ndarray | None = None,
          y_val:   np.ndarray | None = None) -> Pipeline:
    """
    Latih model Random Forest.

    X: (N, 78)  — fitur posisi + sudut
    y: (N,)     — label string e.g. 'A', 'B', ...
    """
    le = LabelEncoder()
    le.fit(SIBI_LABELS)
    y_enc = le.transform(y_train)

    pipeline = build_rf_pipeline()
    pipeline.fit(X_train, y_enc)

    # Simpan
    joblib.dump(pipeline, MODEL_PATHS["random_forest"])
    joblib.dump(le,       MODEL_PATHS["label_encoder"])
    print(f"[RF] Model disimpan: {MODEL_PATHS['random_forest']}")

    if X_val is not None and y_val is not None:
        y_val_enc = le.transform(y_val)
        acc = pipeline.score(X_val, y_val_enc)
        print(f"[RF] Akurasi validasi: {acc:.4f}")

    return pipeline


def evaluate(pipeline: Pipeline, X_test: np.ndarray,
             y_test: np.ndarray) -> dict:
    """Evaluasi model dengan laporan lengkap."""
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 accuracy_score)

    le       = joblib.load(MODEL_PATHS["label_encoder"])
    y_enc    = le.transform(y_test)
    y_pred   = pipeline.predict(X_test)
    y_labels = le.inverse_transform(y_pred)

    acc    = accuracy_score(y_enc, y_pred)
    report = classification_report(y_test, y_labels, target_names=SIBI_LABELS)
    cm     = confusion_matrix(y_enc, y_pred)

    print(f"\n[RF] Akurasi Test: {acc:.4f}")
    print(report)

    return {"accuracy": acc, "report": report, "confusion_matrix": cm}
