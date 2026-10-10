# models/cnn_model.py
# Pengganti CNN — SVM dengan fitur HOG (Histogram of Oriented Gradients)
# Tidak butuh PyTorch/TensorFlow, kompatibel Python 3.15

import numpy as np
import joblib
import cv2
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from config import MODEL_PATHS, N_CLASSES, SIBI_LABELS, CNN_INPUT_SIZE


def extract_hog_features(image: np.ndarray,
                          size: tuple = CNN_INPUT_SIZE) -> np.ndarray:
    """
    Ekstrak fitur HOG dari gambar crop tangan.
    HOG (Histogram of Oriented Gradients) sangat efektif
    menggantikan CNN untuk klasifikasi bentuk tangan.

    Input : image BGR atau grayscale, ukuran bebas
    Output: array fitur HOG (1D)
    """
    # Resize ke ukuran standar
    img = cv2.resize(image, size)

    # Konversi ke grayscale jika perlu
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ekualisasi histogram untuk normalisasi pencahayaan
    img = cv2.equalizeHist(img)

    # HOG descriptor
    win_size    = size
    block_size  = (16, 16)
    block_stride = (8, 8)
    cell_size   = (8, 8)
    n_bins      = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
                            cell_size, n_bins)
    feats = hog.compute(img).flatten()
    return feats.astype(np.float32)


def build_svm_pipeline() -> Pipeline:
    """
    Pipeline: StandardScaler → PCA → SVM (RBF kernel)
    Setara kemampuan CNN untuk dataset skala kecil-menengah.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=150, whiten=True, random_state=42)),
        ("svm",    SVC(
            kernel       = "rbf",
            C            = 10.0,
            gamma        = "scale",
            probability  = True,      # perlu untuk confidence score
            class_weight = "balanced",
            random_state = 42,
        )),
    ])


def train(X_train: np.ndarray, y_train: np.ndarray,
          X_val:   np.ndarray | None = None,
          y_val:   np.ndarray | None = None) -> Pipeline:
    """
    Latih SVM+HOG.
    X: (N, HOG_dim)  — fitur HOG dari crop tangan
    y: (N,)          — label string 'A'-'Z'
    """
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(SIBI_LABELS)
    y_enc = le.transform(y_train)

    pipeline = build_svm_pipeline()
    print(f"[SVM] Melatih dengan {len(X_train)} sampel...")
    pipeline.fit(X_train, y_enc)

    joblib.dump(pipeline, MODEL_PATHS["cnn"])       # simpan di slot 'cnn'
    joblib.dump(le,       MODEL_PATHS["label_encoder"])
    print(f"[SVM] Model disimpan: {MODEL_PATHS['cnn']}")

    if X_val is not None and y_val is not None:
        y_val_enc = le.transform(y_val)
        acc = pipeline.score(X_val, y_val_enc)
        print(f"[SVM] Akurasi validasi: {acc:.4f}")

    return pipeline


def evaluate(pipeline: Pipeline,
             X_test: np.ndarray, y_test: np.ndarray) -> dict:
    from sklearn.metrics import classification_report, accuracy_score
    le     = joblib.load(MODEL_PATHS["label_encoder"])
    y_enc  = le.transform(y_test)
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_enc, y_pred)
    report = classification_report(y_test, le.inverse_transform(y_pred))
    print(f"\n[SVM] Akurasi Test: {acc:.4f}")
    print(report)
    return {"accuracy": acc, "report": report}
