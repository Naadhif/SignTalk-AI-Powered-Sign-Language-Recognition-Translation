# utils/hand_detector.py
# Deteksi tangan TANPA MediaPipe — menggunakan deteksi warna kulit + kontur

import cv2
import numpy as np
from config import (HAND_DETECT_METHOD, MAX_HANDS,
                    FRAME_WIDTH, FRAME_HEIGHT, N_LANDMARKS)


class SkinColorHandDetector:
    """
    Deteksi tangan berbasis segmentasi warna kulit (HSV + YCrCb).
    Tidak memerlukan MediaPipe atau TensorFlow.
    Menghasilkan 21 landmark pseudo dari kontur tangan.
    """

    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # ── Segmentasi ──────────────────────────────────────────────────────────
    def _skin_mask(self, frame: np.ndarray) -> np.ndarray:
        """Gabungkan masker HSV + YCrCb untuk deteksi kulit lebih robust."""
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        mask_hsv = cv2.inRange(hsv,
                               np.array([0, 20, 70],  dtype=np.uint8),
                               np.array([25, 255, 255], dtype=np.uint8))

        mask_ycrcb = cv2.inRange(ycrcb,
                                 np.array([0,  133, 77], dtype=np.uint8),
                                 np.array([255, 173, 127], dtype=np.uint8))

        mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        return mask

    # ── Landmark pseudo (21 titik dari skeleton kontur) ─────────────────────
    def _extract_pseudo_landmarks(self, contour: np.ndarray,
                                  hull_indices: np.ndarray,
                                  bbox: tuple) -> list[tuple[float, float, float]]:
        """
        Bangkitkan 21 pseudo-landmark dari:
          - Centroid tangan
          - Titik-titik convex hull (ujung jari & sela jari)
          - Interpolasi sepanjang kontur
        Mengembalikan list of (x_norm, y_norm, z_approx).
        """
        x, y, w, h = bbox
        cx = x + w // 2
        cy = y + h // 2

        hull_pts = [tuple(contour[i][0]) for i in hull_indices.flatten()]

        # Ambil paling banyak 20 titik dari hull, lalu tambah centroid sebagai #0
        sampled = hull_pts[:20]
        while len(sampled) < 20:
            # Interpolasi tambahan jika hull sedikit
            idx = len(sampled) % max(len(hull_pts), 1)
            sampled.append(hull_pts[idx])

        landmarks = [(float(cx), float(cy), 0.0)]  # titik 0 = pangkal telapak
        for px, py in sampled:
            # z diperkirakan dari jarak ke centroid (lebih jauh = ujung jari)
            dist = np.hypot(px - cx, py - cy)
            z    = dist / max(w, h, 1)
            landmarks.append((float(px), float(py), float(z)))

        return landmarks  # 21 titik

    # ── API Utama ────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Deteksi tangan pada frame.

        Menyaring kontur kulit agar tidak menangkap wajah atau objek non-tangan.
        """
        mask = self._skin_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        h_img, w_img = frame.shape[:2]
        min_area = 0.006 * FRAME_WIDTH * FRAME_HEIGHT
        max_area = 0.30 * w_img * h_img

        filtered = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 0 or h <= 0:
                continue

            cx = x + w / 2
            cy = y + h / 2
            aspect = w / max(h, 1)
            extent = area / max(w * h, 1)

            # ROI: fokus ke bagian bawah frame. Wajah biasanya ada di atas/ tengah.
            if cy < h_img * 0.20:
                continue

            # Tangan biasanya lebih panjang dari tinggi, bukan oval seperti wajah.
            if aspect < 0.25 or aspect > 2.5:
                continue

            # Wajah biasanya area kulit yang luas dan bentuk hampir persegi / oval.
            if extent > 0.80 and 0.65 <= aspect <= 1.40:
                continue

            # Abaikan area kulit yang jauh dari bawah kamera.
            if y < h_img * 0.15:
                continue

            # Jangan ambil objek terlalu besar di tengah frame yang mirip wajah.
            if abs(cx - w_img / 2) < w_img * 0.18 and cy < h_img * 0.60:
                continue

            filtered.append(cnt)

        if not filtered:
            return []

        # Prioritaskan kontur yang paling bawah dan paling besar dalam ROI.
        filtered = sorted(
            filtered,
            key=lambda c: (
                cv2.contourArea(c),
                cv2.boundingRect(c)[1],
            ),
            reverse=True,
        )

        results = []
        for cnt in filtered[:MAX_HANDS]:
            hull_idx = cv2.convexHull(cnt, returnPoints=False)
            if hull_idx is None or len(hull_idx) < 3:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 0 or h <= 0:
                continue

            # Filter tambahan agar bukan wajah:
            if w * h > 0.18 * w_img * h_img and 0.8 <= (w / max(h, 1)) <= 1.2:
                continue

            lms = self._extract_pseudo_landmarks(cnt, hull_idx, (x, y, w, h))
            crop = mask[y:y + h, x:x + w] if h > 0 and w > 0 else np.zeros((1, 1))

            results.append({
                "bbox":      (x, y, w, h),
                "landmarks": lms,
                "mask_crop": crop,
            })

        return results

    # ── Gambar anotasi ───────────────────────────────────────────────────────
    @staticmethod
    def draw(frame: np.ndarray, hands: list[dict]) -> np.ndarray:
        """Gambar bounding box dan landmark di frame."""
        out = frame.copy()
        for hand in hands:
            x, y, w, h = hand["bbox"]
            cv2.rectangle(out, (x, y), (x+w, y+h), (60, 200, 120), 2)
            for i, (lx, ly, _) in enumerate(hand["landmarks"]):
                color = (0, 165, 255) if i == 0 else (255, 255, 255)
                cv2.circle(out, (int(lx), int(ly)), 4, color, -1)
        return out


class YOLOHandDetector:
    """
    Deteksi tangan menggunakan YOLOv8 (ultralytics).
    Memerlukan: pip install ultralytics
    Model: yolov8n.pt (general) atau hand-specific weights.
    """

    def __init__(self, weights: str = "yolov8n.pt"):
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights)
            print(f"[YOLOHandDetector] Model dimuat: {weights}")
        except ImportError:
            raise ImportError("Install ultralytics: pip install ultralytics")

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = self.model(frame, verbose=False, conf=0.4)
        hands   = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                # Tanpa landmark detail — buat pseudo dari grid 4×5
                lms = _grid_landmarks(x1, y1, w, h)
                hands.append({"bbox": (x1, y1, w, h), "landmarks": lms})
        return hands[:MAX_HANDS]

    @staticmethod
    def draw(frame, hands):
        return SkinColorHandDetector.draw(frame, hands)


def _grid_landmarks(x, y, w, h) -> list[tuple]:
    """Buat 21 landmark pada posisi grid 4×5+1 di dalam bbox."""
    pts = [(x + w // 2, y + h // 2, 0.0)]  # pangkal
    cols, rows = 4, 5
    for r in range(rows):
        for c in range(cols):
            px = x + int(w * (c + 1) / (cols + 1))
            py = y + int(h * (r + 1) / (rows + 1))
            pts.append((float(px), float(py), float(r) / rows))
    return pts[:21]


def get_detector():
    """Factory — pilih detektor sesuai config."""
    if HAND_DETECT_METHOD == "yolo":
        return YOLOHandDetector()
    return SkinColorHandDetector()
