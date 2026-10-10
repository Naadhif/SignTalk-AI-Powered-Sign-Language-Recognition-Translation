@echo off
:: install_windows.bat
:: Installer khusus Python 3.15 di Windows
:: Semua pakai --only-binary :all: agar tidak perlu C++ compiler
:: Jalankan: klik dua kali, atau dari CMD di folder project

echo ============================================
echo  SIBI Recognition - Install Windows
echo  Python 3.15 compatible
echo ============================================
echo.

:: Pastikan pip terbaru
python -m pip install --upgrade pip

echo.
echo [1/5] Menginstall numpy, scipy, joblib...
python -m pip install --only-binary :all: numpy scipy joblib

echo.
echo [2/5] Menginstall scikit-learn...
python -m pip install --only-binary :all: scikit-learn

echo.
echo [3/5] Menginstall OpenCV...
python -m pip install --only-binary :all: opencv-python

echo.
echo [4/5] Menginstall pandas, matplotlib, tqdm...
python -m pip install --only-binary :all: pandas matplotlib tqdm

echo.
echo [5/5] Menginstall TTS (pyttsx3, gTTS)...
python -m pip install pyttsx3
python -m pip install gTTS

echo.
echo ============================================
echo  Selesai! Coba jalankan: python main.py
echo ============================================
pause
