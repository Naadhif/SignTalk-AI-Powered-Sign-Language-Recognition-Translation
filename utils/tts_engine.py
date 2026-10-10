# utils/tts_engine.py
# Text-to-Speech tanpa pygame — menggunakan pyttsx3 (offline) atau gTTS (online)

import threading
import queue
import os
import tempfile
from config import TTS_ENGINE, TTS_RATE, TTS_LANG


class TTSEngine:
    """
    Engine TTS non-blocking dengan antrian.
    Menggunakan pyttsx3 (offline) atau gTTS+playsound (online).
    Thread terpisah agar tidak memblokir loop kamera.
    """

    def __init__(self):
        self._queue   : queue.Queue = queue.Queue()
        self._running : bool        = False
        self._thread  : threading.Thread | None = None
        self._engine_type = TTS_ENGINE

        if self._engine_type == "pyttsx3":
            self._init_pyttsx3()
        else:
            self._init_gtts()

    # ── Init ────────────────────────────────────────────────────────────────
    def _init_pyttsx3(self):
        try:
            import pyttsx3
            self._pyttsx3_engine = pyttsx3.init()
            self._pyttsx3_engine.setProperty("rate", TTS_RATE)

            # Pilih suara bahasa Indonesia jika tersedia
            voices = self._pyttsx3_engine.getProperty("voices")
            for v in voices:
                if "id" in v.id.lower() or "indonesia" in v.name.lower():
                    self._pyttsx3_engine.setProperty("voice", v.id)
                    break

            print("[TTS] pyttsx3 siap (offline)")
        except ImportError:
            print("[TTS] pyttsx3 tidak tersedia, beralih ke gTTS")
            self._engine_type = "gtts"
            self._init_gtts()

    def _init_gtts(self):
        try:
            import gtts  # noqa: F401
            # Cek player yang tersedia: sounddevice, winsound (Windows), atau subprocess
            self._audio_player = self._detect_audio_player()
            print(f"[TTS] gTTS siap (butuh internet) — player: {self._audio_player}")
        except ImportError:
            print("[TTS] Tidak ada engine TTS yang tersedia.")
            self._engine_type = "none"

    def _detect_audio_player(self) -> str:
        """Deteksi otomatis cara putar audio yang tersedia."""
        try:
            import sounddevice  # noqa: F401
            import soundfile    # noqa: F401
            return "sounddevice"
        except ImportError:
            pass
        import sys
        if sys.platform == "win32":
            return "winsound"
        # Linux/Mac fallback — gunakan subprocess dengan afplay/aplay
        import shutil
        for cmd in ("afplay", "aplay", "mpg123", "ffplay"):
            if shutil.which(cmd):
                return cmd
        return "none"

    # ── Antrian & Thread ────────────────────────────────────────────────────
    def start(self):
        """Mulai thread TTS."""
        self._running = True
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        """Hentikan thread TTS."""
        self._running = False
        self._queue.put(None)   # Sentinel

    def speak(self, text: str):
        """Masukkan teks ke antrian untuk diucapkan."""
        if text.strip():
            # Kosongkan antrian lama agar tidak menumpuk
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put(text)

    # ── Worker Thread ────────────────────────────────────────────────────────
    def _worker(self):
        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is None:
                break

            self._say(text)

    def _say(self, text: str):
        if self._engine_type == "pyttsx3":
            self._say_pyttsx3(text)
        elif self._engine_type == "gtts":
            self._say_gtts(text)
        else:
            print(f"[TTS] (no engine) {text}")

    def _say_pyttsx3(self, text: str):
        try:
            self._pyttsx3_engine.say(text)
            self._pyttsx3_engine.runAndWait()
        except Exception as e:
            print(f"[TTS pyttsx3] Error: {e}")

    def _say_gtts(self, text: str):
        try:
            from gtts import gTTS
            import subprocess, sys

            # Simpan ke file mp3 sementara
            tts = gTTS(text=text, lang=TTS_LANG, slow=False)
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp.close()
            tts.save(tmp.name)

            player = getattr(self, "_audio_player", self._detect_audio_player())

            if player == "sounddevice":
                # Konversi mp3 → wav via pydub (jika ada) atau soundfile
                self._play_with_sounddevice(tmp.name)

            elif player == "winsound":
                # Windows: konversi mp3→wav dulu via subprocess ffmpeg,
                # atau langsung pakai winsound jika sudah wav
                wav_tmp = tmp.name.replace(".mp3", ".wav")
                ret = subprocess.run(
                    ["ffmpeg", "-y", "-i", tmp.name, wav_tmp],
                    capture_output=True)
                if ret.returncode == 0:
                    import winsound
                    winsound.PlaySound(wav_tmp, winsound.SND_FILENAME)
                    os.unlink(wav_tmp)
                else:
                    # ffmpeg tidak ada — fallback ke pyttsx3
                    self._say_pyttsx3(text)

            elif player != "none":
                # aplay / afplay / mpg123 / ffplay
                subprocess.run([player, tmp.name],
                               capture_output=True, timeout=15)
            else:
                print(f"[TTS] (no audio player) {text}")

            os.unlink(tmp.name)
        except Exception as e:
            print(f"[TTS gTTS] Error: {e}")

    def _play_with_sounddevice(self, mp3_path: str):
        """Putar mp3 via sounddevice (konversi dulu ke numpy array)."""
        import subprocess, tempfile as tf, os
        wav_path = mp3_path.replace(".mp3", "_sd.wav")
        # Coba konversi dengan ffmpeg
        ret = subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ar", "22050",
             "-ac", "1", wav_path],
            capture_output=True)
        if ret.returncode == 0:
            try:
                import soundfile as sf
                import sounddevice as sd
                data, sr = sf.read(wav_path)
                sd.play(data, sr)
                sd.wait()
            except Exception as e:
                print(f"[TTS sounddevice] {e}")
            finally:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
        else:
            # ffmpeg tidak ada — fallback pyttsx3
            self._say_pyttsx3(mp3_path)


# ── Singleton ─────────────────────────────────────────────────────────────────
_tts: TTSEngine | None = None

def get_tts() -> TTSEngine:
    global _tts
    if _tts is None:
        _tts = TTSEngine()
        _tts.start()
    return _tts
