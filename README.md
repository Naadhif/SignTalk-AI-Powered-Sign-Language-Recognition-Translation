# 🤟 SignTalk: AI-Powered Sign Language Recognition & Translation

> Platform pintar berbasis Computer Vision dan Deep Learning untuk menerjemahkan Bahasa Isyarat secara real-time demi mewujudkan komunikasi inklusif tanpa batas.

---

## 📖 Deskripsi Proyek

**SignTalk** adalah aplikasi berbasis kecerdasan buatan (AI) yang dirancang untuk menjembatani hambatan komunikasi antara komunitas Tuli (*Deaf/Hard of Hearing*) dan masyarakat luas. 

Banyak penyandang disabilitas rungu/wicara menghadapi tantangan dalam berinteraksi sehari-hari karena keterbatasan pemahaman bahasa isyarat oleh publik. **SignTalk** hadir sebagai solusi terpadu yang memanfaatkan teknik *Computer Vision* dan *Deep Learning* untuk mendeteksi gerakan tangan, ekspresi wajah, serta gestur tubuh melalui kamera, lalu menerjemahkannya secara langsung (*real-time*) menjadi teks maupun suara (dan sebaliknya).

Dengan mendukung gerakan **BISINDO (Bahasa Isyarat Indonesia)** / **SIBI**, platform ini bertujuan untuk mendorong inklusivitas di berbagai sektor seperti pelayanan publik, pendidikan, dan komunikasi harian.

---

## 🎯 Tujuan Proyek

* 🤝 **Mendorong Inklusivitas Komunikasi:** Memudahkan interaksi dua arah antara pengguna Bahasa Isyarat dan penutur verbal tanpa perlu hadirnya penerjemah fisik setiap saat.
* ⚡ **Penerjemahan Akurat & Real-Time:** Mengubah gestur tangan dan gerakan menjadi teks/suara dengan latensi rendah secara presisi.
* 📚 **Sarana Edukasi Interaktif:** Menyediakan modul pembelajaran bahasa isyarat mandiri bagi masyarakat umum yang ingin mempelajari BISINDO/SIBI.
* 🌐 **Aksesibilitas Layanan Publik:** Menjadi modul terintegrasi untuk pendaftaran rumah sakit, layanan perbankan, dan administrasi pemerintahan yang ramah disabilitas.

---

## ✨ Fitur dan Fungsi Utama

* 🎥 **Real-Time Gesture Recognition:** Deteksi gestur tangan dan gerakan jari secara instan menggunakan kamera web (*webcam*) atau kamera smartphone.
* 🔄 **Bidirectional Translation (Two-Way):**
  * **Sign-to-Text/Speech:** Menerjemahkan bahasa isyarat menjadi teks dan suara.
  * **Speech/Text-to-Sign:** Mengubah input suara atau teks menjadi animasi/avatar gerakan bahasa isyarat.
* 📖 **Interactive Dictionary & Learning Module:** Kamus bahasa isyarat interaktif dilengkapi panduan visual, latihan mandiri, dan tes kuis.
* 🎯 **Custom Gesture Trainer:** Fitur bagi pengguna untuk mendaftarkan atau melatih kata/isyarat lokal baru ke dalam model AI.
* 🔊 **Text-to-Speech (TTS) Integration:** Mengeluarkannya dalam bentuk audio jernih sehingga memudahkan pendengar memahami alfabet atau kata yang diperagakan.

---

## 🛠 Teknologi yang Digunakan

| Kategori | Teknologi / Library |
| :--- | :--- |
| **Computer Vision / AI** | MediaPipe, OpenCV, TensorFlow / PyTorch |
| **Frontend** | React.js / Next.js, Tailwind CSS |
| **Backend** | Python (FastAPI / Flask) |
| **Speech Processing** | Web Speech API / Google Text-to-Speech (gTTS) |
| **Deployment** | Docker, Vercel / AWS / Hugging Face Spaces |

---

## 🚀 Tata Cara Menjalankan

Ikuti langkah-langkah di bawah ini untuk menjalankan proyek **SignTalk** di komputer lokal Anda.

### 📋 Prasyarat

Pastikan perangkat Anda telah terpasang software berikut:

* **Python** (v3.9 atau v3.10) — [Unduh di sini](https://www.python.org/)
* **Node.js** (v18.0 atau terbaru) — [Unduh di sini](https://nodejs.org/)
* **Git** — [Unduh di sini](https://git-scm.com/)
* **Webcam / Kamera** yang berfungsi dengan baik.

---

### 📥 Langkah Instalasi

1. **Kloning Repositori**
   ```bash
   git clone https://github.com/username-anda/signtalk.git
   cd signtalk
   ```

2. **Setup Virtual Environment (Backend)**
   ```bash
   # Buat environment baru
   python -m venv venv

   # Aktifkan environment (Windows)
   venv\Scripts\activate

   # Atau aktifkan environment (Mac/Linux)
   source venv/bin/activate
   ```

3. **Install Dependensi Python**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Dependensi Frontend**
   ```bash
   cd frontend
   npm install
   ```

---

### 💻 Cara Menggunakan

1. **Jalankan Backend Server**
   ```bash
   # Dari direktori utama (root) dengan venv aktif
   python app.py
   ```
   *Backend server akan berjalan di `http://localhost:5000` atau `http://localhost:8000`*

2. **Jalankan Frontend Application**
   ```bash
   # Buka terminal baru, masuk ke folder frontend
   cd frontend
   npm run dev
   ```

3. **Akses Aplikasi**
   Buka browser Anda dan navigasikan ke:
   ```text
   http://localhost:3000
   ```

4. **Penggunaan**
   * Berikan izin akses kamera (*allow camera permission*) pada browser Anda.
   * Arahkan tangan Anda ke depan kamera dan mulailah memperagakan isyarat tangan.
   * Teks hasil terjemahan akan langsung muncul di layar.

---

## 🤝 Kontribusi

Kami sangat menyambut kontribusi dari komunitas, terutama dari teman-teman Komunitas Tuli untuk validasi bahasa isyarat!

1. *Fork* repositori ini.
2. Buat branch fitur baru (`git checkout -b fitur/IsyaratBaru`).
3. Commit perubahan Anda (`git commit -m 'Menambahkan dataset alfabet A-Z'`).
4. Push ke branch (`git push origin fitur/IsyaratBaru`).
5. Buka sebuah *Pull Request*.

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah **MIT License** — lihat file [LICENSE](LICENSE) untuk detail selengkapnya.
