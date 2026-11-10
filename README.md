# IndoBERT REST API di Streamlit Community Cloud

Versi Streamlit dari **`huggingface-space/`**, memakai bobot
`indobert-labeler/model.int8.onnx` dan tokenizer/config yang sama. Model dasar
adalah `indolem/indobert-base-uncased`. Ini IndoBERT ONNX INT8, bukan model Lite
atau TF-IDF. Paket mandiri ini tidak memerlukan Docker, PyTorch, Transformers,
training, atau konversi model saat startup.

Dashboard dan REST memakai Streamlit 1.67 `st.App`, dengan kontrak
`POST /predict` yang sama seperti versi TF-IDF. Bobot, normalisasi lowercase/trim,
urutan label, threshold default 0.5, serta batas 128 token mengikuti paket asal.

Verifikasi lokal: **14 tes integrasi lulus** memakai bobot ONNX asli. Hash seluruh
lima file model/tokenizer cocok dengan sumber `huggingface-space/`, dan probabilitas
pada fixture pendek maupun teks terpotong cocok dengan inferensi langsung model
sumber. Startup CLI `streamlit run app.py` juga berhasil mengembalikan JSON
`SAMPAH` dan `LUBANG` untuk contoh laporan jalan berlubang dan banyak sampah.
Pemeriksaan dependency lulus. Ini pengujian lokal; deployment Cloud belum dilakukan.

## Isi paket

```text
app.py                         entrypoint server Streamlit dan route REST
dashboard.py                   UI pengujian, berbagi model/cache dengan REST
engine.py                      runtime IndoBERT ONNX INT8
requirements.txt
.gitattributes                 konfigurasi Git LFS untuk ONNX
.streamlit/config.toml
indobert-labeler/
    config.json
    label_config.json
    tokenizer.json
    tokenizer_config.json
    model.int8.onnx
```

## Upload bobot dengan Git LFS

Bobot ONNX berukuran **111.733.466 byte (sekitar 106,6 MiB)**, melewati batas
100 MiB untuk file Git biasa. Upload lewat halaman web GitHub juga dibatasi
25 MiB per file. Gunakan **Git LFS**; Community Cloud mendukung repository LFS.

Untuk repository deployment baru, ekstrak isi ZIP paket ke folder kerja baru,
buka terminal pada folder tersebut, lalu gunakan alur berikut. Ganti URL remote
dengan repository GitHub tujuan yang sudah dibuat:

```powershell
git init
git lfs install --local
git lfs track "*.onnx"
git add .gitattributes .gitignore .streamlit README.md requirements.txt app.py dashboard.py engine.py indobert-labeler
git commit -m "Add Streamlit IndoBERT ONNX API"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
git lfs ls-files
```

Pastikan model masuk daftar `git lfs ls-files`. Contoh ini untuk repository baru;
sesuaikan alur Git jika repository tujuan sudah berisi commit. Pastikan konten
file model di checkout deployment benar-benar terunduh, bukan pointer LFS kecil.
Runtime memberi pesan khusus jika yang ditemukan hanya pointer LFS.

Rujukan: [batas file GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github),
[dukungan Git LFS Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).

## Deploy ke Streamlit Community Cloud

1. Pakai repository deployment yang memuat seluruh isi paket ini, termasuk bobot
   ONNX melalui Git LFS. Tidak perlu mengganti deployment TF-IDF yang sudah berjalan.
2. Di Community Cloud pilih **Create app**, repository/branch, dan entrypoint
   **`app.py`**, bukan `dashboard.py`.
3. Buka **Advanced settings → Python version → 3.12 → Save**, lalu **Deploy**.
   Periksa log untuk memastikan `Using Python 3.12...`. Versi interpreter tidak
   dipilih oleh `requirements.txt`.
4. Pilih subdomain untuk varian IndoBERT, lalu uji endpoint backend di bawah.
   Pengaturan visibility harus mengizinkan akses yang dibutuhkan aplikasi pemanggil.

Jika paket tetap diletakkan dalam subfolder repository, pilih
`streamlit-indobert-api/app.py` sebagai entrypoint, letakkan dependency file di
sebelah entrypoint, dan letakkan konfigurasi `.streamlit/config.toml` pada root
repository sesuai [aturan organisasi file Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).

Jika app terlanjur memakai versi Python lain, Community Cloud memerlukan
deployment dibuat ulang untuk mengganti interpreter. Catat subdomain, repository,
branch, entrypoint, dan secrets sebelum mengganti deployment.
[Petunjuk resmi versi Python](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python).

## Endpoint publik dan format request

Pada routing Community Cloud yang sudah diverifikasi pada varian TF-IDF, backend
memakai prefix `/~/+`. Ganti `NAMA-APP-INDOBERT` dengan subdomain deployment baru:

```text
POST https://NAMA-APP-INDOBERT.streamlit.app/~/+/predict
GET  https://NAMA-APP-INDOBERT.streamlit.app/~/+/health
GET  https://NAMA-APP-INDOBERT.streamlit.app/~/+/docs
GET  https://NAMA-APP-INDOBERT.streamlit.app/~/+/openapi.json
```

Alamat tanpa `/~/+` dapat mengembalikan halaman HTML Cloud. Prefix ini berasal
dari proxy platform; periksa kembali routing jika layanan hosting mengubahnya.
Route Python dan URL localhost tetap `/predict`, `/health`, dan seterusnya.

```json
{"text":"jalan berlubang dan banyak sampah","threshold":0.5}
```

Response JSON berisi `labels`, `probabilities`, dan `threshold`. Label tersedia:
`SAMPAH`, `LUBANG`, `KARATAN`, `RETAKAN`. Threshold opsional; teks harus 1–20.000
karakter setelah trim, dan threshold harus `0 < threshold <= 1`. Teks dipotong
sampai 128 token sesuai konfigurasi asli. Input tidak valid menghasilkan HTTP 422.

```powershell
Invoke-RestMethod -Uri https://NAMA-APP-INDOBERT.streamlit.app/~/+/predict -Method Post -ContentType 'application/json' -Body '{"text":"jalan berlubang dan banyak sampah"}'
```

Untuk pengujian UI, jalankan `api-tester/app.py` dari proyek asal dan ganti URL
endpoint ke deployment IndoBERT baru. Tester yang sama dapat dipakai karena
kontrak request/response tetap sama. `/docs` menyediakan petunjuk HTML dan
`/openapi.json` menyediakan skema API.

## Profil CPU dan startup

- Satu model dimuat saat startup dan dibagikan antara UI serta REST.
- Satu inferensi aktif; request REST lain menunggu secara async.
- Intra/inter-op ONNX satu thread, eksekusi sekuensial, tanpa spinning.
- BLAS/OpenMP/tokenizer dibatasi satu thread. Affinity Linux memilih satu CPU
  yang diizinkan sistem; `CPU_AFFINITY=0` menonaktifkannya. Windows hanya memakai
  batas thread. Penolakan affinity dicatat dan tidak menggagalkan layanan.
- Cache RAM maksimal 128 teks yang dinormalisasi; probabilitas dipakai ulang
  saat threshold berubah. `PREDICTION_CACHE_SIZE=0` mematikan cache.
- Dashboard menghitung prediksi hanya ketika form dikirim. Watcher file dan
  telemetry dimatikan pada konfigurasi paket.

`MODEL_PATH` dapat menunjuk folder model lain yang memiliki format kompatibel.
Model yang lebih besar ini membutuhkan lebih banyak memori dan waktu startup
daripada TF-IDF. Community Cloud tetap memiliki resource limits dan hibernation
setelah 12 jam tanpa trafik; respons saat app belum siap mungkin bukan JSON.
[Batas Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).

## Menjalankan lokal

Dengan environment Streamlit proyek yang memiliki dependensi paket ini:

```powershell
.\.venv-streamlit\Scripts\python.exe -m pip install -r streamlit-indobert-api/requirements.txt
.\.venv-streamlit\Scripts\python.exe -m streamlit run streamlit-indobert-api/app.py --server.port 8502
```

Dashboard: `http://localhost:8502/`; API: `http://localhost:8502/predict`.
Untuk environment baru, gunakan Python 3.12 dan install `requirements.txt` dari
folder paket. Tidak perlu memasang dependency training atau scikit-learn.

## Verifikasi dari root proyek asal

Install dependency paket dan `httpx` pada environment pengujian, lalu jalankan
masing-masing suite pada proses terpisah:

```powershell
python -m unittest discover -s tests -p test_streamlit_indobert_api.py -v
python tests/smoke_streamlit_cli.py --space streamlit-indobert-api
```

Paket disiapkan untuk deployment; belum diunggah atau dideploy ke akun pengguna.
