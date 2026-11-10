# TF-IDF REST API di Streamlit Community Cloud

Paket mandiri tanpa Docker dan tanpa Hugging Face Spaces. Memakai **Streamlit
1.67.0** dan public API `st.App(routes=...)` untuk menyediakan dashboard dan
endpoint REST pada **port yang sama**. Tidak membuka server FastAPI di port lain.

Sudah diuji lokal: **12 tes integrasi lulus** dan startup asli
`streamlit run app.py` berhasil melayani UI, `/health`, dan `POST /predict` pada
port yang sama. Bobot dan probabilitas tetap cocok dengan model TF-IDF asal.
Deploy di Community Cloud pengguna belum dilakukan.

Streamlit menambahkan dukungan ASGI/custom HTTP melalui `st.App` pada 1.53 dan
menjadikan Starlette/Uvicorn server bawaan pada 1.57. Tutorial lama yang menyatakan
Streamlit tidak dapat menyediakan REST API tidak menggambarkan kemampuan versi
yang dipin paket ini. Lihat [release notes resmi](https://docs.streamlit.io/develop/quick-reference/release-notes/2026).

## Deploy ke Community Cloud

1. Upload isi folder ini ke repository GitHub yang dapat diakses akun Streamlit.
   File yang diperlukan: `app.py`, `dashboard.py`, `engine.py`, `model.joblib`,
   `requirements.txt`, dan `.streamlit/config.toml`.
2. Buka [Streamlit Community Cloud](https://share.streamlit.io/), pilih **Create app**,
   lalu pilih repository dan branch.
3. Entrypoint **harus `app.py`**, bukan `dashboard.py`. Jika folder ini masih
   bersarang di repository, pilih `streamlit-tfidf-api/app.py` dan pastikan
   `requirements.txt` serta `.streamlit/` tetap berada di sebelah entrypoint.
4. Pada Advanced settings, pilih **Python 3.12**, lalu deploy.
5. Agar aplikasi lain dapat memanggil API tanpa login browser, gunakan visibility
   aplikasi yang sesuai untuk akses publik. Repository dan aplikasi tidak harus
   dipublikasikan oleh script; pengaturan akses dilakukan pemilik akun.
6. Setelah startup, uji `https://NAMA-APP.streamlit.app/health`, lalu `/predict`.

Paket tidak memerlukan HF token, server lokal tambahan, atau Dockerfile.
Bobot model dan cache probabilitas sama dengan paket TF-IDF sebelumnya.

## Kontrak REST

```http
POST /predict
Content-Type: application/json

{"text":"jalan berlubang dan banyak sampah","threshold":0.5}
```

Hasil berisi `labels`, `probabilities`, dan `threshold`. Threshold opsional,
default `0.5`. Teks dinormalisasi dengan lowercase dan trim. Teks kosong atau
lebih dari 20.000 karakter, threshold di luar `0 < threshold <= 1`, dan JSON
tidak valid menghasilkan HTTP `422`.

```powershell
Invoke-RestMethod -Uri https://NAMA-APP.streamlit.app/predict -Method Post -ContentType 'application/json' -Body '{"text":"jalan berlubang dan banyak sampah"}'
```

- `GET /`: dashboard Streamlit.
- `GET /health`: status JSON `{"status":"ok"}`.
- `GET /docs`: petunjuk API HTML sederhana.
- `GET /openapi.json`: skema OpenAPI.

`/docs` adalah halaman petunjuk, bukan Swagger UI FastAPI. Route `/predict`
tetap mempertahankan bentuk request/response sukses aplikasi lama.

## Batas layanan gratis

Community Cloud memiliki batas CPU/memori tersendiri. Penggunaan HF CPU Basic
tidak dihitung karena aplikasi ini dijalankan pada layanan Streamlit.
Community Cloud bukan API yang dijamin selalu aktif: aplikasi tanpa trafik
selama **12 jam** tidur. Pengguna mungkin perlu membuka halaman aplikasi dan
menekan tombol untuk membangunkannya sebelum API tersedia kembali. Pemanggil
harus menangani startup lambat dan respons non-JSON saat aplikasi belum siap.
Lihat [hibernation dan resource limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).

Profil aplikasi: satu inferensi aktif, satu thread BLAS/OpenMP, `n_jobs=1`,
cache 128 teks, dan model dimuat sekali saat startup. Dashboard hanya menghitung
prediksi saat form dikirim. Tidak ada keep-alive/polling yang sengaja mencegah
hibernation. `CPU_AFFINITY=0` dapat menonaktifkan affinity Linux dan
`PREDICTION_CACHE_SIZE=0` dapat mematikan cache.

Paket siap diuji/deploy, tetapi belum diunggah ke akun GitHub/Streamlit pengguna.

## Lokal tanpa Docker

Python 3.12, dari folder paket:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Buka `http://localhost:8501/` atau uji `http://localhost:8501/predict`.
Pada Linux gunakan `.venv/bin/python`. Gunakan satu proses aplikasi.

Untuk mengulang tes dari root proyek asal, install dependensi paket dan `httpx`,
lalu jalankan pada environment terpisah dari tes Gradio/FastAPI:

```powershell
python -m unittest discover -s tests -p test_streamlit_api.py -v
python tests/smoke_streamlit_cli.py
```
