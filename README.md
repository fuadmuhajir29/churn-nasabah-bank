# Prediksi Churn Nasabah Bank

Tugas Besar Machine Learning (Kelas Y7L)
**Nama:** Fuad Muhajir Hilmi | **NPM:** 202343501787

Aplikasi web untuk memprediksi peluang nasabah bank menutup rekening (churn).
Model jaringan saraf (PyTorch) dilatih dengan notebook Google Colab, disimpan sebagai `churn_model.pth`, lalu dimuat oleh aplikasi. Model sudah termasuk di folder `model/`.

- **Backend:** FastAPI (Python), endpoint `/predict`
- **Frontend:** Vue.js 3 (halaman tunggal, `templates/index.html`)
- **Model:** `model/churn_model.pth` + `model/preprocess.json`
- **Data:** [Churn Modelling (Kaggle)](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
- **Metodologi:** CRISP-DM

## Struktur folder
```
app.py                  server FastAPI + pemuatan model
model_def.py            arsitektur jaringan (sama dengan notebook)
templates/index.html    antarmuka pengguna (Vue.js)
static/style.css        gaya tampilan
model/                  churn_model.pth dan preprocess.json (hasil dari Colab)
notebook/               notebook training untuk Google Colab
requirements.txt
```

## Cara menjalankan
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app:app --reload
```
Buka http://127.0.0.1:8000

## Cara membuat model
1. Buka `notebook/training_churn_bank.ipynb` di Google Colab.
2. Unggah `Churn_Modelling.csv` dari Kaggle, lalu jalankan semua sel.
3. Unduh `churn_model.pth` dan `preprocess.json`, simpan ke folder `model/`.

## Contoh pemanggilan API
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "CreditScore": 619, "Geography": "France", "Gender": "Female", "Age": 42, "Tenure": 2,
  "Balance": 0, "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 101348.88
}'
```

## Catatan
Aplikasi dibuat untuk keperluan tugas kuliah. Hasil prediksi bukan keputusan akhir bank.
