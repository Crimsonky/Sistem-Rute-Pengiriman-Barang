# 🚚 Sistem Optimasi Rute Pengiriman Barang - Tebet, Jakarta Selatan

Proyek ini adalah aplikasi berbasis **Streamlit** yang memvisualisasikan rute pengiriman barang di wilayah **Tebet, Jakarta Selatan** menggunakan algoritma **Dijkstra** dan **A\***. Aplikasi ini mengoptimasi pengiriman dari depot ke ruko/gedung berdasarkan prioritas, kapasitas kendaraan, dan permintaan pengiriman.

## 🔥 Fitur
- **Pemilihan Lokasi Awal dan Tujuan**: Pilih titik awal (Depot) dan tujuan (Ruko/Gedung) secara fleksibel.
- **Dukungan Multi-Kendaraan**: Optimasi pengiriman dengan beberapa kendaraan berdasarkan kapasitas.
- **Pemilihan Algoritma**: Bebas memilih antara algoritma **Dijkstra** atau **A\***.
- **Visualisasi Interaktif**: Peta Folium yang menampilkan jalur pengiriman secara real-time.
- **Pengaturan Prioritas dan Permintaan**: Tentukan prioritas dan berat permintaan setiap pengiriman.

## 📦 Dataset
- `transportation_nodes.csv`: Berisi data titik lokasi (Depot, Ruko, Gedung) dengan atribut seperti latitude dan longitude.
- `transportation_edges_augmented.csv`: Berisi data jalan antar titik, termasuk jarak, kecepatan rata-rata, tingkat kemacetan, dan arah jalan.

## ⚙️ Cara Menjalankan

1. **Clone repository ini**:
   ```bash
   git clone https://github.com/username/repository-name.git
   cd repository-name
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi Streamlit**:
   ```bash
   streamlit run app.py
   ```

4. **Akses aplikasi** melalui browser di alamat:
   ```
   http://localhost:8501
   ```

## 🛠️ Struktur File
```
.
├── app.py
├── transportation_nodes.csv
├── transportation_edges_augmented.csv
├── requirements.txt
└── README.md
```

## 📚 Library yang Digunakan
- `streamlit`
- `pandas`
- `networkx`
- `folium`
- `streamlit-folium`
- `scikit-learn` (untuk perhitungan jarak Haversine di A*)

## ✨ Rencana Pengembangan
- Menambahkan dukungan **optimasi rute multiple drops** (banyak tujuan dalam 1 kendaraan).
- Integrasi **penjadwalan waktu** berdasarkan jam operasional.
- Visualisasi tambahan seperti **animasi perjalanan**.

## 🧑‍💻 Kontributor
- Ryan Delon Pratama
- Ferry Saputra
- Rafael Aryapati Soebagijo
- Atong Nazarius
- Rifky Mustaqim Handoko
- Ahmad Iqbal
- Sandy W. Simatupang
