# ============================================================
# Nama      : Inaaya Azeen Nadira & Nena Haryadi Puspanegara
# NPM       : 140810240024 & 140810240034
# Deskripsi : Tugas DSS AHP dengan hardcode data untuk pemilihan lokasi ruang himpunan mahasiswa
# ============================================================

import numpy as np

# Deklarasi Tabel Indeks Random (RI) standar Saaty berdasarkan ukuran matriks (n).
# Digunakan sebagai pembagi untuk menghitung Consistency Ratio (CR).
RI_TABLE = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def normalize_matrix(A):
    """
    Langkah 1 AHP: Normalisasi matriks perbandingan berpasangan.
    Membagi setiap elemen kolom dengan total jumlah kolomnya, sehingga total nilai pada setiap kolom menjadi bernilai 1.
    """
    col_sum = A.sum(axis=0)  # Total nilai tiap kolom
    return A / col_sum       # Membagi tiap elemen dengan total kolomnya


def calc_weights(A):
    """
    Langkah 2 AHP: Menghitung Vektor Bobot / Priority Vector (w).
    Mengambil nilai rata-rata dari tiap baris matriks yang sudah dinormalisasi.
    Total penjumlahan seluruh nilai dalam vektor w selalu bernilai 1 (100%).
    """
    A_norm = normalize_matrix(A)
    w = A_norm.mean(axis=1)  # Rata-rata per baris
    return w


def consistency_ratio(A, w):
    """
    Langkah 3 AHP: Uji Konsistensi Matriks.
    Memastikan penilaian/rasio perbandingan berpasangan rasional dan konsisten.
    """
    n = A.shape[0]  # Ukuran/ordo matriks (jumlah kriteria atau alternatif)

    # Perkalian matriks A dengan vektor bobot w
    Aw = A @ w

    # Menghitung Lambda Max (t), yaitu rata-rata elemen (A @ w)_i / w_i
    t = np.mean(Aw / w)

    # Menghitung Consistency Index (CI)
    # CI = (Lambda_Max - n) / (n - 1)
    CI = (t - n) / (n - 1) if n > 1 else 0

    # Ambil Random Index (RI) sesuai ukuran matriks n
    RI = RI_TABLE.get(n, 1.49)  # Default 1.49 jika n > 10

    # Menghitung Consistency Ratio (CR)
    # Jika CR <= 0,1 (10%), penilaian dianggap konsisten dan valid.
    CR = CI / RI if RI != 0 else 0
    return CI, CR, t


def print_matrix_info(name, A):
    """
    Fungsi untuk membantu proses perhitungan bobot (w), menampilkan matriks, serta mencetak statistik uji konsistensi (CI & CR)
    """
    w = calc_weights(A)
    CI, CR, t = consistency_ratio(A, w)
    print(f"\n=== {name} ===")
    print("Matriks perbandingan berpasangan:")
    print(A)
    print("Vektor bobot (w):", np.round(w, 4))
    print(f"Lambda max (t) : {t:.4f}")
    print(f"CI : {CI:.4f}")
    print(f"CR : {CR:.4f}  -> {'KONSISTEN' if CR <= 0.1 else 'TIDAK KONSISTEN'}")
    return w


# ==================================================
# KASUS: PEMILIHAN LOKASI RUANG HIMPUNAN
# Alternatif: A01 (Lokasi 1), A02 (Lokasi 2), A03 (Lokasi 3)
# Kriteria  : C01 (Kos), C02 (Kampus), C03 (Pujasera), C04 (Biaya), C05 (Luas)
# ==================================================

# --- 1. MATRIKS PERBANDINGAN KRITERIA UTAMA (C01 s.d. C05) ---
# Menilai seberapa penting satu kriteria dibanding kriteria lain (skala 1-9 Saaty).
# Contoh bacanya: Baris 1 Kolom 3 bernilai 3 -> C01 (Jarak kos) 3x lebih penting dari C03 (Jarak pujasera).
A_kriteria = np.array([
    [1,     1,   3,   1,     3],  # C01 (Jarak kos)
    [1,     1,   2,   1,     1],  # C02 (Jarak kampus)
    [1/3, 0.5,   1,   1,     2],  # C03 (Jarak pujasera)
    [1,     1,   1,   1,     3],  # C04 (Biaya)
    [1/3,   1, 0.5, 1/3,     1],  # C05 (Luas bangunan)
])
kriteria_names = [
    "C01 (Jarak kos)",
    "C02 (Jarak kampus)",
    "C03 (Jarak pujasera)",
    "C04 (Biaya)",
    "C05 (Luas bangunan)"
]

# Hitung dan cetak bobot kepentingan tiap kriteria utama
w_kriteria = print_matrix_info("Bobot Kriteria Utama", A_kriteria)


# --- 2. MATRIKS PERBANDINGAN ALTERNATIF UNTUK TIAP KRITERIA ---
# Membandingkan ketiga lokasi (A01, A02, A03) secara spesifik di tiap kriteria.
alternatif_names = ["A01 (Lokasi 1)", "A02 (Lokasi 2)", "A03 (Lokasi 3)"]

# Membandingkan keunggulan lokasi berdasarkan Jarak ke Kos (C01)
A_C01 = np.array([
    [1,     3, 3],
    [1/3,   1, 2],
    [1/3, 0.5, 1]
])

# Membandingkan keunggulan lokasi berdasarkan Jarak ke Kampus (C02)
A_C02 = np.array([
    [1,     2, 4],
    [0.5,   1, 3],
    [0.25, 1/3, 1]
])

# Membandingkan keunggulan lokasi berdasarkan Jarak ke Pujasera (C03)
A_C03 = np.array([
    [1,   2, 1],
    [0.5, 1, 2],
    [1, 0.5, 1]
])

# Membandingkan keunggulan lokasi berdasarkan Biaya (C04)
A_C04 = np.array([
    [1,     2, 3],
    [0.5,   1, 6],
    [1/3, 1/6, 1]
])

# Membandingkan keunggulan lokasi berdasarkan Luas Bangunan (C05)
A_C05 = np.array([
    [1,    4, 3],
    [0.25, 1, 2],
    [1/3, 0.5, 1]
])

# Mengelompokkan semua matriks alternatif ke dalam struktur dictionary untuk memudahkan iterasi
alt_matrices = {
    "C01 (Jarak kos)": A_C01,
    "C02 (Jarak kampus)": A_C02,
    "C03 (Jarak pujasera)": A_C03,
    "C04 (Biaya)": A_C04,
    "C05 (Luas bangunan)": A_C05
}

# Hitung dan simpan vektor bobot alternatif untuk masing masing kriteria
alt_weights = {}
for kriteria_nama, A_alt in alt_matrices.items():
    w = print_matrix_info(f"Alternatif berdasarkan {kriteria_nama}", A_alt)
    alt_weights[kriteria_nama] = w


# ===============================================
# 3. PERANKINGAN AKHIR (TOTAL SKOR AHP)
# ===============================================
# Total Skor Alternatif_j = Sum(Bobot Alternatif_j pada Kriteria_i * Bobot Kriteria_i)

S = np.zeros(len(alternatif_names))
for idx, kriteria_nama in enumerate(kriteria_names):
    # Akumulasi nilai dengan perkalian bobot kriteria dengan bobot lokasi pada kriteria tersebut
    S += alt_weights[kriteria_nama] * w_kriteria[idx]

print("\n=== HASIL AKHIR PERANKINGAN ===")
for nama, skor in zip(alternatif_names, S):
    print(f"{nama} : {skor:.4f}")

# Menentukan lokasi terbaik berdasarkan skor tertinggi
pemenang = alternatif_names[np.argmax(S)]
print(f"\n=== Alternatif terbaik: {pemenang} (skor tertinggi) ===")