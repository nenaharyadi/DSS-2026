import numpy as np

# Tabel Indeks Random (RI) untuk uji konsistensi
RI_TABLE = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def normalize_matrix(A):
    """Normalisasi setiap kolom matriks A agar jumlahnya = 1"""
    col_sum = A.sum(axis=0)
    return A / col_sum


def calc_weights(A):
    """Hitung vektor bobot (w) dari matriks perbandingan berpasangan"""
    A_norm = normalize_matrix(A)
    w = A_norm.mean(axis=1)  # rata-rata tiap baris
    return w


def consistency_ratio(A, w):
    """Hitung CI dan CR untuk uji konsistensi matriks A"""
    n = A.shape[0]
    Aw = A @ w                      # (A)(w^T)
    t = np.mean(Aw / w)             # lambda max (rata-rata elemen ke-i / w_i)
    CI = (t - n) / (n - 1) if n > 1 else 0
    RI = RI_TABLE.get(n, 1.49)      # default jika n > 10
    CR = CI / RI if RI != 0 else 0
    return CI, CR, t


def print_matrix_info(name, A):
    w = calc_weights(A)
    CI, CR, t = consistency_ratio(A, w)
    print(f"\n=== {name} ===")
    print("Matriks perbandingan berpasangan:")
    print(A)
    print("Vektor bobot (w):", np.round(w, 4))
    print(f"Lambda max (t)  : {t:.4f}")
    print(f"CI              : {CI:.4f}")
    print(f"CR              : {CR:.4f}  -> {'KONSISTEN' if CR <= 0.1 else 'TIDAK KONSISTEN'}")
    return w


# ============================================================
# KASUS: PEMILIHAN LOKASI RUANG HIMPUNAN
# Alternatif: A01 (Lokasi 1), A02 (Lokasi 2), A03 (Lokasi 3)
# Kriteria  : C01 (Kos), C02 (Kampus), C03 (Pujasera), C04 (Biaya), C05 (Luas)
# ============================================================

# --- 1. Matriks perbandingan KRITERIA (C01 s.d. C05) ---
A_kriteria = np.array([
    [1,     1,   3,   1,     3],
    [1,     1,   2,   1,     1],
    [1/3, 0.5,   1,   1,     2],
    [1,     1,   1,   1,     3],
    [1/3,   1, 0.5, 1/3,     1],
])
kriteria_names = [
    "C01 (Jarak kos)",
    "C02 (Jarak kampus)",
    "C03 (Jarak pujasera)",
    "C04 (Biaya)",
    "C05 (Luas bangunan)"
]

w_kriteria = print_matrix_info("Bobot Kriteria Utama", A_kriteria)

# --- 2. Matriks perbandingan ALTERNATIF untuk tiap kriteria ---
alternatif_names = ["A01 (Lokasi 1)", "A02 (Lokasi 2)", "A03 (Lokasi 3)"]

# Matriks Alternatif berdasarkan C01 (Jarak ke kos)
A_C01 = np.array([
    [1,     3, 3],
    [1/3,   1, 2],
    [1/3, 0.5, 1]
])

# Matriks Alternatif berdasarkan C02 (Jarak ke kampus)
A_C02 = np.array([
    [1,     2, 4],
    [0.5,   1, 3],
    [0.25, 1/3, 1]
])

# Matriks Alternatif berdasarkan C03 (Jarak ke pujasera)
A_C03 = np.array([
    [1,   2, 1],
    [0.5, 1, 2],
    [1, 0.5, 1]
])

# Matriks Alternatif berdasarkan C04 (Biaya)
A_C04 = np.array([
    [1,     2, 3],
    [0.5,   1, 6],
    [1/3, 1/6, 1]
])

# Matriks Alternatif berdasarkan C05 (Luas bangunan)
A_C05 = np.array([
    [1,    4, 3],
    [0.25, 1, 2],
    [1/3, 0.5, 1]
])

alt_matrices = {
    "C01 (Jarak kos)": A_C01,
    "C02 (Jarak kampus)": A_C02,
    "C03 (Jarak pujasera)": A_C03,
    "C04 (Biaya)": A_C04,
    "C05 (Luas bangunan)": A_C05
}

alt_weights = {}
for kriteria_nama, A_alt in alt_matrices.items():
    w = print_matrix_info(f"Alternatif berdasarkan {kriteria_nama}", A_alt)
    alt_weights[kriteria_nama] = w

# ============================================================
# 3. PERANKINGAN AKHIR
# S_j = sum( s_ij * w_i )
# ============================================================

S = np.zeros(len(alternatif_names))
for idx, kriteria_nama in enumerate(kriteria_names):
    S += alt_weights[kriteria_nama] * w_kriteria[idx]

print("\n=== HASIL AKHIR PERANKINGAN ===")
for nama, skor in zip(alternatif_names, S):
    print(f"{nama} : {skor:.4f}")

pemenang = alternatif_names[np.argmax(S)]
print(f"\n>>> Alternatif terbaik: {pemenang} (skor tertinggi)")