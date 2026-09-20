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
    CI = (t - n) / (n - 1)
    RI = RI_TABLE.get(n, 1.49)      # default kalau n > 10
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
# CONTOH KASUS: MEMBELI HP (N70, N73, N80, N90)
# Kriteria: Harga, Memori, Warna, Kamera, Berat, Keunikan

# --- 1. Matriks perbandingan KRITERIA (H, M, W, K, B, U) ---
A_kriteria = np.array([
    [1,     5,    5,    5,    3,    3   ],
    [1/5,   1,    1,    1,    1/3,  1/3 ],
    [1/5,   1,    1,    1,    1/3,  1/3 ],
    [1/5,   1,    1,    1,    1/3,  1/3 ],
    [1/3,   3,    3,    3,    1,    1   ],
    [1/3,   3,    3,    3,    1,    1   ],
])
kriteria_names = ["Harga", "Memori", "Warna", "Kamera", "Berat", "Keunikan"]

w_kriteria = print_matrix_info("Bobot Kriteria", A_kriteria)

# --- 2. Matriks perbandingan ALTERNATIF untuk tiap kriteria ---
# Data properti HP: Harga, Memori, Warna, Kamera, Berat
alternatif_names = ["N70", "N73", "N80", "N90"]

data = {
    "Harga":  [2.3, 3.1, 3.7, 4.7],   # semakin kecil semakin baik (cost)
    "Memori": [35, 42, 40, 90],        # semakin besar semakin baik (benefit)
    "Kamera": [2, 3.2, 3.2, 2],        # benefit
    "Berat":  [126, 116, 134, 191],    # cost
}

def matrix_from_ratio(values, cost=False):
    """
    Bangun matriks perbandingan berpasangan otomatis dari data kuantitatif.
    cost=True -> semakin kecil nilai semakin baik (mis. harga, berat)
    cost=False -> semakin besar nilai semakin baik (mis. memori, kamera)
    """
    values = np.array(values, dtype=float)
    n = len(values)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if cost:
                A[i, j] = values[j] / values[i]
            else:
                A[i, j] = values[i] / values[j]
    return A

alt_weights = {}
for kriteria, nilai in data.items():
    cost = kriteria in ["Harga", "Berat"]
    A_alt = matrix_from_ratio(nilai, cost=cost)
    w = print_matrix_info(f"Alternatif berdasarkan {kriteria}", A_alt)
    alt_weights[kriteria] = w

# Untuk kriteria "Warna" (kualitatif) & "Keunikan" (kualitatif),
# isi manual berdasarkan penilaian subjektif pengguna (skala Saaty 1-9)
A_warna = np.array([
    [1,    1,    1,    1/64],
    [1,    1,    1,    1/64],
    [1,    1,    1,    1/64],
    [64,   64,   64,   1   ],
])
alt_weights["Warna"] = print_matrix_info("Alternatif berdasarkan Warna (manual)", A_warna)

A_keunikan = np.array([
    [1,     1/2,  1/3,  1/6],
    [2,     1,    2/3,  1/3],
    [3,     3/2,  1,    1/2],
    [6,     3,    2,    1  ],
])
alt_weights["Keunikan"] = print_matrix_info("Alternatif berdasarkan Keunikan (manual)", A_keunikan)

# ============================================================
# 3. PERANKINGAN AKHIR
# sj = sum( s_ij * w_i )
# ============================================================

S = np.zeros(len(alternatif_names))
for idx, kriteria in enumerate(kriteria_names):
    S += alt_weights[kriteria] * w_kriteria[idx]

print("\n=== HASIL AKHIR PERANKINGAN ===")
for nama, skor in zip(alternatif_names, S):
    print(f"{nama} : {skor:.4f}")

pemenang = alternatif_names[np.argmax(S)]
print(f"\n>>> Alternatif terbaik: {pemenang} (skor tertinggi)")