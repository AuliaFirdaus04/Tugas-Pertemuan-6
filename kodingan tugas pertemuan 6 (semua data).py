import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import export_text

# 1. Membaca Seluruh Dataset
df = pd.read_csv('Mall_Customers.csv')

# Membuat target biner: Spending Score > 50 -> 1, <= 50 -> 0
df['Target'] = (df['Spending Score (1-100)'] > 50).astype(int)

# Encoding Gender: Male = 0, Female = 1
df['Gender_Enc'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Mendefinisikan Fitur (X) dan Target (y)
X = df[['Age', 'Annual Income (k$)', 'Gender_Enc']]
y = df['Target']

# 2. Membagi dataset menjadi 80% Latih dan 20% Uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- INFORMASI DATA ---")
print(f"Jumlah Data Latih (80%) : {len(X_train)} baris")
print(f"Jumlah Data Uji (20%)   : {len(X_test)} baris\n")

# 3. Membangun Model Random Forest (25 Pohon)
# n_estimators diubah menjadi 25 sesuai permintaan
rf_model = RandomForestClassifier(n_estimators=25, max_features=2, max_depth=1, random_state=42)
rf_model.fit(X_train, y_train)

# Melakukan Prediksi
y_pred = rf_model.predict(X_test)

# 4. Evaluasi Model
acc = accuracy_score(y_test, y_pred)
print("--- HASIL EVALUASI ---")
print(f"Akurasi Random Forest : {acc * 100:.2f}%\n")

print("--- Laporan Klasifikasi ---")
print(classification_report(y_test, y_pred))

# 5. WAJIB DEFINISIKAN INI TERLEBIH DAHULU (Membuat grid untuk 7 pohon)
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
axes = axes.flatten()

# 6. Baru jalankan looping untuk memplot pohon ke-1 sampai ke-7
for i in range(7):
    plot_tree(rf_model.estimators_[i], 
              feature_names=['Age', 'Annual Income (k$)', 'Gender_Enc'], 
              class_names=['Kelas 0', 'Kelas 1'], 
              filled=True, 
              rounded=True, 
              ax=axes[i])
    axes[i].set_title(f"Pohon ke-{i+1}")

# 7. Menghapus sisa kotak grid kosong yang tidak terpakai
for j in range(7, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()