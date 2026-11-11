import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text

# 1. Membaca Dataset dan mengambil tepat 11 data pertama secara berurutan
df = pd.read_csv('Mall_Customers.csv').head(11)

# Membuat target biner: Spending Score > 50 -> 1, <= 50 -> 0
df['Target'] = (df['Spending Score (1-100)'] > 50).astype(int)

# Encoding Gender: Male = 0, Female = 1
df['Gender_Enc'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Mendefinisikan Fitur (X) dan Target (y)
X = df[['Age', 'Annual Income (k$)', 'Gender_Enc']]
y = df['Target']

# 2. Sesuai skenario tugas: 10 data pertama sebagai data latih, data ke-11 sebagai data uji
X_train = X.iloc[:10]
y_train = y.iloc[:10]
X_test = X.iloc[10:11]
y_test = y.iloc[10:11]

print("--- INFORMASI DATA (11 Baris Total) ---")
print(f"Jumlah Data Latih : {len(X_train)} baris")
print(f"Jumlah Data Uji   : {len(X_test)} baris (Data ke-11)\n")

# 3. Membangun Model Random Forest (5 Pohon dengan kedalaman max_depth=1)
rf_model = RandomForestClassifier(n_estimators=5, max_features=2, max_depth=1, random_state=42)
rf_model.fit(X_train, y_train)

# Melakukan Prediksi pada Data Uji ke-11
y_pred = rf_model.predict(X_test)

print("--- HASIL PREDIKSI DATA UJI KE-11 ---")
print(f"Prediksi Model : Kelas {int(y_pred[0])}")
print(f"Target Asli    : Kelas {int(y_test.values[0])}\n")

# 4. Menampilkan Struktur 5 Pohon ke Terminal
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
axes = axes.flatten()

for i, tree in enumerate(rf_model.estimators_):
    plot_tree(tree, 
              feature_names=['Age', 'Annual Income (k$)', 'Gender_Enc'], 
              class_names=['Kelas 0', 'Kelas 1'], 
              filled=True, 
              rounded=True, 
              ax=axes[i])
    axes[i].set_title(f"Pohon ke-{i+1}")


fig.delaxes(axes[5])

plt.tight_layout()
plt.show()