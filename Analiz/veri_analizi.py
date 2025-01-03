import pandas as pd
import matplotlib.pyplot as plt

# Veri seti yolu
veri_dosyasi = "Veri/temizlenmis_online_retail_II.csv"

# Veri setini yükle
data = pd.read_csv(veri_dosyasi)

# Hedef sütun oluşturma
ortalama_harcama = data['Total Spending'].mean()
data['Hedef'] = (data['Total Spending'] > ortalama_harcama).astype(int)

# Veri seti bilgileri
print("Veri Seti Genel Bilgisi:")
print(data.info())

# Veri setinin ilk 5 satırını yazdır
print("\nİlk 5 Satır:")
print(data.head())

# Temel istatistiksel özet
print("\nİstatistiksel Özet:")
print(data.describe())

# Eksik değer kontrolü
print("\nEksik Değerler:")
eksik_degerler = data.isnull().sum()
print(eksik_degerler[eksik_degerler > 0])

# Hedef değişken analizi
print("\nHedef Değişken Dağılımı:")
print(data['Hedef'].value_counts(normalize=True).rename_axis('Hedef').reset_index(name='Oran'))

# Korelasyon analizi (sadece sayısal sütunlar)
print("\nKorelasyon Matrisi:")
numeric_columns = data.select_dtypes(include=['float64', 'int64'])
print(numeric_columns.corr())

# Boxplot ile uç değer analizi
for column in data.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(10, 5))
    plt.title(f"{column} için Boxplot")
    plt.boxplot(data[column].dropna())
    plt.show()