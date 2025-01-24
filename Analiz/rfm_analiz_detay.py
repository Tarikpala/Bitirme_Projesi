import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# RFM skorlarının yüklenmesi
rfm = pd.read_csv("Analiz/rfm_skorlari.csv")

# Segment bazında özet
segment_ozeti = rfm.groupby('Segment').agg({
    'Customer ID': 'count',
    'Monetary': 'sum',
    'Frequency': 'mean',
    'Recency': 'mean'
}).rename(columns={
    'Customer ID': 'Müşteri Sayısı',
    'Monetary': 'Toplam Harcama',
    'Frequency': 'Ortalama Sıklık',
    'Recency': 'Ortalama Yakınlık'
}).reset_index()

# Özet sonuçların kaydedilmesi
segment_ozeti.to_csv("Analiz/segment_ozeti.csv", index=False)
print("Segment Özet Dosyası Kaydedildi: Analiz/segment_ozeti.csv")
print(segment_ozeti)

# Müşteri sayısının görselleştirilmesi
plt.figure(figsize=(10, 6))
sns.barplot(x='Segment', y='Müşteri Sayısı', data=segment_ozeti)
plt.title("Segment Bazında Müşteri Sayısı")
plt.xlabel("Segment")
plt.ylabel("Müşteri Sayısı")
plt.xticks(rotation=45)
plt.savefig("Analiz/segment_musteri_sayisi.png")
plt.show()

# Toplam harcamanın görselleştirilmesi
plt.figure(figsize=(10, 6))
sns.barplot(x='Segment', y='Toplam Harcama', data=segment_ozeti)
plt.title("Segment Bazında Toplam Harcama")
plt.xlabel("Segment")
plt.ylabel("Toplam Harcama")
plt.xticks(rotation=45)
plt.savefig("Analiz/segment_toplam_harcama.png")
plt.show()

# Segment bazında Recency dağılımını görselleştirme
plt.figure(figsize=(10, 6))
sns.boxplot(data=rfm, x="Segment", y="Recency")
plt.title("Segmentlere Göre Recency Dağılımı")
plt.xlabel("Segment")
plt.ylabel("Recency (Gün)")
plt.xticks(rotation=45)
plt.show()

# Ortalama Recency'ye göre churn süresi belirleme
recency_means = rfm.groupby("Segment")["Recency"].mean()

# Her segment için churn süresi
segment_churn_days = {segment: int(recency * 1.5) for segment, recency in recency_means.items()}  # Ortalama Recency'nin %50 fazlası
print(segment_churn_days)

# Çarpıklık ve basıklık analizi
metrikler = ['Recency', 'Frequency', 'Monetary']
for metrik in metrikler:
    skewness = skew(rfm[metrik])
    kurt = kurtosis(rfm[metrik])
    print(f"{metrik} için Çarpıklık (Skewness): {skewness:.2f}")
    print(f"{metrik} için Basıklık (Kurtosis): {kurt:.2f}")

    # Histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(rfm[metrik], bins=20, kde=True, color='blue')
    plt.title(f"{metrik} için Histogram")
    plt.xlabel(metrik)
    plt.ylabel("Frekans")
    plt.savefig(f"Analiz/histogram_{metrik}.png")
    plt.show()

# Çıktıların kontrol edilmesi
print("\nRFM Çarpıklık Analizi ve Detaylar Tamamlandı.")