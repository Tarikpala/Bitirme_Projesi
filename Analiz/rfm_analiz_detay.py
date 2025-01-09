import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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