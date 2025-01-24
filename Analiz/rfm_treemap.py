import pandas as pd
import matplotlib.pyplot as plt
import squarify

# RFM skorlarının yüklenmesi
rfm_dosyasi = "Analiz/rfm_skorlari.csv"
rfm = pd.read_csv(rfm_dosyasi)

# Segment bazında müşteri sayısı ve toplam harcamayı hesapla
segment_analizi = rfm.groupby('Segment').agg({
    'Customer ID': 'count',  # Müşteri sayısı
    'Monetary': 'sum'        # Toplam harcama
}).reset_index()

# Sütun isimlerini düzenle
segment_analizi.columns = ['Segment', 'Customer_Count', 'Total_Spending']

# Treemap görselleştirmesi için hazırla
labels = [
    f"{row['Segment']}\nMüşteri Sayısı: {row['Customer_Count']}\nToplam Harcama: {row['Total_Spending']:.2f}"
    for _, row in segment_analizi.iterrows()
]

sizes = segment_analizi['Customer_Count']  # Boyut olarak müşteri sayısını kullan
colors = plt.cm.Spectral(sizes / sizes.max())  # Segmentlere göre renk oluştur

# Treemap oluştur
plt.figure(figsize=(16, 8))
squarify.plot(
    sizes=sizes,
    label=labels,
    color=colors,
    alpha=0.8
)
plt.axis('off')
plt.title("RFM Segmentasyonu - Müşteri Sayısı ve Harcama Treemap", fontsize=16)
plt.savefig("Analiz/rfm_treemap.png")
plt.show()