import pandas as pd
import matplotlib.pyplot as plt

# RFM skorlarının yüklenmesi
rfm = pd.read_csv("Analiz/rfm_skorlari.csv")

# Histogram oluşturulması
metrikler = ['Recency', 'Frequency', 'Monetary']
for metrik in metrikler:
    plt.figure(figsize=(8, 6))
    plt.hist(rfm[metrik], bins=20, edgecolor='black')
    plt.title(f"{metrik} için Histogram")
    plt.xlabel(metrik)
    plt.ylabel("Frekans")
    plt.savefig(f"Analiz/histogram_{metrik}.png")  # Sonuçları kaydet
    plt.show()