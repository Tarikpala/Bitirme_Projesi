import pandas as pd
import matplotlib.pyplot as plt

# RFM skorlarının yüklenmesi
rfm = pd.read_csv("Analiz/rfm_skorlari.csv")

# Boxplot oluşturulması
metrikler = ['Recency', 'Frequency', 'Monetary']
for metrik in metrikler:
    plt.figure(figsize=(8, 6))
    plt.boxplot(rfm[metrik].dropna(), vert=False)
    plt.title(f"{metrik} için Boxplot")
    plt.xlabel(metrik)
    plt.savefig(f"Analiz/boxplot_{metrik}.png")  # Sonuçları kaydet
    plt.show()