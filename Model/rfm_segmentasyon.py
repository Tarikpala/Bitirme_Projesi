import pandas as pd
from datetime import datetime

# Verinin yüklenmesi
veri_dosyasi = "Veri/temizlenmis_online_retail_II.csv"
data = pd.read_csv(veri_dosyasi)

# Referans tarih tanımlanması
reference_date = datetime(2025, 1, 1)

# RFM Metriklerinin Hesaplanması
rfm = data.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (reference_date - pd.to_datetime(x.max())).days,  # Recency
    'Invoice': 'count',                                                       # Frequency
    'Total Spending': 'sum'                                                   # Monetary
}).reset_index()

# Sütun adlarının düzenlenmesi
rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']

# Recency için segmentasyon yapılması (1: En yakın, 4: En uzak)
rfm['Recency_Score'] = pd.qcut(rfm['Recency'], 4, labels=[1, 2, 3, 4])

# Frequency için segmentasyon yapılması (1: En az sık, 4: En sık)
rfm['Frequency_Score'] = pd.qcut(rfm['Frequency'], 4, labels=[4, 3, 2, 1])

# Monetary için segmentasyon yapılması (1: En düşük harcama, 4: En yüksek harcama)
rfm['Monetary_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])

# RFM Skorunun oluşturulması
rfm['RFM_Score'] = rfm['Recency_Score'].astype(str) + rfm['Frequency_Score'].astype(str) + rfm['Monetary_Score'].astype(str)

# Müşteri segmentlerinin tanımlanması
rfm['Segment'] = rfm['RFM_Score'].replace({
    '111': 'Best Customers',
    '112': 'Loyal Customers',
    '211': 'Potential Loyalists',
    '411': 'Need Attention',
    '444': 'Lost Customers'
}, regex=True).fillna('Other')

# Sonuçları kaydedilmesi
rfm.to_csv("Analiz/rfm_skorlari.csv", index=False)

# Çıktıları kontrol edilmesi
print("RFM Analizi Tamamlandı. İlk Satırlar:")
print(rfm.head())