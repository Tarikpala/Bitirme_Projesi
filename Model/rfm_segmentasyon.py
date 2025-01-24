import pandas as pd
from datetime import datetime

# Verinin yüklenmesi
veri_dosyasi = "Veri/temizlenmis_online_retail_II.csv"
data = pd.read_csv(veri_dosyasi)

# InvoiceDate'in tarih formatına çevrilmesi (daha güvenli bir kod olması için)
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# "Cancelled" sütununa göre filtreleme (iptal edilen işlemler dahil edilmez)
data = data[~data['Cancelled']]  # Cancelled olanları hariç tut

# Referans tarih tanımlanması (verinin son işlem tarihi baz alındı)
reference_date = data['InvoiceDate'].max()

# RFM Metriklerinin Hesaplanması
rfm = data.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
    'Invoice': 'nunique',                                      # Frequency
    'Total Spending': 'sum'                                    # Monetary
}).reset_index()

# Sütun adlarının düzenlenmesi
rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']

# Recency için segmentasyon yapılması (1: En uzak, 5: En yakın)
rfm['Recency_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])

# Frequency için sıralama bazlı segmentasyon yapılması (1: En az sık, 5: En sık)
rfm['Frequency_Score'] = rfm['Frequency'].rank(method='first', ascending=True).astype(int) #Frequency çakışmalarını önlemek için sırala
rfm['Frequency_Score'] = pd.qcut(rfm['Frequency_Score'], 5, labels=[1, 2, 3, 4, 5])

# Monetary için segmentasyon yapılması (1: En düşük harcama, 5: En yüksek harcama)
rfm['Monetary_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

# Recency ve Frequency skorlarının birleştirilmesi
rfm['RF_Score'] = rfm['Recency_Score'].astype(str) + rfm['Frequency_Score'].astype(str)

# Müşteri segmentlerinin tanımlanması
seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At Risk',
    r'[1-2]5': 'Cant Loose',
    r'3[1-2]': 'About to Sleep',
    r'33': 'Need Attention',
    r'[3-4][4-5]': 'Loyal Customers',
    r'41': 'Promising',
    r'51': 'New Customers',
    r'[4-5][2-3]': 'Potential Loyalists',
    r'5[4-5]': 'Champions'
}

rfm['Segment'] = rfm['RF_Score'].replace(seg_map, regex=True)

# Segment bazında Recency, Frequency, Monetary analizini yap
segment_analizi = rfm[["Segment", "Recency", "Frequency", "Monetary"]].groupby("Segment").agg(["mean", "count"])

# Segment analizi sonuçlarını yazdır
print("Segment Analizi:")
print(segment_analizi)

# Sonuçların kaydedilmesi
rfm.to_csv("Analiz/rfm_skorlari.csv", index=False)

# Çıktıların kontrol edilmesi
print("RFM Analizi Tamamlandı. İlk Satırlar:")
print(rfm.head())