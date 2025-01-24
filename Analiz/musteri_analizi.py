import pandas as pd

# Verilerin yüklenmesi
temiz_veri_dosyasi = "Veri/temizlenmis_online_retail_II.csv"
rfm_skorlari_dosyasi = "Analiz/rfm_skorlari.csv"

# Temizlenmiş veri setini oku
data = pd.read_csv(temiz_veri_dosyasi)

# Toplam harcama sütunu oluştur
data['Total'] = data['Quantity'] * data['Price']

# RFM skorlarını yükle ve ana veri setine entegre et
rfm_skorlari = pd.read_csv(rfm_skorlari_dosyasi)
data = data.merge(rfm_skorlari[['Customer ID', 'Segment']], on='Customer ID', how='left')

# Müşteri bazında özet veri oluştur
customer_summary = data.groupby(['Customer ID', 'Country']).agg({
    'Total': 'sum',  # Toplam harcama
    'Invoice': 'nunique',  # Fatura sayısı (işlem sıklığı)
    'InvoiceDate': ['min', 'max']  # İlk ve son alışveriş tarihi
}).reset_index()

# Sütun isimlerini düzenle
customer_summary.columns = ['Customer ID', 'Country', 'Total Spending', 'Frequency', 'First Purchase', 'Last Purchase']

# Alışveriş süresi (Tenure) hesapla
customer_summary['Tenure'] = (pd.to_datetime(customer_summary['Last Purchase']) -
                              pd.to_datetime(customer_summary['First Purchase'])).dt.days

# Segmentleri ekle
customer_summary = customer_summary.merge(rfm_skorlari[['Customer ID', 'Segment']], on='Customer ID', how='left')

# En son işlem tarihini al
son_tarih = pd.to_datetime(data['InvoiceDate']).max()

# Segment bazlı churn sürelerini belirleme
segment_churn_thresholds = {
    "Champions": 60,
    "Loyal Customers": 120,
    "At Risk": 90,
    "Hibernating": 180,
    "Potential Loyalists": 90,
    "New Customers": 30,
    "Promising": 60,
    "About to Sleep": 120,
    "Need Attention": 90,
    "Cant Lose Them": 60
}

# Dinamik churn hesaplama fonksiyonu
def calculate_churn(row):
    segment = row["Segment"]
    last_purchase_date = pd.to_datetime(row["Last Purchase"])
    threshold = segment_churn_thresholds.get(segment, 90)  # Varsayılan threshold 90 gün
    days_since_last_purchase = (son_tarih - last_purchase_date).days
    return 1 if days_since_last_purchase > threshold else 0

# Churn hesaplaması
customer_summary["Churn"] = customer_summary.apply(calculate_churn, axis=1)

# Özet veri setini kaydet
customer_summary.to_csv("Analiz/musteri_ozeti.csv", index=False)

# İlk birkaç satırı göster
print("\nMüşteri Bazlı Özet:")
print(customer_summary.head())