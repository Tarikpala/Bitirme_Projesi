import pandas as pd

# Temizlenmiş veri setini oku
data = pd.read_csv("Veri/temizlenmis_online_retail_II.csv")

# Toplam harcama sütunu oluştur
data['Total'] = data['Quantity'] * data['Price']

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

# Churn etiketi ekle (son 90 gün içinde alışveriş yapmayanlar churn olarak işaretlenir)
son_tarih = pd.to_datetime(data['InvoiceDate']).max()
customer_summary['Churn'] = customer_summary['Last Purchase'].apply(
    lambda x: 1 if (son_tarih - pd.to_datetime(x)).days > 90 else 0
)

# Özet veri setini kaydet
customer_summary.to_csv("Veri/musteri_ozeti.csv", index=False)

# İlk birkaç satırı göster
print("\nMüşteri Bazlı Özet:")
print(customer_summary.head())