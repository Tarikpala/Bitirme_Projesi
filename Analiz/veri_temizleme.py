import pandas as pd
from datetime import datetime

# Doğru dosya yolunu belirt
veri_dosyasi = "Veri/online_retail_II.xlsx"

# Veri setini oku
data = pd.read_excel(veri_dosyasi, engine="openpyxl")

# Veri setinin ilk birkaç satırını görüntüle
print("Veri Setinin İlk Satırları:")
print(data.head())

# Veri hakkında genel bilgi
print("\nVeri Seti Bilgileri:")
print(data.info())

# Eksik değerlerin kontrolü
print("\nEksik Değer Sayıları:")
print(data.isnull().sum())

#Temizleme

# Eksik 'Description' ve 'Customer ID' değerlerini temizle
data.dropna(subset=['Description', 'Customer ID'], inplace=True)

# Negatif miktar ve fiyat değerlerini temizle
data = data[(data['Quantity'] > 0) & (data['Price'] > 0)]

# Cancelled işlemleri sütunu ekle (Rfm'de hariç, Churn'de dahil tutucaz)
data['Cancelled'] = data['Invoice'].astype(str).str.contains('C', na=False)

# Gereksiz sütunları kaldır
data = data.drop(columns=['StockCode'])

# Temizleme işlemleri tamamlandı, şimdi yeni özellik mühendisliği ekleniyor

# Veri setindeki en son işlem tarihini referans tarih olarak belirle
reference_date = data['InvoiceDate'].max()

# 1. Son satın alma tarihinden itibaren geçen gün sayısı
data['Days_Since_Last_Purchase'] = (reference_date - data['InvoiceDate']).dt.days

# 2. Satışların hafta günü ve ay bilgisi
data['Weekday'] = data['InvoiceDate'].dt.weekday  # Pazartesi = 0, Pazar = 6
data['Month'] = data['InvoiceDate'].dt.month

# 3. Ortalama harcama (Total Spending / Frequency)
data['Total Spending'] = data['Quantity'] * data['Price']
data['Frequency'] = data.groupby('Customer ID')['Invoice'].transform('count')
data['Avg_Spending'] = data['Total Spending'] / data['Frequency']

# 4. Son 6 ay içinde yapılan harcama
last_6_months = reference_date - pd.Timedelta(days=180)
recent_spending = data[data['InvoiceDate'] >= last_6_months]
data['Recent_Spending'] = data['Customer ID'].map(
    recent_spending.groupby('Customer ID')['Total Spending'].sum()
)

# 5. Ülke bilgisi kodlama
data['Country_Code'] = data['Country'].astype('category').cat.codes

# Eksik değerleri doldurun
data['Recent_Spending'] = data['Recent_Spending'].fillna(0)

# Özellik mühendisliği bitti

# Temizlenmiş veri setini kaydet
temiz_veri_yolu = "Veri/temizlenmis_online_retail_II.csv"
data.to_csv(temiz_veri_yolu, index=False)

# Temizlenmiş veri seti hakkında bilgi
print("\nTemizlenmiş Veri Seti Bilgileri:")
print(data.info())

# Temizlenmiş veri setinin ilk birkaç satırını görüntüle
print("\nTemizlenmiş Veri Setinin İlk Satırları:")
print(data.head())