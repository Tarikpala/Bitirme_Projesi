import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# RFM skorlarını yükle
rfm_dosyasi = "Analiz/rfm_skorlari.csv"
rfm_data = pd.read_csv(rfm_dosyasi)

# Yeni hedef değişkeni oluştur (1: Lost Customers, 0: Other)
rfm_data['Churn'] = (rfm_data['Segment'] == 'Lost Customers').astype(int)

# Özellikleri ve hedefi ayır
X = rfm_data[['Recency', 'Frequency', 'Monetary']]
y = rfm_data['Churn']

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Modeli tanımla ve eğit
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Tahminler yap
y_pred = model.predict(X_test)

# Sonuçları değerlendir
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))