import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükle
ana_veri_dosyasi = "Veri/temizlenmis_online_retail_II.csv"
musteri_ozeti_dosyasi = "Analiz/musteri_ozeti.csv"
rfm_skorlari_dosyasi = "Analiz/rfm_skorlari.csv"

data = pd.read_csv(ana_veri_dosyasi)
musteri_ozeti = pd.read_csv(musteri_ozeti_dosyasi)
rfm_skorlari = pd.read_csv(rfm_skorlari_dosyasi)

# InvoiceDate'in datetime formatına dönüştürülmesi
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# RFM skorlarını ana veri setine ekle
data = data.merge(rfm_skorlari[['Customer ID', 'Segment']], on='Customer ID', how='left')

# Segmentleri sayısal kodlara çevir
label_encoder = LabelEncoder()
data['Segment_Encoded'] = label_encoder.fit_transform(data['Segment'])

# İşlem düzenliliği (Transaction Regularity) metriğini hesapla
# Her müşteri için toplam işlem (Invoice) sayısını toplam satın alma gününe böl
data['Transaction_Regularity'] = (
    data.groupby('Customer ID')['Invoice'].transform('nunique') / 
    data.groupby('Customer ID')['InvoiceDate'].transform(lambda x: (x.max() - x.min()).days + 1)
)

# İşlem düzenliliği eksik değerlerini doldur
data['Transaction_Regularity'] = data['Transaction_Regularity'].fillna(0)

# Hedef sütun oluşturma: Müşteri kaybı Churn sütunu ile tanımlandı

# `musteri_ozeti` dosyasından `Churn` sütununu ana veri setine ekle
if 'Churn' not in musteri_ozeti.columns:
    raise ValueError("Müşteri özetinde 'Churn' sütunu bulunamadı. Lütfen müşteri analizi adımlarını kontrol edin.")

# Customer ID üzerinden birleştirme
data = data.merge(musteri_ozeti[['Customer ID', 'Churn']], on='Customer ID', how='left')

# Eğer `Churn` sütunu hâlâ eksikse hata ver
if 'Churn' not in data.columns or data['Churn'].isnull().any():
    raise ValueError("Birleştirme sonrası 'Churn' sütunu eksik veya hatalı. Lütfen veri kaynaklarını kontrol edin.")

# Hedef sütun: Churn
y = data['Churn']

# Bazı sütunları model performasına nitelikli bir etki yapmadığı veya overfit (fazla öğrenim ve dengesizlik) yaptığı için çıkarıyoruz.
cikarilan_sutunlar = [
    'Invoice', 'Description', 'InvoiceDate', 'Country', 'Country_Code',
    'Customer ID', 'Month', 'Weekday', 'Total Spending', 'Price',
    'Avg_Spending', 'Cancelled', 'Segment', 'Quantity', 'Frequency'
]
X = data.drop(columns=cikarilan_sutunlar + ['Churn'])

# Eğitim ve test verisi ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE kullanarak veriyi dengeliyoruz
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# SMOTE sonrası boyutları kontrol ediyoruz
print(f"SMOTE sonrası eğitim veri boyutları: {X_train_smote.shape}, {y_train_smote.shape}")
print(f"Test veri boyutları: {X_test.shape}, {y_test.shape}")

# Özelliklerin ölçeklendiriyoruz
scaler = StandardScaler()
X_train_smote = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Hiperparametre araması ile modelin öğrenme düzeyini ayarlıyor, overfit olmasını engelleyip dengeli hale gelmesini sağlıyoruz
param_grid = {
    'n_estimators': [100],
    'learning_rate': [0.01],
    'max_depth': [3],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [10, 20]
}

grid_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=2,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1  # Daha hızlı model eğitimi için bilgisayarın tüm çekirdekleri kullanıyoruz
)

grid_search.fit(X_train_smote, y_train_smote)

# En iyi parametreleri yazdır
print("En iyi parametreler:", grid_search.best_params_)

# En iyi modeli kullan
best_model = grid_search.best_estimator_

# Model performansı değerlendirme
cross_val_scores = cross_val_score(best_model, X_train_smote, y_train_smote, cv=5, scoring='accuracy')
print(f"Cross-Validation Sonuçları: {cross_val_scores}")
print(f"Cross-Validation Ortalama Doğruluk: {np.mean(cross_val_scores):.4f}")

# Özellik önem sırasını elde et
ozellik_onemi = pd.DataFrame({
    'Ozellik': X.columns,
    'Onem': best_model.feature_importances_
}).sort_values(by='Onem', ascending=False)

# Özellik önemini yazdır
print("\nÖzellik Önem Sırası:")
print(ozellik_onemi)

# Özellik önemini görselleştirme
plt.figure(figsize=(10, 6))
sns.barplot(x='Onem', y='Ozellik', data=ozellik_onemi)
plt.title("Özellik Önem Sırası")
plt.xlabel("Önem Skoru")
plt.ylabel("Özellikler")
plt.show()

# Özellik önemini CSV dosyasına kaydet
ozellik_onem_dosyasi = "Analiz/churnm_ozellik_onem_sirasi.csv"
ozellik_onemi.to_csv(ozellik_onem_dosyasi, index=False)

# Test verisi ile tahmin yap
y_pred = best_model.predict(X_test_scaled)

# Sınıflandırma raporu
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# En iyi modeli kaydet
model_dosyasi = "Model/churn_model.pkl"
joblib.dump(best_model, model_dosyasi)

ozellik_onem_dosyasi = "Analiz/churnm_rfm_ozellik_onem_sirasi.csv"
ozellik_onemi.to_csv(ozellik_onem_dosyasi, index=False)

print("RFM ile zenginleştirilmiş churn modeli tamamlandı.")

# Sınıflandırma raporunu kaydet
siniflandirma_raporu_dosyasi = "Analiz/churn_siniflandirma_raporu.csv"
rapor = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(rapor).transpose().to_csv(siniflandirma_raporu_dosyasi, index=False)