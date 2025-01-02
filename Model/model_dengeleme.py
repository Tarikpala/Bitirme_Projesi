import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import joblib

# Veri setini yükle
veri_dosyasi = "Veri/temizlenmis_online_retail_II.csv"  # Klasör yolunu düzelttik
data = pd.read_csv(veri_dosyasi)

# Hedef sütun oluşturma: Ortalama harcamaya göre sınıflandırma
ortalama_harcama = data['Total Spending'].mean()
data['Hedef'] = (data['Total Spending'] > ortalama_harcama).astype(int)

# Veri setindeki sütun isimlerini kontrol edin
print("Veri setindeki sütunlar:")
print(data.columns)

# Özellikler ve hedef
X = data.drop(columns=['Hedef', 'Invoice', 'Description', 'InvoiceDate', 'Country' , 'Total Spending'])  # Gereksiz sütunları kaldırıyoruz
y = data['Hedef']

# Düşük önem taşıyan özellikleri çıkar
X = X.drop(columns=['Days_Since_Last_Purchase', 'Weekday', 'Frequency', 'Country_Code', 'Recent_Spending', 'Avg_Spending', 'Customer ID', 'Month', 'Quantity'])

# Eğitim ve test verisi ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE ile dengeleme
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# SMOTE sonrası boyutları kontrol et
print(f"SMOTE sonrası eğitim veri boyutları: {X_train_smote.shape}, {y_train_smote.shape}")
print(f"Test veri boyutları (SMOTE sonrası değişmemesi lazım): {X_test.shape}, {y_test.shape}")

# Model oluştur ve eğit
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=500, random_state=42)
model.fit(X_train_smote, y_train_smote)

from sklearn.model_selection import cross_val_score
import numpy as np

# Model için cross-validation (5 katmanlı)
cross_val_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

# Sonuçları yazdır
print(f"Cross-Validation Sonuçları: {cross_val_scores}")
print(f"Cross-Validation Ortalama Doğruluk: {np.mean(cross_val_scores):.4f}")

# Özellik önem sırasını elde et
ozellik_onemi = pd.DataFrame({
    'Ozellik': X_train.columns,
    'Onem': model.feature_importances_
}).sort_values(by='Onem', ascending=False)

# Özellik önemini yazdır
print("\nÖzellik Önem Sırası:")
print(ozellik_onemi)

# Özellik önemini CSV dosyasına kaydet
ozellik_onemi.to_csv("Analiz/ozellik_onem_sirasi.csv", index=False)

# Test verisi ile tahmin yap
y_pred = model.predict(X_test)

# Sonuçları yazdır
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# Modeli kaydet
joblib.dump(model, "Model/egitimli_model_smote.pkl")  # Modeli 'Model' klasörüne kaydediyoruz

# Sınıflandırma raporunu kaydet
rapor = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(rapor).transpose().to_csv("Analiz/siniflandirma_raporu_smote.csv", index=False)