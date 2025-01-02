import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import joblib

# Veri setini yükle
veri_dosyasi = "Veri/temizlenmis_online_retail_II.csv"
data = pd.read_csv(veri_dosyasi)

# Hedef sütun oluşturma: Ortalama harcamaya göre sınıflandırma
ortalama_harcama = data['Total Spending'].mean()
data['Hedef'] = (data['Total Spending'] > ortalama_harcama).astype(int)

# Özellikler ve hedef
X = data.drop(columns=['Hedef', 'Invoice', 'Description', 'InvoiceDate', 'Country',
                       'Days_Since_Last_Purchase', 'Weekday', 'Frequency', 'Country_Code'])
y = data['Hedef']

# Eğitim ve test verisi ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE ile dengeleme
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Logistic Regression Modeli
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_smote, y_train_smote)

# Cross-validation ile doğruluk kontrolü
cross_val_scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring='accuracy')
print(f"Cross-Validation Sonuçları: {cross_val_scores}")
print(f"Cross-Validation Ortalama Doğruluk: {np.mean(cross_val_scores):.4f}")

# Test verisi ile tahmin yap
y_pred = model.predict(X_test)

# Sınıflandırma raporu
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# Koefisyanlar üzerinden özellik etkisi
ozellik_etkisi = pd.DataFrame({
    'Ozellik': X.columns,
    'Katsayi': model.coef_[0]
}).sort_values(by='Katsayi', ascending=False)

print("\nÖzellik Etkisi (Katsayılar):")
print(ozellik_etkisi)

# Özellik etkisini CSV dosyasına kaydet
ozellik_etkisi.to_csv("Analiz/ozellik_etkisi_katsayilar.csv", index=False)

# Modeli kaydet
joblib.dump(model, "Model/egitimli_model_logistic.pkl")

# Sınıflandırma raporunu kaydet
rapor = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(rapor).transpose().to_csv("Analiz/siniflandirma_raporu_logistic.csv", index=False)