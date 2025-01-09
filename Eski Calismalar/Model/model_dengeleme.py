import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
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
X = data.drop(columns=['Hedef', 'Invoice', 'Description', 'InvoiceDate', 'Country','Country_Code', 'Customer ID', 'Month', 'Weekday', 'Frequency']) # Gereksiz sütunları kaldırıyoruz
y = data['Hedef']

# Eğitim ve test verisi ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE ile dengeleme
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# SMOTE sonrası boyutları kontrol et
print(f"SMOTE sonrası eğitim veri boyutları: {X_train_smote.shape}, {y_train_smote.shape}")
print(f"Test veri boyutları: {X_test.shape}, {y_test.shape}")

# Özelliklerin ölçeklendirilmesi
scaler = StandardScaler()
X_train_smote = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Hiperparametre arama
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
    n_jobs=-1  # Tüm çekirdekleri kullan
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

# Özellik önemini CSV dosyasına kaydet
ozellik_onemi.to_csv("Analiz/ozellik_onem_sirasi.csv", index=False)

# Test verisi ile tahmin yap
y_pred = best_model.predict(X_test_scaled)

# Sınıflandırma raporu
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# En iyi modeli kaydet
joblib.dump(best_model, "Model/egitimli_model_gradient_tuned.pkl")

# Sınıflandırma raporunu kaydet
rapor = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(rapor).transpose().to_csv("Analiz/siniflandirma_raporu_gradient.csv", index=False)