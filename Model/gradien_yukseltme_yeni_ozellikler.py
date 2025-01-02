import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# Yeni temizlenmis veriyi yükle
temiz_veri_yolu = "Veri/temizlenmis_online_retail_II.csv"
data = pd.read_csv(temiz_veri_yolu)

# Ozellikler ve hedef degisken
X = data[['Days_Since_Last_Purchase', 'Weekday', 'Month', 'Total Spending',
          'Frequency', 'Avg_Spending', 'Recent_Spending', 'Country_Code']]  # Ozellikler
y = (data['Days_Since_Last_Purchase'] > 180).astype(int)  # Hedef: Musteri kaybi (ornek)

# Veriyi egitim ve test setine bol
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Yukselme modelini olustur ve egit
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)

# Performans metriklerini goster
print("Dogruluk Orani:", accuracy_score(y_test, y_pred))
print("\nSiniflandirma Raporu:\n", classification_report(y_test, y_pred))

# Ozellik onem sirasini goster
feature_importances = pd.DataFrame({
    'Ozellik': X.columns,
    'Onem': model.feature_importances_
}).sort_values(by='Onem', ascending=False)

print("\nOzellik Onem Sirasi:")
print(feature_importances)