import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükle
data = pd.read_csv("Veri/musteri_ozeti.csv")

# Country sütununu sayısal formata çevir (Encoding)
data['Country_Code'] = data['Country'].astype('category').cat.codes

# Özellikler ve hedef
X = data[['Total Spending', 'Frequency', 'Tenure', 'Country_Code']]  # Country_Code eklendi
y = data['Churn']

# Veri dengesini sağlamak için SMOTE kullan
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Eğitim ve test setlerine böl
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Parametre optimizasyonu için GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# En iyi model ile tahmin yap
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Model performansını değerlendirme
print("\nSınıflandırma Raporu (Country Dahil):")
print(classification_report(y_test, y_pred))

# Confusion Matrix Görselleştirme
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Country Dahil)")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()

# En iyi parametreleri yazdır
print("\nEn İyi Parametreler:")
print(grid_search.best_params_)