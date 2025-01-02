import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini yükle
data = pd.read_csv("Veri/musteri_ozeti.csv")

# Country sütununu sayısal formata çevir
data['Country_Code'] = data['Country'].astype('category').cat.codes

# Özellikler ve hedef değişken
X = data[['Total Spending', 'Frequency', 'Tenure', 'Country_Code']]
y = data['Churn']

# Veri dengesini sağlamak için SMOTE kullan
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Eğitim ve test setlerine böl
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Parametre grid'i tanımlayın
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Gradient Boosting modeli oluştur
gb_model = GradientBoostingClassifier(random_state=42)

# GridSearchCV ile en iyi parametreleri bulun
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# En iyi modeli alın
best_model = grid_search.best_estimator_

# Test seti üzerinde tahmin yapın
y_pred_gb = best_model.predict(X_test)

# Sınıflandırma raporu
print("\nEn İyi Parametreler:")
print(grid_search.best_params_)
print("\nSınıflandırma Raporu (Optimize Edilmiş Gradient Boosting):")
print(classification_report(y_test, y_pred_gb))

# Confusion matrix görselleştirme
cm_gb = confusion_matrix(y_test, y_pred_gb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Optimize Edilmiş Gradient Boosting)")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()