import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Veri setini yükle
data = pd.read_csv("Veri/musteri_ozeti.csv")

# Country sütununu sayısal forma dönüştür
data['Country_Code'] = data['Country'].astype('category').cat.codes

# Özellikler ve hedef değişken
X = data[['Total Spending', 'Frequency', 'Tenure', 'Country_Code']]
y = data['Churn']

# Veri dengesini sağlamak için SMOTE kullan
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Eğitim ve test setlerine böl
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression

# Lojistik regresyon modeli oluştur ve eğit
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)

# Tahmin yap
y_pred_log = log_model.predict(X_test)

# Performans değerlendirme
print("\nSınıflandırma Raporu (Lojistik Regresyon):")
print(classification_report(y_test, y_pred_log))

# Confusion matrix görselleştirme
cm_log = confusion_matrix(y_test, y_pred_log)
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Lojistik Regresyon)")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()

from xgboost import XGBClassifier

# XGBoost modeli oluştur ve eğit
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Tahmin yap
y_pred_xgb = xgb_model.predict(X_test)

# Performans değerlendirme
print("\nSınıflandırma Raporu (XGBoost):")
print(classification_report(y_test, y_pred_xgb))

# Confusion matrix görselleştirme
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (XGBoost)")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()

from sklearn.ensemble import GradientBoostingClassifier

# Gradient Boosting modeli oluştur ve eğit
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Tahmin yap
y_pred_gb = gb_model.predict(X_test)

# Performans değerlendirme
print("\nSınıflandırma Raporu (Gradient Boosting):")
print(classification_report(y_test, y_pred_gb))

# Confusion matrix görselleştirme
cm_gb = confusion_matrix(y_test, y_pred_gb)
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Gradient Boosting)")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()