import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükle
ana_veri_dosyasi = "Analiz/musteri_ozeti.csv"
data = pd.read_csv(ana_veri_dosyasi)

# Sadece sayısal sütunları seç
numeric_columns = ['Total Spending', 'Frequency', 'Tenure']
data_numeric = data[numeric_columns]

# Eksik değer kontrolü
if data_numeric.isnull().any().any():
    print("Eksik değerler var, lütfen kontrol edin.")
    print(data_numeric.isnull().sum())
    data_numeric = data_numeric.dropna()

# Özellikleri ölçeklendirme
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# En uygun küme sayısını belirlemek için Elbow Method
inertia = []
range_n_clusters = range(1, 11)

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

# Dirsek grafiği çizimi
plt.figure(figsize=(8, 5))
plt.plot(range_n_clusters, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Küme Sayısı')
plt.ylabel('Inertia')
plt.show()

# Optimal küme sayısını seç (örneğin 4)
optimal_clusters = 4

# KMeans modeli ile segmentasyon
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Segment'] = kmeans.fit_predict(data_scaled)

# Segmentleri analiz et
print("\nSegmentlere Göre Özet:")
print(data.groupby('Segment')[numeric_columns].mean())

# Segmentlerin dağılımını görselleştirme
plt.figure(figsize=(8, 6))
sns.countplot(x='Segment', data=data, palette='viridis')
plt.title('Segment Dağılımı')
plt.xlabel('Segment')
plt.ylabel('Müşteri Sayısı')
plt.show()

# Segment bilgilerini içeren dosyayı kaydet
data.to_csv("Analiz/musteri_segmentasyonu.csv", index=False)
print("\nSegment bilgileri kaydedildi: Analiz/musteri_segmentasyonu.csv")