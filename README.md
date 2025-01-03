# Bitirme Projesi: Müşteri Davranışları Analizi ve Tahmini

## Proje İçeriği
- **Veri Analizi**: Müşteri davranışlarının analiz edilmesi.
- **Model Eğitimi**: Gradient Boosting Classifier kullanılarak model eğitimi.
- **Çıktılar**: Özellik önem sırası ve sınıflandırma raporu.

## Klasör Yapısı
- **Veri/**: Projede kullanılan veri setleri.
- **Analiz/**: Analiz çıktıları ve raporlar.
- **Model/**: Model eğitim kodu ve eğitilmiş model.

## Çalıştırma Talimatları
1. Gereksinimleri yükleyin: `pip install -r requirements.txt`
2. Modeli eğitmek için: `python Model/model_egitimi.py`
3. Çıktıları `Analiz/` klasöründe bulabilirsiniz.

Dosyalama konusunda şimdilik bir sadeleşmeye gidilememiştir. Tüm çalışma bir çok dosya açılarak gerçekleştirilmiştir. Farklı denemeler ve geliştirmeler yapılmıştır.
Özellikle Total Spending ve Average Spending verilerinin modeli overfitting yapması yani fazla öğretmesi dolayısıyla dengesizleştiği görülmüş ancak bu verilerden bir süre vazgeçilmek
istenmeyip bu konuda denemeler yapılmıştır. Ancak parametre ayarlamaları yapılmasına rağmen ve daha öncesinde de lojistik regresyon gibi nispeten daha az duyarlı bir modelde bile overfit olduğu gözlemlendiği
için bu veri sütunlarından vazgeçilmiş ve son modele erişilmiştir. Son modelde Last Purchase ve Recent Spending sütunlarının veriye doğrudan etkisi olmamıştır. Bunların çıkarılıp veya Spending Ratio gibi 
bu verilerden üretilmiş başka bir sütunun oluşturulması konusunda denemeler yapılamamıştır. Elimizdeki modelin başarısı göz önüne alınarak model gelişimi durdurulmuştur.

Klasörler tüm çalışmalarımı içerse de ana dosyalar

Veri/ veri_analizi.py
Veri/ veri_temizleme.py
Model/ model_egitimi.py olarak şekillenmiştir.

Dosya sadeleştirmesi ve dosya karmaşasını dana açıklanır hale getirme adımları tam olarak tamamlanamamıştır. Dosyalar en kısa süre içinde daha anlaşılır ve adımları anlaşılır şekilde düzenlenecektir.

Model Performans Sonuçları kontrollerimde

Performans Metrikleri:
Accuracy: %94
Precision (Sınıf 1): %78
Recall (Sınıf 1): %98
F1-Score (Sınıf 1): %87
Özellik Önemi:
En önemli özellikler:
Quantity (%66.3)
Price (%33.7)

şeklinde gerçekleşmiştir.
