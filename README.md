# Amazon Hisse Senedi Fiyat Tahmini (LSTM)

Bu proje, Amazon (AMZN) hisse senedi geçmiş verilerini kullanarak gelecekteki fiyat hareketlerini tahmin etmek amacıyla bir **LSTM (Long Short-Term Memory)** sinir ağı modeli geliştirir. Model; açılış, en yüksek, en düşük, kapanış fiyatları ve işlem hacmi (OHLCV) verilerini kullanarak zaman serisi tahmini yapar.

## İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Veri Seti](#veri-seti)
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
- [Metodoloji](#metodoloji)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Model Mimarisi](#model-mimarisi)
- [Sonuçlar ve Değerlendirme](#sonuçlar-ve-değerlendirme)
- [Proje Yapısı](#proje-yapısı)
- [Geliştirme Fikirleri](#geliştirme-fikirleri)

## Proje Hakkında

Hisse senedi fiyatları zaman içinde karmaşık, doğrusal olmayan örüntüler sergiler. Bu proje, geçmiş fiyat hareketlerindeki örüntüleri öğrenerek gelecekteki fiyatları tahmin edebilen bir derin öğrenme modeli (LSTM) kurar ve bu modelin performansını basit bir referans yöntemle (naive baseline) karşılaştırır.

## Veri Seti

- **Kaynak:** Amazon (AMZN) hisse senedi geçmiş fiyat verisi
- **Sütunlar:** `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- **Zaman aralığı:** Veri, tarihe göre sıralanarak kronolojik sırayla işlenir
- **Bölünme oranı:**
  - Eğitim (Train): %70
  - Doğrulama (Validation): %15
  - Test: %15

> Not: Veri seti zaman sırasına göre bölünmüştür (rastgele değil), çünkü zaman serisi tahmininde gelecekten geçmişe bilgi sızıntısını (data leakage) önlemek kritik önem taşır.

## Kullanılan Teknolojiler

- Python 3
- PyTorch
- Pandas / NumPy
- Matplotlib
- scikit-learn (metrik hesaplamaları için)

## Metodoloji

1. **Veri Ön İşleme**
   - Tarih sütunu datetime formatına çevrilir ve veri kronolojik olarak sıralanır
   - Veri, train/validation/test olarak zaman sırasına göre bölünür
   - Min-Max normalizasyonu **sadece eğitim verisi** üzerinden hesaplanır ve tüm setlere uygulanır (veri sızıntısını önlemek için)

2. **Sequence Oluşturma**
   - Zaman serisi verisi, kayan pencere (sliding window) yöntemiyle giriş-çıkış çiftlerine dönüştürülür
   - Pencere uzunluğu (`seq_length`): 30 gün
   - Her pencere, bir sonraki günün OHLCV değerlerini tahmin etmek için kullanılır

3. **Model Eğitimi**
   - 2 katmanlı LSTM ağı ile eğitim yapılır
   - Kayıp fonksiyonu: MSE (Mean Squared Error)
   - Optimizasyon: Adam
   - **Early stopping** uygulanır: doğrulama kaybı belirli bir sabır (patience) süresi boyunca iyileşmezse eğitim durdurulur ve en iyi model ağırlıkları geri yüklenir

4. **Değerlendirme**
   - Test seti üzerinde tahminler üretilir
   - Tahminler, normalize edilmiş ölçekten gerçek fiyat ölçeğine geri çevrilir (inverse scaling)
   - Kapanış (Close) fiyatı için MAE, RMSE ve MAPE metrikleri hesaplanır
   - Sonuçlar, "bir önceki günün fiyatını tahmin olarak kullanma" mantığına dayanan basit bir **naive baseline** ile karşılaştırılır

## Kurulum

```bash
pip install torch pandas numpy matplotlib scikit-learn
```

`dataset.csv` dosyasını proje klasörüne (veya Colab kullanıyorsan çalışma dizinine) yerleştirin. Dosya en az şu sütunları içermelidir: `Date, Open, High, Low, Close, Volume`.

## Kullanım

```bash
python lstm_stock_prediction.py
```

Google Colab kullanıyorsanız, dosyayı önce yüklemeniz gerekir:

```python
from google.colab import files
uploaded = files.upload()
```

## Model Mimarisi

| Parametre | Değer |
|---|---|
| Giriş boyutu (input_size) | 5 (Open, High, Low, Close, Volume) |
| Gizli katman boyutu (hidden_size) | 64 |
| LSTM katman sayısı (num_layers) | 2 |
| Çıkış boyutu (output_size) | 5 |
| Dropout | 0.2 |
| Sequence uzunluğu | 30 gün |
| Optimizer | Adam (lr=0.001) |
| Kayıp fonksiyonu | MSE |
| Maksimum epoch | 100 |
| Early stopping patience | 15 |

## Sonuçlar ve Değerlendirme

Model performansı, Close (kapanış) fiyatı üzerinden üç metrikle değerlendirilir:

- **MAE (Mean Absolute Error):** Ortalama mutlak sapma, orijinal fiyat biriminde
- **RMSE (Root Mean Squared Error):** Büyük hatalara daha duyarlı sapma ölçüsü
- **MAPE (Mean Absolute Percentage Error):** Yüzdesel sapma, farklı ölçekler arası karşılaştırma için

Bu metrikler ayrıca "bir önceki günün fiyatını tahmin olarak kullanma" (naive baseline) yöntemiyle karşılaştırılarak, modelin gerçekten anlamlı bir örüntü öğrenip öğrenmediği test edilir.

*(Buraya kendi çalıştırdığın sonuçlardan elde ettiğin MAE / RMSE / MAPE değerlerini ve grafik görsellerini ekleyebilirsin.)*

## Proje Yapısı

```
.
├── dataset.csv                    # Amazon hisse senedi verisi
├── lstm_stock_prediction.py       # Ana proje kodu
├── best_lstm_model.pt             # Eğitim sırasında kaydedilen en iyi model
├── lstm_model_final.pt            # Final (en iyi) model ağırlıkları
├── loss_curve.png                 # Eğitim / doğrulama kaybı grafiği
├── prediction_vs_actual.png       # Gerçek vs tahmin edilen fiyat grafiği
└── README.md
```

## Geliştirme Fikirleri

- Mini-batch eğitimi için `DataLoader` kullanımına geçiş
- Farklı `seq_length`, `hidden_size`, `num_layers` değerleriyle hiperparametre optimizasyonu
- Teknik göstergelerin (RSI, MACD, hareketli ortalamalar vb.) ek özellik olarak eklenmesi
- Attention mekanizması veya Transformer tabanlı modellerle karşılaştırma
- Çoklu hisse senedi / sektör verisiyle genelleme kapasitesinin test edilmesi

---

**Not:** Bu proje eğitim ve araştırma amaçlıdır. Üretilen tahminler yatırım tavsiyesi niteliği taşımaz.
