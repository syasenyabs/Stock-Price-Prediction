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

**Not:** Bu proje eğitim ve araştırma amaçlıdır. Üretilen tahminler yatırım tavsiyesi niteliği taşımaz.


---



# Amazon Stock Price Prediction (LSTM)

This project develops an LSTM (Long Short-Term Memory) neural network model to predict future price movements using Amazon's (AMZN) historical stock data. The model performs time series forecasting using OHLCV (Open, High, Low, Close, Volume) data.

## Table of Contents
- About the Project
- Dataset
- Technologies Used
- Methodology
- Setup
- Usage
- Model Architecture
- Results and Evaluation
- Project Structure
- Ideas for Improvement

## About the Project
Stock prices exhibit complex, non-linear patterns over time. This project builds a deep learning model (LSTM) that learns patterns in historical price movements to predict future prices, and compares this model's performance against a simple reference method (naive baseline).

## Dataset
- **Source:** Amazon (AMZN) historical stock price data
- **Columns:** Date, Open, High, Low, Close, Volume
- **Time range:** Data is processed in chronological order, sorted by date
- **Split ratio:**
  - Train: 70%
  - Validation: 15%
  - Test: 15%

**Note:** The dataset is split chronologically (not randomly), since preventing data leakage from the future into the past is critical in time series forecasting.

## Technologies Used
- Python 3
- PyTorch
- Pandas / NumPy
- Matplotlib
- scikit-learn (for metric calculations)

## Methodology

**Data Preprocessing**
- The date column is converted to datetime format and the data is sorted chronologically
- Data is split into train/validation/test sets in chronological order
- Min-Max normalization is calculated only on the training data and applied to all sets (to prevent data leakage)

**Sequence Creation**
- Time series data is converted into input-output pairs using the sliding window method
- Window length (seq_length): 30 days
- Each window is used to predict the next day's OHLCV values

**Model Training**
- Training is performed with a 2-layer LSTM network
- Loss function: MSE (Mean Squared Error)
- Optimization: Adam
- Early stopping is applied: if validation loss doesn't improve for a certain patience period, training stops and the best model weights are restored

**Evaluation**
- Predictions are generated on the test set
- Predictions are inverse-scaled from the normalized scale back to actual price scale
- MAE, RMSE, and MAPE metrics are calculated for the Close price
- Results are compared against a simple naive baseline based on "using the previous day's price as the prediction"

## Setup
```
pip install torch pandas numpy matplotlib scikit-learn
```
Place the `dataset.csv` file in the project folder (or working directory if using Colab). The file must contain at least these columns: Date, Open, High, Low, Close, Volume.

## Usage
```
python lstm_stock_prediction.py
```
If using Google Colab, you need to upload the file first:
```python
from google.colab import files
uploaded = files.upload()
```

## Model Architecture

| Parameter | Value |
|---|---|
| Input size | 5 (Open, High, Low, Close, Volume) |
| Hidden layer size | 64 |
| Number of LSTM layers | 2 |
| Output size | 5 |
| Dropout | 0.2 |
| Sequence length | 30 days |
| Optimizer | Adam (lr=0.001) |
| Loss function | MSE |
| Maximum epochs | 100 |
| Early stopping patience | 15 |

## Results and Evaluation
Model performance is evaluated on the Close price using three metrics:
- **MAE (Mean Absolute Error):** Average absolute deviation, in original price units
- **RMSE (Root Mean Squared Error):** A deviation measure more sensitive to large errors
- **MAPE (Mean Absolute Percentage Error):** Percentage deviation, for comparison across different scales

These metrics are also compared against the "using the previous day's price as the prediction" (naive baseline) method to test whether the model has actually learned a meaningful pattern.

*(You can add your own MAE / RMSE / MAPE values and chart images from your own runs here.)*

## Project Structure
```
.
├── dataset.csv                    # Amazon stock data
├── lstm_stock_prediction.py       # Main project code
├── best_lstm_model.pt             # Best model saved during training
├── lstm_model_final.pt            # Final (best) model weights
├── loss_curve.png                 # Training / validation loss chart
├── prediction_vs_actual.png       # Actual vs predicted price chart
└── README.md
```

## Ideas for Improvement
- Switching to using a DataLoader for mini-batch training
- Hyperparameter optimization with different `seq_length`, `hidden_size`, `num_layers` values
- Adding technical indicators (RSI, MACD, moving averages, etc.) as additional features
- Comparison with attention mechanisms or Transformer-based models
- Testing generalization capacity with multi-stock / multi-sector data

**Note:** This project is for educational and research purposes. The predictions produced do not constitute investment advice.

