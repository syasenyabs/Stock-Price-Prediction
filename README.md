# 📈 Hisse Senedi Fiyat Tahmini — LSTM Modeli (PyTorch)

Bu proje, geçmiş hisse senedi verilerini (**Open, High, Low, Close, Volume** — kısaca **OHLCV**) kullanarak bir sonraki günün fiyat ve hacim değerlerini tahmin etmeyi amaçlayan çok değişkenli (multivariate) bir zaman serisi tahmin modelidir. Model, **PyTorch** ile sıfırdan yazılmış bir **LSTM (Long Short-Term Memory)** ağı kullanır ve **Google Colab** üzerinde çalışacak şekilde hazırlanmıştır (GPU desteği ile).

---

## 📑 İçindekiler

- [Amaç](#-amaç)
- [Dosya Yapısı](#-dosya-yapısı)
- [Gereksinimler ve Kurulum](#️-gereksinimler-ve-kurulum)
- [Değerlendirme Metrikleri](#-değerlendirme-metrikleri)
- [Hiperparametreler](#-hiperparametreler)
- [Çalıştırma Talimatları](#️-çalıştırma-talimatları)
- [Sorumluluk Reddi](#️-sorumluluk-reddi)

---

## 🎯 Amaç

Zaman serisi tahmini, özellikle finansal piyasalarda oldukça zorlu bir problemdir; çünkü fiyat hareketleri gürültülü, kısmen rastgele ve birçok dışsal faktöre (haberler, makroekonomik veriler, piyasa psikolojisi vb.) bağlıdır. Bu proje, klasik bir **çok-değişkenli, tek-adım-öncesi (single-step-ahead) tahmin** yaklaşımı sergiler:

- **Girdi:** Son 30 günün Open, High, Low, Close, Volume değerleri (5 özellik × 30 zaman adımı)
- **Çıktı:** Bir sonraki günün Open, High, Low, Close, Volume değerleri (5 özellik)

Model, tüm OHLCV vektörünü aynı anda tahmin eder (çoklu çıktı regresyonu). Bu, modelin yalnızca kapanış fiyatına değil, günün genel yapısına (açılış, en yüksek, en düşük, hacim) dair bir bütün öğrenmesini sağlamayı amaçlar. Süreç şu aşamalardan oluşur:

1. Verinin okunması, kronolojik sıralanması ve eğitim/validasyon/test olarak bölünmesi.
2. Yalnızca eğitim verisinden hesaplanan min-max değerleriyle normalizasyon (veri sızıntısını önlemek için).
3. 30 günlük kayan pencere (sliding window) ile girdi/çıktı sekanslarının oluşturulması.
4. 2 katmanlı bir LSTM ağının eğitilmesi ve validasyon kaybına göre early stopping uygulanması.
5. Test setinde performansın MAE, RMSE ve MAPE ile ölçülmesi ve naive baseline ile karşılaştırılması.
6. Eğitim/validasyon kayıp grafiği ile gerçek/tahmin karşılaştırma grafiğinin çizilmesi.
7. Veri setindeki son 30 güne dayanarak bir sonraki günün OHLCV değerlerinin tahmin edilmesi.

---

## 📁 Dosya Yapısı

```
.
├── stockprice.ipynb          # Ana Colab/Jupyter notebook (tüm kod burada)
├── dataset.csv                # Girdi verisi (kullanıcı tarafından sağlanmalı)
├── best_lstm_model.pt         # Eğitim sırasında oluşur — en iyi validasyon skoruna sahip ağırlıklar
├── lstm_model_final.pt        # Eğitim sonunda oluşur — best_lstm_model.pt ile aynı ağırlıklar
├── loss_curve.png             # Eğitim sonunda oluşur — train/val loss grafiği
├── PredictionVsActual.png     # Eğitim sonunda oluşur — test setinde gerçek vs. tahmin grafiği
└── README.md                  
```

> 📝 `dataset.csv` dosyasının aynı klasörde bulunması ve `Date, Open, High, Low, Close, Volume` sütunlarını içermesi gerekir. `.pt` ve `.png` dosyaları repoya dahil değildir; notebook her çalıştırıldığında yeniden üretilir.

---

## ⚙️ Gereksinimler ve Kurulum

### Python Sürümü
Python 3.8 veya üzeri önerilir (Google Colab'ın varsayılan ortamı uyumludur).

### Gerekli Kütüphaneler

| Kütüphane | Kullanım Amacı |
|---|---|
| `torch` | Model tanımı, eğitim, GPU/CPU hesaplama |
| `numpy` | Sayısal işlemler |
| `pandas` | CSV okuma, veri manipülasyonu, tarih işlemleri |
| `matplotlib` | Kayıp eğrisi ve tahmin grafikleri |
| `scikit-learn` | `mean_squared_error`, `mean_absolute_error` metrikleri |

### Kurulum

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

Google Colab kullanıyorsanız bu kütüphaneler genellikle ortamda **hazır** gelir; ekstra kurulum gerekmez.

### GPU Kullanımı

Kod, GPU mevcutsa otomatik olarak onu kullanır:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Google Colab üzerinde GPU'yu etkinleştirmek için: **Çalışma Zamanı → Çalışma zamanı türünü değiştir → Donanım Hızlandırıcı → GPU**.

---

## 📐 Değerlendirme Metrikleri

Model, test seti üzerinde yalnızca `Close` fiyatı için üç standart regresyon metriğiyle değerlendirilir ve sonuçlar "bir önceki günün fiyatını tahmin olarak kullanma" (naive baseline) yöntemiyle karşılaştırılır:

- **MAE (Mean Absolute Error):** Tahmin hatalarının mutlak değerlerinin ortalamasıdır. Birim, orijinal fiyat birimiyle aynıdır ve aykırı değerlere MSE/RMSE kadar duyarlı değildir.

  `MAE = (1/n) * Σ|y_i - ŷ_i|`

- **RMSE (Root Mean Squared Error):** Hataların karelerinin ortalamasının kareköküdür; büyük hataları daha ağır cezalandırır.

  `RMSE = sqrt((1/n) * Σ(y_i - ŷ_i)²)`

- **MAPE (Mean Absolute Percentage Error):** Hataları yüzdesel olarak ifade eder, farklı ölçekteki hisseler arasında karşılaştırma yapmayı kolaylaştırır.

  `MAPE = (100/n) * Σ|(y_i - ŷ_i) / y_i|`

Ayrıca eğitim sırasında model seçimi ve early stopping kararı için **MSELoss** (train/validation loss) izlenir.

---

## 🔢 Hiperparametreler

| Parametre | Değer | Açıklama |
|---|---|---|
| `seq_length` | 30 | Girdi olarak kullanılan geçmiş gün sayısı |
| `input_size` | 5 | Özellik sayısı (Open, High, Low, Close, Volume) |
| `hidden_size` | 64 | LSTM gizli katman boyutu |
| `num_layers` | 2 | LSTM katman sayısı |
| `output_size` | 5 | Tahmin edilen özellik sayısı |
| `dropout` | 0.2 | Katmanlar arası dropout oranı |
| `num_epochs` | 100 (maksimum) | Eğitim epoch sayısı |
| `patience` | 15 | Early stopping sabır değeri |
| `learning_rate` | 0.001 | Adam optimizer öğrenme oranı |
| `loss function` | MSELoss | Ortalama karesel hata |
| `train/val/test oranı` | %70 / %15 / %15 | Kronolojik bölünme |

Bu değerleri notebook içinde ilgili satırları düzenleyerek kolayca değiştirebilirsiniz.

---

## ▶️ Çalıştırma Talimatları

### Google Colab'da Çalıştırma

1. `stockprice.ipynb` dosyasını [Google Colab](https://colab.research.google.com/) üzerinde açın.
2. `dataset.csv` dosyasını Colab ortamına yükleyin (sol taraftaki dosya paneli üzerinden veya `files.upload()` ile).
3. Çalışma zamanı türünü GPU olarak ayarlayın (isteğe bağlı ama önerilir).
4. Hücreleri sırasıyla çalıştırın (`Runtime → Run all` veya tek tek `Shift+Enter`).

### Yerel Ortamda Çalıştırma

```bash
# Sanal ortam oluşturma (isteğe bağlı ama önerilir)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Gerekli kütüphaneleri kurma
pip install torch numpy pandas matplotlib scikit-learn jupyter

# dataset.csv dosyasını proje klasörüne koyun, ardından:
jupyter notebook stockprice.ipynb
```

Notebook çalıştırıldıktan sonra aşağıdaki dosyalar otomatik olarak oluşturulacaktır:

- `best_lstm_model.pt`
- `lstm_model_final.pt`
- `loss_curve.png`
- `PredictionVsActual.png`

### Kaydedilmiş Modeli Tekrar Yükleme

```python
model = LSTMModel(input_size=5, hidden_size=64, num_layers=2, output_size=5)
model.load_state_dict(torch.load("lstm_model_final.pt", map_location=device))
model.eval()
```
## ⚠️ Sorumluluk Reddi

Bu proje **eğitim ve araştırma amaçlıdır**. Burada sunulan model, tahminler ve sonuçlar **finansal tavsiye niteliği taşımaz**. Gerçek para ile alım-satım kararları almadan önce mutlaka bir finansal danışmana başvurun. Geçmiş performans, gelecekteki sonuçların garantisi değildir. Bu kodu kullanarak yapacağınız herhangi bir finansal işlemden doğacak kâr veya zarardan proje sahibi/geliştirici sorumlu tutulamaz.



# 📈 Stock Price Prediction — LSTM Model (PyTorch)

This project is a multivariate time-series forecasting model that predicts the next day's price and volume values using historical stock data (**Open, High, Low, Close, Volume** — **OHLCV**). The model is a **LSTM (Long Short-Term Memory)** network built from scratch with **PyTorch**, and is designed to run on **Google Colab** (with GPU support).

---

## 📑 Table of Contents

- [Purpose](#-purpose)
- [File Structure](#-file-structure)
- [Requirements and Setup](#️-requirements-and-setup)
- [Evaluation Metrics](#-evaluation-metrics)
- [Hyperparameters](#-hyperparameters)
- [How to Run](#️-how-to-run)
- [Disclaimer](#️-disclaimer)

---

## 🎯 Purpose

Time-series forecasting is a particularly challenging problem in financial markets, since price movements are noisy, partly random, and depend on many external factors (news, macroeconomic data, market psychology, etc.). This project demonstrates a classic **multivariate, single-step-ahead forecasting** approach:

- **Input:** OHLCV values for the past 30 days (5 features × 30 time steps)
- **Output:** OHLCV values for the next day (5 features)

The model predicts the entire OHLCV vector at once (multi-output regression). This is meant to help the model learn about the overall structure of a trading day (open, high, low, volume), not just the closing price. The pipeline consists of the following stages:

1. Reading the data, sorting it chronologically, and splitting it into train/validation/test sets.
2. Normalizing the data using min-max values computed **only from the training set** (to prevent data leakage).
3. Building input/output sequences with a 30-day sliding window.
4. Training a 2-layer LSTM network with early stopping based on validation loss.
5. Evaluating test-set performance with MAE, RMSE, and MAPE, and comparing it against a naive baseline.
6. Plotting the training/validation loss curve and the actual-vs-predicted comparison chart.
7. Predicting the next day's OHLCV values based on the last 30 days in the dataset.

---

## 📁 File Structure

```
.
├── stockprice.ipynb          # Main Colab/Jupyter notebook (all code lives here)
├── dataset.csv                # Input data (must be provided by the user)
├── best_lstm_model.pt         # Generated during training — weights with the best validation score
├── lstm_model_final.pt        # Generated after training — same weights as best_lstm_model.pt
├── loss_curve.png             # Generated after training — train/val loss chart
├── PredictionVsActual.png     # Generated after training — actual vs. predicted chart on the test set
└── README.md                  
```

> 📝 `dataset.csv` must be in the same folder and must contain the `Date, Open, High, Low, Close, Volume` columns. The `.pt` and `.png` files are not included in the repo; they are regenerated every time the notebook runs.

---

## ⚙️ Requirements and Setup

### Python Version
Python 3.8 or higher is recommended (compatible with Google Colab's default environment).

### Required Libraries

| Library | Purpose |
|---|---|
| `torch` | Model definition, training, GPU/CPU computation |
| `numpy` | Numerical operations |
| `pandas` | CSV reading, data manipulation, date handling |
| `matplotlib` | Loss curve and prediction plots |
| `scikit-learn` | `mean_squared_error`, `mean_absolute_error` metrics |

### Installation

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

If you're using Google Colab, these libraries are usually **pre-installed**; no extra setup is needed.

### GPU Usage

The code automatically uses a GPU if one is available:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

To enable GPU on Google Colab: **Runtime → Change runtime type → Hardware accelerator → GPU**.

---

## 📐 Evaluation Metrics

The model is evaluated on the test set for the `Close` price only, using three standard regression metrics, and the results are compared against a "use yesterday's price as tomorrow's prediction" (naive baseline) approach:

- **MAE (Mean Absolute Error):** The average of the absolute values of the prediction errors. It's in the same unit as the original price and is less sensitive to outliers than MSE/RMSE.

  `MAE = (1/n) * Σ|y_i - ŷ_i|`

- **RMSE (Root Mean Squared Error):** The square root of the average of squared errors; it penalizes larger errors more heavily.

  `RMSE = sqrt((1/n) * Σ(y_i - ŷ_i)²)`

- **MAPE (Mean Absolute Percentage Error):** Expresses errors as a percentage, making it easier to compare across stocks of different scales.

  `MAPE = (100/n) * Σ|(y_i - ŷ_i) / y_i|`

**MSELoss** (train/validation loss) is also tracked during training for model selection and the early stopping decision.

---

## 🔢 Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `seq_length` | 30 | Number of past days used as input |
| `input_size` | 5 | Number of features (Open, High, Low, Close, Volume) |
| `hidden_size` | 64 | LSTM hidden layer size |
| `num_layers` | 2 | Number of LSTM layers |
| `output_size` | 5 | Number of predicted features |
| `dropout` | 0.2 | Dropout rate between layers |
| `num_epochs` | 100 (maximum) | Number of training epochs |
| `patience` | 15 | Early stopping patience |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `loss function` | MSELoss | Mean squared error |
| `train/val/test ratio` | 70% / 15% / 15% | Chronological split |

You can easily change these values by editing the corresponding lines in the notebook.

---

## ▶️ How to Run

### Running on Google Colab

1. Open `stockprice.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Upload `dataset.csv` to the Colab environment (via the file panel on the left or `files.upload()`).
3. Set the runtime type to GPU (optional but recommended).
4. Run the cells in order (`Runtime → Run all` or one by one with `Shift+Enter`).

### Running Locally

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install required libraries
pip install torch numpy pandas matplotlib scikit-learn jupyter

# Place dataset.csv in the project folder, then:
jupyter notebook stockprice.ipynb
```

After running the notebook, the following files will be generated automatically:

- `best_lstm_model.pt`
- `lstm_model_final.pt`
- `loss_curve.png`
- `PredictionVsActual.png`

### Reloading the Saved Model

```python
model = LSTMModel(input_size=5, hidden_size=64, num_layers=2, output_size=5)
model.load_state_dict(torch.load("lstm_model_final.pt", map_location=device))
model.eval()
```

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. The model, predictions, and results presented here **do not constitute financial advice**. Always consult a financial advisor before making any real trading decisions. Past performance is not a guarantee of future results. The project owner/developer is not responsible for any profit or loss resulting from financial transactions made using this code.

---
