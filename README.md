📈 PyTorch LSTM Time-Series ForecastingBu proje, finansal zaman serisi verilerini (OHLCV: Open, High, Low, Close, Volume) kullanarak PyTorch mimarisi üzerinde sıfırdan geliştirilmiş çift katmanlı bir LSTM (Long Short-Term Memory) derin öğrenme modelini içermektedir.Model, finansal piyasalardaki karmaşık ve doğrusal olmayan ilişkileri öğrenmek amacıyla 30 günlük geçmiş zaman penceresini (seq_length = 30) analiz eder ve bir sonraki işlem gününün değerlerini tahmin eder.🏗️ Proje Mimarisi ve Çalışma MantığıPlaintext[Ham Veri (CSV)] ➔ [Tarih Sıralı Bölme] ➔ [Fit Transform (Scale)] ➔ [30 Günlük Pencereler] ➔ [LSTM Modeli] ➔ [Ters Ölçekleme] ➔ [Değerlendirme & Grafik]
1. Veri Ön İşleme Boru Hattı (Data Pipeline)Tarih Odaklı Indeksleme: Date sütunu zaman serisine uygun olarak indekslenir ve veri seti kronolojik olarak siralanir.Data Leakage (Bilgi Sızıntısı) Engeli:Veri seti kronolojik sırada %70 Eğitim (Train), %15 Doğrulama (Validation) ve %15 Test olarak bölünür.MinMaxScaler sadece Eğitim verisi üzerinde eğitilir (fit_transform), doğrulama ve test setleri bu parametrelerle dönüştürülür (transform).Zaman Penceresi (Sliding Window): create_sequences fonksiyonu ile 30 günlük geçmiş veriyi ($t_{-30}, \dots, t_{-1}$) girdi ($X$), 31. günün değerlerini ($t_{0}$) hedef ($y$) olarak paketler.2. Derin Öğrenme Modeli (LSTM Architecture)PyTorch nn.Module sınıfı türetilerek yazılan model mimarisi:Girdi Boyutu (input_size=5): Open, High, Low, Close, VolumeGizli Katman (hidden_size=64, num_layers=2): Çift katmanlı derin LSTM yapısı.Düzenlileştirme (dropout=0.2): Aşırı öğrenmeyi (Overfitting) engellemek amacıyla LSTM katmanları arasında %20 dropout uygulanır.Tam Bağlantılı Çıkış (Linear(64, 5)): Son zaman adımının gizli durumunu (Hidden State) alarak 5 boyutlu tahmin üretir.3. Eğitim & Erken Durdurma (Training Loop & Early Stopping)Kayıp Fonksiyonu: MSELoss (Mean Squared Error)Optimizasyon: Adam Optimizer (learning_rate = 0.001)Early Stopping:Model maksimum 100 epoch boyunca eğitilir.Her epoch sonunda validation loss takip edilir.Validation kaybı 15 epoch boyunca iyileşmezse eğitim otomatik kesilir ve en iyi ağırlıklar (best_lstm_model.pt) geri yüklenir.🛠️ Gereksinimler ve KurulumProjeyi çalıştırmadan önce gerekli kütüphaneleri yükleyin:Bashpip install torch numpy pandas matplotlib scikit-learn
🚀 Projeyi ÇalıştırmaProje dizinine finansal veri setinizi dataset.csv adıyla ekleyin. Veri setinizin aşağıdaki sütun yapısında olduğundan emin olun:Date, Open, High, Low, Close, VolumeKodlarınızı içerisine yapıştırdığınız main.py dosyasını çalıştırın:Bashpython main.py
📂 Proje Dizin YapısıPlaintext├── dataset.csv            # Girdi zaman serisi verisi
├── main.py                # Tüm pipeline'ı çalıştıran ana kod dosyası
├── best_lstm_model.pt     # Validation loss en düşük olduğu andaki model ağırlıkları
├── lstm_model_final.pt    # Eğitimin sonundaki model ağırlıkları
├── loss_curve.png         # Train vs Validation Loss öğrenme eğrisi
├── PredictionVsActual.png # Test setindeki Gerçek vs Tahmin Fiyatı karşılaştırması
└── README.md              # Proje dokümantasyonu
📉 Performans Değerlendirmesi & Baseline KıyaslamasıModelin başarısını değerlendirmek için Close (Kapanış) fiyatı üzerinden şu metrikler hesaplanır:MAE (Mean Absolute Error)RMSE (Root Mean Squared Error)MAPE (Mean Absolute Percentage Error)Naive Baseline (Rastgele Yürüyüş / Random Walk)Finansal zaman serilerinde modelin gerçek başarısını ölçmek için model, Naive Baseline yöntemiyle kıyaslanır. Naive Baseline, "Yarının kapanış fiyatı bugünün kapanış fiyatı ile aynıdır" diyen en basit yöntemdir.Aşağıda örnek bir konsol çıktısı yer almaktadır:Plaintext***LSTM modelinde Close Fiyatı için Test Metrikleri***
MAE:  3.5182
RMSE: 4.8821
MAPE: 2.85%

***Bir önceki günde Close Fiyatı için Naive Baseline***
MAE:  2.1104
RMSE: 3.1250
MAPE: 1.71%

==================================================
***Karşılaştırma***
Metrik  LSTM        Naive Baseline  Sonuç
--------------------------------------------------
MAE     3.5182      2.1104          Naive daha iyi
RMSE    4.8821      3.1250          Naive daha iyi
MAPE    2.85%       1.71%           Naive daha iyi

***Bir Sonraki Günün Tahmini***
Open:   121.2841
High:   123.1092
Low:    119.8450
Close:  121.9512
Volume: 3412890.5000
Not: Finansal piyasaların yüksek gürültülü (noisy) yapısı nedeniyle Naive Baseline sıklıkla güçlü bir rakiptir. Model hiper-parametrelerini (örneğin hidden_size, seq_length veya eklenen teknik indikatörler) değiştirerek LSTM performansını artırabilirsiniz.

