# BERT FineTune Dökümanı

## Data Preprocessor

- Modele vermeden önce ham text ve ham labelları alarak datanın eğitime hazırlanması için her veri adımını içerir.

Fonksiyonların görevleri:

- `prepare`:
    - İlk olarak `LabelEncode`r kullanılarak döküman tipleri encode edilir, `label_mapping` sözlüğünde bu bilgiler saklanır. Bu mapping inference esnasında tahminleri decode etmek için checkpointe kaydedilir.
    - Veri train, validation ve test kümelerine bölünür.
    - Train sette sample sayısı az olan labelların datalarına oversampling uygulanır.
- `compute_class_weights`:
    - `CrossEntropyLoss`’a geçilmek üzere azınlık sınıfa daha yüksek önem verilmesi için sınıf ağırlıkları burada hesaplanır.
- `create_weighted_sampler`:
    - Her bir batche koyulacak belge tipi sayısını dengeleyerek batch oluşturur. Yani dengesiz veriyi batch düzeyinde çözümler (dengeler).
    - `replacement=True` özeliği ile azınlık sınıfın örneği biterde batchlerdeki dengeyi bozmamak için aynı örneğin birden fazla kez seçilebilmesini sağlar.
- `_oversample`:
    - `minority_threshold` sayısından daha az örneğe sahip belge tipleri duplicate edilerek oversampling yapılır.
    - Kopyalama sayısı ise `oversample_factor` değişkeni ile seçilir.

## Windowed Dataset Creator

- Data preprocessor’un işlediği datayı input olarak alır.
- Bu datayı birbirleriyle overlap eden windowlara böler, sonrasında her bir windowu ayrı bir örnek gibi DataLoader’a sunar.
- Değerlendirme aşamasında aynı belgeye ait windowlar `doc_id` ile tekrar bir araya getirilip pooling yapılıp tahmin üretilir.

Fonksiyonların görevleri:

- `__init__`:
    - Parametreleri saklar ve `_build_windows` fonksiyonunu çağırır.
- `_build_windows`:
    - Datayı tokenize edip windowlara bölme işlemleri gerçekleştirilir.
    - Burada `[CLS]` ve `[SEP]` tokenleri eklenmez.
- `_add_window`:
    - Token listesinin başına `[CLS]` ve sonuna `[SEP]` tokenlerini ekler ve bu windowu kaydeder.
- `__get_item__`:
    - Belirli indexe göre bir windowu çeker, max_length’e göre pad eder ve ilgili windowu tensor olarak döndürür.

## Base Text Classifier

- Proje içerisinde eğitilecek tüm modeller için bir üst sınıf görevindedir.
- Sisteme yeni bir model ekleneceğinde bu sınıftan inherit edilip `extract_features` kodunun yazılması yeterlidir.

Fonksiyonların görevleri:

- `__init__`:
    - Encoderdan gelen embeddingi alır, dropout uygular ve `num_classes` boyutunda logit üretir.
    - Bilinmeyen dosyaların tespiti için kullanılacak bazı bufferlar burada kaydedilir.
- `extract_features`:
    - Her modelin encoder’ı farklıdır (biri RoBERTa kullanır, biri BERT) ama hepsinden beklenen çıktı aynıdır. `[batch_size, hidden_size]` boyutundaki bir embedding tensörüdür.
    - Bu ortak çıktı formatı dolayısıyla bu fonksiyon abstract olarak tasarlanmıştır.
- `forward`:
    - Tüm modellerde akış aynıdır: extract_features → dropout → classifier → logits.
    - return_embeddings kullanıldığında artık fonksiyon ham embeddingi de döndürür.
    - Dönen bu embedding 2 iş için kullanılır:
        - Training esnasında `update_centroids`’i çağırmak için,
        - Inference sırasında unknown detection için.
- `update_centroids`:
    - Her training batch’inde çağırılır, her class için o sınıfa ait embeddinglerin hareketli ortalamasını tutar. (Yani sınıfın embedding uzayındaki merkezini öğrenir)
    - Bu incremental hesaplanan formül sayesinde centroidler UnknownDetector tarafından “bu sample bilinen bir sınıfa mı benziyor?” sorusunu cevaplamak için kullanılır.

## XLM RoBERTa Text Classifier

- XLM RoBERTa modelinin encoderının nasıl çalışacağı ile ilgili kod burada bulunur.
- Çoğu fonksiyonu üst sınıf olan `BaseTextClassifier`’dan hazır olarak gelir.

Fonksiyonların görevleri:

- `__init__`:
    - Kullanılacak modelin config verileri indirilerek üst sınıfa gönderilirler.
    - Daha sonra da tüm model indirilir.
- `extract_features`:
    - Encoder’ı çalıştırır.
    - Daha sonra `[batch_size, hidden_size]` boyutunda `[CLS]` embeddingini döndürür.

## Unknown Detector

- Amacı, modelin hiç görmediği türden bir belge geldiğinde bunu tespit etmektir.

Fonksiyonların görevleri:

- `calibrate`:
    - Validasyon aşamasında iken modelin doğru tahmin ettiği sample’ların sınıf embeddinginin centroidine olan uzaklığı alınır ve listeye eklenir.
    - Tahmni edilen fatura, ve gerçekten de fatura ise `[CLS]` embedding’i fatura centroid’inden uzaklığı `distances`’e eklenir. Aynı durum atr olarak tahmin edilip, gerçekten de atr olduğu bulunan belge için de geçerli. Bu uzaklıkların hepsi ortak `distances` listesine eklenir.
    - Daha sonra `distance_threshold_percentile` ile belirlenen bir yüzdelikteki değer hesaplanır. Örneğin “Elde edilen mesafelerin %95’i 4.8’in altında”. Artık `distance_threshold` değeri belirlenmiş olur.
    - `predict` fonksiyonu kullanılırken yeni bir belge geldiğinde embedding’i tahmin edilen sınıfın centroid’ine 4.8’den uzaksa bu “belge o sınıfın bilinen örneklerine benzemiyor” çıkarımı yapılır ve belke unknown olarak etiketlenir.
- `predict`:
    - Bu fonksiyon yeni gelen sample’ın bilinip bilinmediğini tespit etmek için 2 strateji kullanır.
        - Confidence threshold: softmax sonrası en yüksek olasılık `confidence_threshold`’dan küçükse bu bir işarettir.
        - Centroid distance: Embedding’in tahmin edilen sınıfın merkezinden `distance_threshold`’dan uzaksa belge o sınıfa ait olmayabilir.
    - Bu iki stratejinin kombinasyonu ise model configinde belirlenen `unknown_logic` ile belirlenir:
        - `OR`: İki stratejinin birinden unknown sinyali gelirse belge unknown olarak işaretlenir.
        - `AND`: Her iki stratejiden de unknown sinyali gelirse belge unknown olarak işaretlenir. OR’a göre daha seçicidir.

## Trainer

- Eğitim pipeline’ının orkestratörüdür.
- Model ne olursa olsun training loop aynı kalır.

Fonksiyonların görevleri:

- `__init__`:
    - Imbalance bir veri ile ilgileniliyorsa kullanılacak loss `class_weight`’lerle oluşturulur.
    - Eğer model configinde `unknown_detector` etkinleştirilmişse validation aşamasında kullanılmak üzere değişken oluşturulur.
- `train`:
    - Tüm epoch döngülerini ve unknown detector kalibrasyon süreçlerini çalıştırır.
    - İzlenen metrik iyileşirse checkpointi kaydeder.
    - `train_loader` değişkeni bu fonksiyon çağırılmadan önce oluşturulmuş olmalıdır.
    - Tüm epoch döngüleri bitince unknown detector kalibrasyonu validation seti üzerinde yapılır.
- `evaluate`:
    - Bu fonksiyon hem validation hem test için kullanılmaktadır. Test ederken label_mapping parametresi verilirse classification report yazdırır. Validation’da bu parametre verilmez.
    - Modelin sliding window aggregation yöntemiyle değerlendirilmesi burada yapılır.
    - Aynı belgeye ait windowlar tek bir tahmine birleştirilir. Bu yapılırken model configindeki ayar kullanılarak `_aggegate_windows` fonksiyonu çağırılarak pooling yapılır.
- `_setup_optimizer`:
    - Weight decay ayrımı buradadır.
    - Bias ve LayerNorm parametrelerine weight decay uygulanmaz.
    - Optimizer ve linear warmup scheduler burada oluşturulur.
- `_train_epoch`:
    - Gradient Accumulation ile tam bir training epoch’unu çalıştırır.
    - Unknown detection kalibrasyonu için her bir batch’de class centroid’leri güncelenir.
    - Loss, model configindeki `gradient_accumulation_steps`'e bölünerek backward yapılır.
- `plot_history`:
    - Eğtim ile ilgili grafiklerin gösterilmesini ve kaydedilmesini (checkpoint’in içine) sağlayan fonksiyondur.
- `load_best`:
    - Verilen checkpoint model configindeki modeli yükler.
    - Eğitilmiş bir modelin checkpoints klasöründeki `config.json` dosyası ile çalışır.
- `_aggregate_windows`:
    - Bir belgeye ait oluşturulan window logitlerinin pool edilip tek bir çıktı elde edilmesi burada gerçekleştirilir.
    - Model configindeki `window_pooling` keyi ile bu ayarlanabilir.
    - max, mean veya first değerlerini alabilir.
- `_save_checkpoint`:
    - Model ağırlıklarının ve eğitim ile alakalı kullanışlı verilerin kaydedilmesini sağlar.
    - İlgili eğitimdeki en iyi model bu fonksiyon ile kaydedilir.

## train.py

- Tüm eğitim pipeline’ı burada kurulur, verinin hazırlanmasından modelin eğitilip test edilmesine kadar.

Fonksiyonların görevleri:

- `resolve_checkpoint_dir`:
    - Aynı model daha önce eğitildiyse versiyon numarası artırarak checkpoint klasörünü oluşturmaya yarar.
- `load_model_class`:
    - Projenin config dosyasındaki `MODEL_REGISTRY`’e kayıt edilmiş modellerden kullanılmak istenen model dinamik bir şekilde model import edilir.
- `main`:
    1. Modele ait json config dosyası yüklenir ve içerisine `resolve_checkpoint_dir` fonksiyonundan gelen versiyon bilgisi eklenmiş bir şekilde `checkpoint_dir` değişkeni de eklenir.
    2. `DataPreprocessor` model configi ile oluşturulur. Train, validation ve test setleri elde edilir.
    Elde edilen label mapping modelin checkpoint klasörüne kaydedilir.
    3. Model configindeki `model_name` bilgisi kullanılarak tokenizer oluşturulur.
    4. `WindowedDatasetCreator` fonksiyonu ile train, validation ve test kümeleri model configindeki ayarlara göre windowed hale getirilirler.
    5. Eğer `use_class_weights` etkinleştirilmişse DataPreprocessor kullanılarak sınıf ağırlıkları hesaplanır, sonrasında `create_weighted_sampler` kullanılarak train kümesi için balanced batch yapısı oluşturulmuş olur.
    6. DataLoader kullanılarak kümeler Pytorch modeline yüklemeye uygun hale getirilirler. Eğer sınıf ağırlıkları kullanılıyorsa `train_loader` shuffle edilmez, validation ve test edilir. Sınıf ağırlıkları kullanılmayacaksa train de shuffle edilir. 
    7. Seçilmiş model, config ayarları ile birlikte oluşturulur.
    8. Trainer oluşturulur, `train_loader` ve `val_loader` kullanılarak model eğitimi başlatılır.
    9. Modelin başarısı `test_loader` üzerinde test edilir.
    10. Test sonuçları ve model config bilgisi checkpointe kaydedilir.