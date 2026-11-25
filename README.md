<p align="center"><img src="assets/banner.svg" alt="Pneumonia Detection banner" width="100%"></p>

<div dir="rtl">

مشروع **Computer Vision** بيصنّف صورة **Chest X-Ray** لحالتين: `PNEUMONIA` أو `NORMAL`، باستخدام شبكة التفافية (**Convolutional Neural Network**) مبنية بـ **Keras** على صور رمادية بحجم 150×150.

| الصور | القيمة |
|---|---|
| Train | 5,216 |
| Test | 624 |
| Validation | 16 |
| Parameters | 1,246,401 |
| Test Accuracy | 73.6% |

## المحتويات

[نظرة عامة](#نظرة-عامة-overview) | [خريطة المشروع](#خريطة-المشروع-pipeline) | [البيانات](#البيانات-dataset) | [المعمارية](#معمارية-النموذج-architecture) | [التدريب](#التدريب-training) | [التقييم](#التقييم-evaluation) | [الأخطاء](#تحليل-الأخطاء-error-analysis) | [التحسينات](#الملاحظات-والتحسينات-next-steps) | [التشغيل](#التشغيل-getting-started)

## نظرة عامة (Overview)

| | الوصف |
|---|---|
| **الهدف (Goal)** | تصنيف ثنائي (Binary Classification): الفئة 0 = Pneumonia، والفئة 1 = Normal |
| **المدخل (Input)** | صورة Grayscale بحجم 150×150×1 بعد القسمة على 255 |
| **المخرج (Output)** | احتمال واحد من Sigmoid؛ لو أكبر من 0.5 يتصنف Normal |
| **الأدوات (Stack)** | Python, TensorFlow/Keras, OpenCV, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn |

## خريطة المشروع (Pipeline)

<p align="center"><img src="assets/pipeline.svg" alt="Project pipeline" width="100%"></p>

## البيانات (Dataset)

هيكل المجلدات `train / test / val`، وكل مجلد فيه فئتين: `PNEUMONIA` و `NORMAL`. الرسوم التالية خارجة من الـ Notebook مباشرة.

<p align="center">
<img src="assets/split_pie.jpg" alt="Dataset split pie" width="32%">
<img src="assets/split_bar.jpg" alt="Dataset split counts" width="32%">
<img src="assets/class_balance.jpg" alt="Class balance" width="32%">
</p>

الأعداد: Train 5,216 (89.1%) و Test 624 (10.7%) و Validation 16 (0.3%). وفي Train فئة Pneumonia حوالي 3 أضعاف Normal، يعني البيانات غير متوازنة.

<p align="center">
<img src="assets/xray_pneumonia.jpg" alt="Pneumonia sample" width="48%">
<img src="assets/xray_normal.jpg" alt="Normal sample" width="48%">
</p>

<p align="center"><img src="assets/samples_grid.jpg" alt="Samples grid" width="90%"></p>

> **جودة البيانات:** فحص `count_corrupted_images` لقى صورتين غير قابلتين للقراءة في Train (من 5218) وصورتين في Val (من 18)، فبقت المحمّلة فعليًا 5,216 و 16.

### التجهيز والتدعيم (Preprocessing & Augmentation)

| المرحلة | التفاصيل |
|---|---|
| التجهيز | `cv2.IMREAD_GRAYSCALE` ثم `resize(150,150)` ثم القسمة على 255 ثم `reshape(-1,150,150,1)` |
| ImageDataGenerator | `rotation 30`, `shift 0.1`, `zoom 0.2`, `horizontal_flip` |

## معمارية النموذج (Architecture)

خمس كتل التفافية، كل كتلة: `Conv → (Dropout) → BatchNorm → MaxPool`. كل ما نعمّق، العرض المكاني بيقل وعدد الفلاتر بيزيد.

<p align="center"><img src="assets/architecture.svg" alt="CNN architecture" width="100%"></p>

- إجمالي الـ Parameters: **1,246,401** (حوالي 4.75 MB)، منها 1,088 غير قابلة للتدريب (BatchNorm).
- الإعدادات: `rmsprop` | `binary_crossentropy` | `batch 32` | `18 epochs` | `ReduceLROnPlateau(factor 0.3)`.

## التدريب (Training)

دقة التدريب وصلت تقريبًا لـ 100%، لكن دقة الـ Validation ثابتة عند 50%.

<p align="center"><img src="assets/accuracy_gap.svg" alt="Accuracy gap" width="80%"></p>

<p align="center"><img src="assets/training_curves.jpg" alt="Training curves" width="100%"></p>

| Epoch | Train Acc | Val Acc | Val Loss | Learning Rate |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 88.63% | 50.00% | 16.75 | 1e-3 |
| 5 | 99.19% | 50.00% | 8.02 | 3e-4 |
| 10 | 99.90% | 50.00% | 5.38 | 8.1e-6 |
| 18 | 99.90% | 50.00% | 5.39 | 1e-6 |

## التقييم (Evaluation)

النتائج على 624 صورة اختبار. الفئة الموجبة في مقاييس Precision / Recall / F1 هي **Normal** (الفئة 1).

<p align="center"><img src="assets/metrics.jpg" alt="Performance metrics" width="55%"></p>

**Confusion Matrix**

| | Pred: Pneumonia | Pred: Normal |
|:-:|:-:|:-:|
| **Actual: Pneumonia** | 390 | 0 |
| **Actual: Normal** | 165 | 69 |

| Class | Precision | Recall | F1 | Support |
|:-:|:-:|:-:|:-:|:-:|
| Pneumonia (0) | 0.70 | 1.00 | 0.83 | 390 |
| Normal (1) | 1.00 | 0.29 | 0.46 | 234 |
| Macro avg | 0.85 | 0.65 | 0.64 | 624 |

النموذج بيلتقط كل حالات الالتهاب (Recall 1.00) لكنه بيصنّف أغلب الصور السليمة كالتهاب: 165 من 234.

## تحليل الأخطاء (Error Analysis)

<p align="center">
<img src="assets/correct_predictions.jpg" alt="Correct predictions" width="40%">
<img src="assets/wrong_predictions.jpg" alt="Wrong predictions" width="40%">
</p>
<p align="center">يمين: تنبؤات صحيحة | يسار: تنبؤات خاطئة (Predicted 0 بينما Actual 1)</p>

## الملاحظات والتحسينات (Next Steps)

| الملاحظة | السبب المحتمل | الحل المقترح |
|---|---|---|
| Val Acc ثابتة 50% | مجموعة Validation فيها 16 صورة فقط | اقتطاع 15-20% من Train كـ Validation مع `stratify` |
| Train 99.9% مقابل Test 73.6% | `datagen` اتطبّق على الـ Validation فقط، والتدريب بدون Augmentation | استخدام `datagen.flow(x_train, y_train)` في `fit` |
| Recall لـ Normal = 0.29 | عدم توازن الفئات وتحيّز للفئة الأكبر | `class_weight` وضبط الـ threshold من منحنى ROC |
| لا يوجد حفظ لأفضل نموذج | 18 epoch ثابتة بدون إيقاف مبكر | `EarlyStopping` و `ModelCheckpoint` |

خطوات إضافية: Transfer Learning (ResNet / DenseNet / EfficientNet)، رسم Grad-CAM لتوضيح مناطق القرار، وقياس ROC-AUC.

## التشغيل (Getting Started)

</div>

```bash
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn tensorflow
```

```text
chest_xray/
  train/  PNEUMONIA/  NORMAL/
  test/   PNEUMONIA/  NORMAL/
  val/    PNEUMONIA/  NORMAL/
```

<div dir="rtl">

عدّل مسارات `get_data(...)` في الـ Notebook لتشير لمكان المجلد عندك، ثم شغّل الخلايا بالترتيب.

> المشروع لأغراض تعليمية وبحثية فقط، ولا يصلح كأداة تشخيص طبي.

</div>
