[نظرة عامة](#overview)[خريطة المشروع](#pipeline)[البيانات](#data)[المعمارية](#model)[التدريب](#train)[التقييم](#eval)[الأخطاء](#errors)[التحسينات](#next)[التشغيل](#run)

Deep Learning · Medical Imaging

# Pneumonia Detection من أشعة الصدر باستخدام CNN

مشروع Computer Vision بيصنّف صورة Chest X-Ray لحالتين: PNEUMONIA أو NORMAL، باستخدام شبكة التفافية (Convolutional Neural Network) مبنية بـ Keras على صور رمادية بحجم 150×150.

**5,216**صور التدريب Train

**624**صور الاختبار Test

**1.25M**عدد الـ Parameters

**73.6%**دقة الاختبار Accuracy

Chest X-ray sample from the dataset

## نظرة عامة Overview

الفكرة باختصار: نحمّل الصور، نجهزها، ندرّب شبكة التفافية، ونقيس أداءها على بيانات لم ترها قبل كده.

**الهدف Goal**

تصنيف ثنائي Binary Classification: الفئة 0 = Pneumonia، والفئة 1 = Normal.

**المدخل Input**

صورة أشعة صدر Grayscale بعد تغيير الحجم لـ 150×150×1 وقسمتها على 255.

**المخرج Output**

احتمال واحد من Sigmoid؛ لو أكبر من 0.5 يتصنف Normal.

PythonTensorFlow / KerasOpenCVNumPyPandasscikit-learnMatplotlibSeaborn

## خريطة المشروع Pipeline Map

مسار البيانات من أول الصورة الخام لحد المقاييس النهائية.

## البيانات Dataset

هيكل المجلدات train / test / val، وكل مجلد فيه فئتين: PNEUMONIA و NORMAL. الرسوم التالية خارجة مباشرة من الـ Notebook.

Dataset split pie

توزيع النسب: Train 89.1% و Test 10.7% و Validation 0.3%.

Dataset split bar

الأعداد الفعلية: 5216 و 624 و 16 صورة.

Class balance

توزيع الفئات في Train: Pneumonia حوالي 3 أضعاف Normal (عدم توازن).

Pneumonia

PNEUMONIA: عينة عشوائية.

Normal

NORMAL: عينة عشوائية.

Samples grid

شبكة عينات: 9 صور من كل فئة.

**جودة البيانات:** فحص count_corrupted_images لقى صورتين غير قابلتين للقراءة في Train (من 5218) وصورتين في Val (من 18)، فبقت المحمّلة فعليًا 5216 و 16.

### التجهيز والتدعيم Preprocessing & Augmentation

**التجهيز**

`cv2.IMREAD_GRAYSCALE` ثم `resize(150,150)` ثم القسمة على 255 ثم `reshape(-1,150,150,1)`.

**ImageDataGenerator**

`rotation 30` · `shift 0.1` · `zoom 0.2` · `horizontal_flip`.

## معمارية النموذج Architecture

خمس كتل التفافية، كل كتلة: Conv → (Dropout) → BatchNorm → MaxPool. كل ما نعمّق، العرض المكاني بيقل وعدد الفلاتر بيزيد.

**الإجمالي**

1,246,401 parameter (حوالي 4.75 MB)، منها 1,088 غير قابلة للتدريب (BatchNorm).

**الإعدادات**

rmsprop · binary_crossentropy · batch 32 · 18 epochs · ReduceLROnPlateau(factor 0.3).

## التدريب Training

دقة التدريب وصلت تقريبًا لـ 100%، لكن دقة الـ Validation ثابتة عند 50%.

Training curves

منحنيات الدقة والخسارة (أخضر = تدريب، أحمر = Validation).

Train**99.9%**

Validation**50.0%**

Test**73.6%**

| Epoch | Train Acc | Val Acc | Val Loss | Learning Rate |
| --- | --- | --- | --- | --- |
| 1 | 88.63% | 50.00% | 16.75 | 1e-3 |
| 5 | 99.19% | 50.00% | 8.02 | 3e-4 |
| 10 | 99.90% | 50.00% | 5.38 | 8.1e-6 |
| 18 | 99.90% | 50.00% | 5.39 | 1e-6 |

## التقييم Evaluation

النتائج على 624 صورة اختبار. الفئة الموجبة في مقاييس Precision / Recall / F1 هي Normal (الفئة 1).

Metrics

Accuracy 73.6% · Precision 100% · Recall 29.5% · F1 45.5%.

### مصفوفة الالتباس Confusion Matrix

Pred: Pneumonia

Pred: Normal

Actual\
Pneumonia

390

0

Actual\
Normal

165

69

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Pneumonia (0) | 0.70 | 1.00 | 0.83 | 390 |
| Normal (1) | 1.00 | 0.29 | 0.46 | 234 |
| Macro avg | 0.85 | 0.65 | 0.64 | 624 |

النموذج بيلتقط كل حالات الالتهاب (Recall 1.00) لكنه بيصنّف أغلب الصور السليمة كالتهاب: 165 من 234.

## تحليل الأخطاء Error Analysis

Correct predictions

تنبؤات صحيحة (Predicted = Actual).

Incorrect predictions

تنبؤات خاطئة: Predicted 0 بينما Actual 1.

## الملاحظات والتحسينات Findings & Next Steps

خريطة سبب ونتيجة وحل، مبنية على اللي ظهر في الـ Notebook.

#### الملاحظة

#### السبب المحتمل

#### الحل المقترح

Val Acc ثابتة 50%

مجموعة Validation فيها 16 صورة فقط

اقتطاع 15-20% من Train كـ Validation مع stratify

Train 99.9% مقابل Test 73.6%

datagen اتطبّق على الـ Validation فقط، والتدريب بدون Augmentation

استخدام datagen.flow(x_train, y_train) في fit

Recall لـ Normal = 0.29

عدم توازن الفئات (Pneumonia حوالي 3 أضعاف) وتحيّز للفئة الأكبر

class_weight وضبط الـ threshold من منحنى ROC

لا يوجد حفظ لأفضل نموذج

18 epoch ثابتة بدون إيقاف مبكر

EarlyStopping و ModelCheckpoint

خطوات إضافية: Transfer Learning (ResNet / DenseNet / EfficientNet)، رسم Grad-CAM لتوضيح مناطق القرار، وقياس ROC-AUC.

## التشغيل Getting Started

```
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn tensorflow

chest_xray/
  train/  PNEUMONIA/  NORMAL/
  test/   PNEUMONIA/  NORMAL/
  val/    PNEUMONIA/  NORMAL/
```

عدّل مسارات `get_data(...)` في الـ Notebook لتشير لمكان المجلد عندك، ثم شغّل الخلايا بالترتيب.

المشروع لأغراض تعليمية وبحثية فقط، ولا يصلح كأداة تشخيص طبي.