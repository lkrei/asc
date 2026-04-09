# Детальный анализ результатов и ошибок

## 1. Итоговая сводка

| Модель | Accuracy | Balanced Accuracy | Macro F1 | Time (min) |
|---|---:|---:|---:|---:|
| `convnext_small_finetune` | 0.7804 | 0.7739 | 0.7746 | 142.7 |
| `swin_s_finetune` | 0.7693 | 0.7610 | 0.7612 | 130.5 |
| `vit_b16_finetune` | 0.7524 | 0.7455 | 0.7464 | 166.5 |
| `resnet50_finetune` | 0.7186 | 0.7097 | 0.7131 | 84.1 |
| `efficientnet_b0_finetune` | 0.6933 | 0.6833 | 0.6824 | 68.7 |

Лучший single-model результат показала модель `convnext_small_finetune`: `accuracy=0.7804`, `balanced_accuracy=0.7739`, `macro_f1=0.7746`.
Ансамбль улучшил результат до `accuracy=0.7934`, `balanced_accuracy=0.7841`, `macro_f1=0.7856`.

## 2. Самые сильные классы у лучшей модели

Классы с максимальным `F1-score` у лучшей одиночной модели:

- `Ancient Egyptian architecture`: `F1=0.9764`, `recall=1.0000`, `support=62`
- `Novelty architecture`: `F1=0.9558`, `recall=0.9310`, `support=58`
- `Achaemenid architecture`: `F1=0.9391`, `recall=0.9000`, `support=60`
- `Gothic architecture`: `F1=0.9320`, `recall=0.9412`, `support=51`
- `Romanesque architecture`: `F1=0.8421`, `recall=0.8696`, `support=46`
- `Art Nouveau architecture`: `F1=0.8352`, `recall=0.8172`, `support=93`
- `Deconstructivism`: `F1=0.8333`, `recall=0.7843`, `support=51`
- `Greek Revival architecture`: `F1=0.8263`, `recall=0.8734`, `support=79`

## 3. Самые трудные классы у лучшей модели

Классы с минимальным `F1-score` у лучшей одиночной модели:

- `Edwardian architecture`: `F1=0.5122`, `precision=0.5250`, `recall=0.5000`
- `Georgian architecture`: `F1=0.5854`, `precision=0.5538`, `recall=0.6207`
- `International style`: `F1=0.6299`, `precision=0.6349`, `recall=0.6250`
- `Bauhaus architecture`: `F1=0.6737`, `precision=0.6809`, `recall=0.6667`
- `Beaux-Arts architecture`: `F1=0.6769`, `precision=0.6769`, `recall=0.6769`
- `Postmodern architecture`: `F1=0.6869`, `precision=0.6800`, `recall=0.6939`
- `Colonial architecture`: `F1=0.7015`, `precision=0.7581`, `recall=0.6528`
- `American craftsman style`: `F1=0.7193`, `precision=0.7069`, `recall=0.7321`

## 4. Классы, которые сильнее всего выиграли относительно ResNet-50


- `Edwardian architecture`: `convnext_small_finetune F1=0.5122` vs `resnet50_finetune F1=0.3250` (прирост `+0.1872`)
- `Colonial architecture`: `convnext_small_finetune F1=0.7015` vs `resnet50_finetune F1=0.5366` (прирост `+0.1649`)
- `American craftsman style`: `convnext_small_finetune F1=0.7193` vs `resnet50_finetune F1=0.6055` (прирост `+0.1138`)
- `Byzantine architecture`: `convnext_small_finetune F1=0.7835` vs `resnet50_finetune F1=0.6739` (прирост `+0.1096`)
- `International style`: `convnext_small_finetune F1=0.6299` vs `resnet50_finetune F1=0.5399` (прирост `+0.0900`)
- `Bauhaus architecture`: `convnext_small_finetune F1=0.6737` vs `resnet50_finetune F1=0.5843` (прирост `+0.0894`)
- `American Foursquare architecture`: `convnext_small_finetune F1=0.7818` vs `resnet50_finetune F1=0.7049` (прирост `+0.0769`)
- `Beaux-Arts architecture`: `convnext_small_finetune F1=0.6769` vs `resnet50_finetune F1=0.6050` (прирост `+0.0719`)

## 5. Глобально самые трудные стили по всем моделям

Ниже перечислены стили, которые в среднем дают наибольшее число ошибок в серии экспериментов.

- `Colonial architecture`: около `144` ошибок по совокупности моделей
- `Beaux-Arts architecture`: около `134` ошибок по совокупности моделей
- `Edwardian architecture`: около `128` ошибок по совокупности моделей
- `Art Deco architecture`: около `121` ошибок по совокупности моделей
- `International style`: около `109` ошибок по совокупности моделей
- `Postmodern architecture`: около `108` ошибок по совокупности моделей
- `Queen Anne architecture`: около `104` ошибок по совокупности моделей
- `Palladian architecture`: около `103` ошибок по совокупности моделей
- `Georgian architecture`: около `102` ошибок по совокупности моделей
- `American craftsman style`: около `94` ошибок по совокупности моделей

## 6. Ранжирование сложности по среднему F1

Наиболее трудные стили по среднему качеству across models:

- `Edwardian architecture`: средний `F1=0.4155`
- `Colonial architecture`: средний `F1=0.5390`
- `Georgian architecture`: средний `F1=0.5610`
- `Beaux-Arts architecture`: средний `F1=0.5732`
- `Postmodern architecture`: средний `F1=0.5894`
- `International style`: средний `F1=0.5924`
- `Bauhaus architecture`: средний `F1=0.5952`
- `American craftsman style`: средний `F1=0.6270`
- `Palladian architecture`: средний `F1=0.6290`
- `American Foursquare architecture`: средний `F1=0.6837`

## 7. Интерпретация результатов

Из полученных метрик следует, что главная сложность задачи сосредоточена не в редких или экзотических стилях как таковых, а в визуально близких архитектурных направлениях. Особенно это видно по классам `Edwardian architecture`, `Colonial architecture`, `Georgian architecture`, `Beaux-Arts architecture` и `International style`.
Сильнее всего современные fine-tuned архитектуры выигрывают у более простого бейзлайна в тех классах, где нужно одновременно учитывать композицию фасада, декоративную насыщенность и локальные структурные признаки.
Ансамбль даёт дополнительный прирост по агрегированным метрикам, что подтверждает комплементарность разных архитектур. Однако для полноценного per-class анализа ансамбля в дальнейшем нужно сохранять и использовать per-sample logits/predictions.

## 8. Что использовать в тексте диплома


- как основную single-model конфигурацию: `convnext_small_finetune`;
- как лучший итоговый практический результат: ансамбль из пяти моделей;
- как центральный аналитический вывод: современные модели лучше всего справляются с ярко выраженными стилями, а основные ошибки концентрируются на исторически и визуально близких направлениях.
