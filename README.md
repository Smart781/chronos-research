# Исследование эффективности модели Chronos для прогнозирования временных рядов

## О проекте

Данный репозиторий содержит материалы исследовательской работы по оценке эффективности модели Chronos (Amazon, 2024) в zero-shot и fine-tuned режимах на 10 датасетах из различных предметных областей.

**Основные направления исследования:**
- Zero-shot прогнозирование на 10 датасетах
- Fine-tuning на отдельных датасетах и их комбинациях
- Сравнение с классическими методами (ARIMA, ETS, Theta, Seasonal Naive)
- Анализ влияния характеристик рядов на качество прогнозов
- Эксперименты с модифицированной архитектурой (Ordinal Head)

## Структура репозитория
```
notebooks/
 energy_forecast_analysis.ipynb # Zero-shot на ряде H1
 Comparing.ipynb # Сравнение с бейзлайнами
 Theta,_DeepAR,_PatchTST.ipynb # Расширенное сравнение
 chronos-datasets.ipynb # Zero-shot на 10 датасетах
 Table.ipynb # AutoGluon сравнение
 imp_fine-tuning.ipynb # Fine-tuning на 3 датасетах
 combined_fine-tuning.ipynb # Комбинированный FT
 compare_models.ipynb # Сравнение с Ordinal Head

 results/
 comparison_results.csv # Итоговые таблицы
 figures/ # Графики и визуализации

 my_chronos/ # Модифицированная архитектура
 ordinal_head.py # Proportional Odds Model
 train_ordinal.py # Обучение ordinal модели
 compare_models.py # Сравнение с оригиналом

 README.md # Описание проекта
 requirements.txt # Зависимости
```

## Запуск
```
python download_datasets.py # скачать датасеты
python train_ordinal.py # запустить обучение
python compare_models.py --all # сравнить модели
python train_ordinal.py --dataset m4_daily --max-series 200 --n-tokens 130 --epochs 50 # обучение с параметрами
```