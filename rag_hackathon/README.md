# RAG Hackathon Baseline

Этот проект реализует полнофункциональный retrieval-пайплайн для хакатона: по каждому вопросу необходимо вернуть топ-5 релевантных веб-страниц, а метрика качества — Hit@5.

## Данные
- `questions_clean.csv`: колонки `q_id`, `query`.
- `websites_updated.csv`: колонки `web_id`, `url`, `kind`, `title`, `text`.
- (опционально) `qrels.csv`: разметка релевантности для расчёта метрики.

## Установка
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Основной сценарий запуска
1. **Построить индекс**
   ```bash
   python scripts/build_index.py --websites data/websites_updated.csv --outdir artifacts/
   ```
2. **Извлечь топ-5 кандидатов**
   ```bash
   python scripts/retrieve_top5.py --questions data/questions_clean.csv --websites data/websites_updated.csv --index artifacts/ --out submit/raw_top5.parquet
   ```
3. **Сформировать сабмит**
   ```bash
   python scripts/make_submission.py --questions data/questions_clean.csv --websites data/websites_updated.csv --index artifacts/ --out submit/submit.csv
   ```

Формат сабмита: CSV с колонками `q_id` и `web_list`, где `web_list` — строка вида `[id1, id2, id3, id4, id5]`.

## Дополнительно: метрика Hit@5
Если доступна разметка `qrels.csv`, можно оценить качество:
```bash
python scripts/evaluate_hit5.py --pred submit/submit.csv --qrels data/qrels.csv
```

## Требования
- Только Open Source библиотеки.
- Python 3.10 или новее.
- Эмбеддинги строятся моделью `ai-forever/ru-en-RoSBERTa` из `sentence-transformers`.
