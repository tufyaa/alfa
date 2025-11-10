# RAG Hackathon Pipeline

Полный baseline для retrieval-задачи: для каждого вопроса из `questions_clean.csv` нужно вернуть топ‑5 релевантных документов из `websites_updated.csv`. Качество оценивается метрикой Hit@5.

## Данные
- `questions_clean.csv`: столбцы `q_id`, `query`.
- `websites_updated.csv`: `web_id`, `url`, `kind`, `title`, `text`.
- Опционально `qrels.csv`: разметка (пары `q_id`, `web_id`) для локальной оценки.

## Установка
```bash
python -m venv venv
venv\Scripts\activate          # Windows (PowerShell)
# source venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

## Основной сценарий
1. **Построить индекс**
   ```bash
   python scripts/build_index.py --websites data/websites_updated.csv --outdir artifacts/
   ```
2. **Получить топ‑5 web_id для вопросов**
   ```bash
   python scripts/retrieve_top5.py \
       --questions data/questions_clean.csv \
       --index artifacts/ \
       --out submit/raw_top5.parquet
   ```
3. **Собрать сабмит**
   ```bash
   python scripts/make_submission.py \
       --questions data/questions_clean.csv \
       --index artifacts/ \
       --out submit/submit.csv
   ```
Все CLI-скрипты принимают пути к данным и параметры пайплайна через флаги. Используются только открытые библиотеки и Python ≥ 3.10. Эмбеддинги строятся моделью `ai-forever/ru-en-RoSBERTa`.

## Формат сабмита
`submit.csv` c колонками:
- `q_id` — идентификатор вопроса.
- `web_list` — строка со списком ровно из пяти целых значений `web_id`, например `[935, 687, 42, 11, 58]`.

## Проверка Hit@5 (опционально)
При наличии `data/qrels.csv`:
```bash
python scripts/evaluate_hit5.py --pred submit/submit.csv --qrels data/qrels.csv
```

## Тесты
```bash
pytest
```

## Что внутри
- Препроцессинг HTML, нормализация текста, чанкинг с перекрытием.
- Эмбеддинги SentenceTransformer (`ai-forever/ru-en-RoSBERTa`) + FAISS IndexFlatIP.
- Гибридный скоринг: dense ANN + BM25 по всему корпусу.
- CLI-скрипты, модульные тесты и README с инструкциями.
