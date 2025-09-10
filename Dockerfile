FROM python:3.8-slim

WORKDIR /app

COPY pyproject.toml poetry.lock ./
COPY app ./app
COPY models ./models

RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
