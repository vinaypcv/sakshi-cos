FROM python:3.12-slim
WORKDIR /app
COPY . .
ENV PYTHONPATH=/app/src
RUN pip install --no-cache-dir -e ".[dev]"
RUN python -m pytest -q
ENTRYPOINT ["python", "-m", "sakshi"]
CMD ["--out", "runs", "all"]
